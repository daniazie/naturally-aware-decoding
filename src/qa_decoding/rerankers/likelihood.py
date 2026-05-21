from transformers import PreTrainedModel, PreTrainedTokenizerBase, AutoModelForCausalLM, AutoTokenizer
from typing import List, Literal
from tqdm import tqdm

import torch

from rerankers.base_reranker import BaseReranker

code2name = {
    "zho": "Chinese",
    "fra": "French",
    "deu": "German",
    "msa": "Malay",
    "kor": "Korean",
}
    
class LikelihoodReranker(BaseReranker):
    def __init__(self, model: str | PreTrainedModel, tokenizer: PreTrainedTokenizerBase | None = None, **model_kwargs):
        if isinstance(model, str):
            self.model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model)
        else:
            self.model = model

        super().__init__(
            model,
            tokenizer
        )

    @torch.no_grad()
    def model_forward(self, batch):
        _exclude_keys = {"completion_mask"}
        inputs = {
            k: v.to(self.model.device)
            for k, v in batch.items()
            if not k in _exclude_keys
        }

        logits: torch.Tensor = self.model(**inputs).logits[:, :-1].cpu()
        labels: torch.Tensor = inputs['input_ids'][:, 1:].cpu()
        completion_mask: torch.Tensor = batch['completion_mask'][:, 1:].bool().cpu()

        return logits, labels, completion_mask
    
    
    def I(self, logits: torch.Tensor, labels: torch.Tensor, completion_mask: torch.Tensor):
        logps = logits.log_softmax(dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        return (- logps).masked_fill(~completion_mask, 0)
    
    def H(self, logits: torch.Tensor, labels: torch.Tensor, completion_mask: torch.Tensor | None = None):
        probs = logits.softmax(dim=-1)
        logps = logits.log_softmax(dim=-1)
        entropy = - (probs * logps).sum(dim=-1)
        if completion_mask is not None:
            return entropy.masked_fill(~completion_mask, 0)
        return entropy
    
    def perplexity(self, logits: torch.Tensor, labels: torch.Tensor):
        entropy = self.H(logits, labels)
        return torch.exp(entropy)
    
    def compute_fn(self, metric: Literal["entropy", "surprisal", "perplexity"] = "entropy"):
        if metric == "entropy":
            return self.H
        if metric == "surprisal":
            return self.I
        if metric == "perplexity":
            return self.perplexity
    
    def rerank(self, srcs: List[str], mts: List[List[str]], tgt_lang: str, metric: Literal["entropy", "surprisal", "perplexity"] = "entropy", return_score: bool = False, normalise_scores: bool = False):
        batches = self.prepare_data(srcs, mts, tgt_lang)
        compute = self.compute_fn(metric)
        preds = []
        for i, batch in enumerate(tqdm(batches, desc="Reranking...")):
            logits, labels, completion_mask = self.model_forward(batch)
            scores = compute(logits, labels)
            if normalise_scores:
                if metric == 'entropy':
                    scores = scores / scores.max(-1)
                elif metric == 'surprisal':
                    scores = scores / (- (scores.min(-1)))
                elif metric == 'perplexity':
                    scores = ((torch.log(torch.tensor(self.model.vocab_size)) - torch.log(scores)) / torch.log(torch.tensor(self.model.vocab_size))).masked_fill(~completion_mask, 0)
            
            scores = scores.sum(dim=1) / completion_mask.sum(dim=1)
            best_idx = scores.argmax().item()
            best_mt = mts[i][best_idx]

            best_score = scores[best_idx].item()
            res = {
                "src": srcs[i],
                "mt": best_mt
            }

            if return_score:
                res.update({"score": best_score})
            preds.append(res)
        return preds