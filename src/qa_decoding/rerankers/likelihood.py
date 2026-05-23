from transformers import PreTrainedModel, PreTrainedTokenizerBase, AutoModelForCausalLM, AutoTokenizer
from typing import List, Literal
from functools import partial
from tqdm import tqdm

import torch

from rerankers.base_reranker import BaseReranker
import json

code2name = {
    "zho": "Chinese",
    "fra": "French",
    "deu": "German",
    "msa": "Malay",
    "kor": "Korean",
}
    
class LikelihoodReranker(BaseReranker):
    def __init__(self, model: str | PreTrainedModel, tokenizer: PreTrainedTokenizerBase | None = None, per_segment_eval: bool = False, **model_kwargs):
        if isinstance(model, str):
            self.model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model)
        else:
            self.model = model

        super().__init__(
            model,
            tokenizer
        )

        self.per_segment_eval = per_segment_eval

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
    
    def per_segment(self, tensor: torch.Tensor, mask: torch.Tensor):
        rewards = []
        segment = []
        for i, m in enumerate(mask):
            if m < 0:
                continue
            if (i == len(mask) - 1):
                segment.append(tensor[i])
                segment = torch.stack(segment)
                segment_rewards = segment.mean()
                rewards.append(segment_rewards)
                continue
            is_end = (m == 1) or (mask[i+1] < 0)
            segment.append(tensor[i])
            if is_end:
                segment = torch.stack(segment)
                segment_rewards = segment.mean()
                rewards.append(segment_rewards)
                segment = []
                continue
        rewards = torch.stack(rewards)
        return rewards.sum()

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
    
    def perplexity(self, logits: torch.Tensor, labels: torch.Tensor, completion_mask: torch.Tensor | None = None):
        entropy = self.H(logits, labels)
        return torch.exp(entropy)
    
    def compute_fn(self, metric: Literal["entropy", "surprisal", "perplexity"] = "entropy"):
        if metric == "entropy":
            return self.H
        if metric == "surprisal":
            return self.I
        if metric == "perplexity":
            return self.perplexity
        
    def compute_mask(self, tensors: torch.Tensor, c_masks: torch.Tensor):
        minima_masks = []
        for tensor in tensors:
            minima_mask = []
            for i in range(len(tensor)):
                if i + 1 == len(tensor):
                    minima_mask.append(tensor[i] < tensor[i-1])
                elif i == 0:
                    minima_mask.append(tensor[i] < tensor[i+1])
                else:
                    minima_mask.append(tensor[i] < tensor[i-1] and tensor[i] < tensor[i+1])
            minima_masks.append(torch.stack(minima_mask))
        minima_masks = torch.stack(minima_masks)

        t_min_seq = (tensors.masked_fill(~minima_masks, 0).masked_fill(~c_masks, 0).sum(dim=-1) / c_masks.sum(dim=-1)).unsqueeze(-1)
        t_min_seg = (tensors.masked_fill(~minima_masks, 0).masked_fill(~c_masks, 0).sum(dim=-1) / minima_masks.int().masked_fill(~c_masks, 0).sum(dim=-1)).unsqueeze(-1)
        
        t_min = torch.amin(torch.cat((t_min_seq, t_min_seg)), keepdim=True)

        masks = minima_masks & (tensors < t_min)
        return masks
    
    def rerank(self, srcs: List[str], mts: List[List[str]], tgt_lang: str, metric: Literal["entropy", "surprisal", "perplexity"] = "entropy", return_score: bool = False, normalise_scores: bool = False):
        batches = self.prepare_data(srcs, mts, tgt_lang)
        compute = self.compute_fn(metric)
        preds = []
        for i, batch in enumerate(tqdm(batches, desc="Reranking...")):
            logits, labels, completion_mask = self.model_forward(batch)
            scores = compute(logits, labels, completion_mask)
            if normalise_scores:
                if metric == 'entropy':
                    scores = scores / scores.amax(-1, keepdim=True)
                elif metric == 'surprisal':
                    scores = scores / ((scores.amax(-1, keepdim=True)))
                elif metric == 'perplexity':
                    scores = ((torch.log(torch.tensor(self.model.vocab_size)) - torch.log(scores)) / torch.log(torch.tensor(self.model.vocab_size))).masked_fill(~completion_mask, 0)
            
            if self.per_segment_eval:
                seg_masks = self.compute_mask(scores, completion_mask)
                scores = [self.per_segment(score, seg_mask) for score, seg_mask in zip(scores.masked_fill(~completion_mask, -100), seg_masks)]
                scores = torch.stack(scores)
            else:
                scores = scores.sum(dim=-1) / completion_mask.sum(dim=-1)

            _batch_mts = mts[i]

            min_idx = scores.argmin().item()
            max_idx = scores.argmax().item()
            max_mt = _batch_mts[max_idx]
            min_mt = _batch_mts[min_idx]

            max_score = scores[max_idx].item()
            min_score = scores[min_idx].item()
            med_score, med_idx = torch.median(scores, -1)
            med = _batch_mts[med_idx.item()]
            _batch_mts = [_batch_mts[i] for i in range(len(_batch_mts)) if (not scores[i] == min_score) and (not scores[i] == max_score)]
            filtered = scores[~(scores == min_score) & ~(scores == max_score)]
            min_scores, min_idxs = torch.topk(filtered, k=min(3, filtered.shape[0]), largest=False)
            max_scores, max_idxs = torch.topk(filtered, k=min(3, filtered.shape[0]))

            _all = [
                {
                    "src": srcs[i],
                    "mt": mt,
                    "score": score.item()
                }
                for mt, score in zip(mts[i], scores)
            ]
            with open(f"outputs.json", "a") as file:
                json.dump(_all, file, indent=2)

            maxes = [
                {
                    "mt": _batch_mts[idx.item()],
                    "score": score.item()
                } 
                for idx, score in zip(max_idxs, max_scores)
            ]

            mins = [
                {
                    "mt": _batch_mts[idx.item()],
                    "score": score.item()
                } 
                for idx, score in zip(min_idxs, min_scores)
            ]

            res = {
                "src": srcs[i],
                "high": maxes,
                "low": mins,
                "median": {
                    "mt": med,
                    "score": med_score.item()
                }
            }

            preds.append(res)
        return preds
    
class SelfReranker(BaseReranker):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, best_of: int, per_segment_eval: bool = False):
        self.per_segment_eval = per_segment_eval
        self.process_sequences = partial(self.process_token_ids, tokenizer=tokenizer, best_of=best_of)
        
    def process_token_ids(self, sequences, input_len, best_of, tokenizer):
        sequences = torch.stack([torch.stack([seq for seq in sequences[i:i+best_of]]) for i in range(0, len(sequences), best_of)])
        srcs = [tokenizer.batch_decode([sequences[i][:input_len] for i in range(0, len(sequences), best_of)])]
        mts = [tokenizer.batch_decode([seq[input_len:] for seq in sequences[i]], skip_special_tokens=True) for i in range(len(sequences))]
        return srcs, mts

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
    
    def perplexity(self, logits: torch.Tensor, labels: torch.Tensor, completion_mask: torch.Tensor | None = None):
        entropy = self.H(logits, labels)
        return torch.exp(entropy)
    
    def per_segment(self, tensor: torch.Tensor, mask: torch.Tensor):
        rewards = []
        segment = []
        for i, m in enumerate(mask):
            if m < 0:
                continue
            if (i == len(mask) - 1):
                segment.append(tensor[i])
                segment = torch.stack(segment)
                segment_rewards = segment.mean()
                rewards.append(segment_rewards)
                continue
            is_end = (m == 1) or (mask[i+1] < 0)
            segment.append(tensor[i])
            if is_end:
                segment = torch.stack(segment)
                segment_rewards = segment.mean()
                rewards.append(segment_rewards)
                segment = []
                continue
        rewards = torch.stack(rewards)
        return rewards.sum()
    
    def compute_fn(self, metric: Literal["entropy", "surprisal", "perplexity"] = "entropy"):
        if metric == "entropy":
            return self.H
        if metric == "surprisal":
            return self.I
        if metric == "perplexity":
            return self.perplexity
        
    def compute_mask(self, tensors: torch.Tensor, c_masks: torch.Tensor):
        minima_masks = []
        for tensor in tensors:
            minima_mask = []
            for i in range(len(tensor)):
                if i + 1 == len(tensor):
                    minima_mask.append(tensor[i] < tensor[i-1])
                elif i == 0:
                    minima_mask.append(tensor[i] < tensor[i+1])
                else:
                    minima_mask.append(tensor[i] < tensor[i-1] and tensor[i] < tensor[i+1])
            minima_masks.append(torch.stack(minima_mask))
        minima_masks = torch.stack(minima_masks)

        t_min_seq = (tensors.masked_fill(~minima_masks, 0).masked_fill(~c_masks, 0).sum(dim=-1) / c_masks.sum(dim=-1)).unsqueeze(-1)
        t_min_seg = (tensors.masked_fill(~minima_masks, 0).masked_fill(~c_masks, 0).sum(dim=-1) / minima_masks.int().masked_fill(~c_masks, 0).sum(dim=-1)).unsqueeze(-1)
        
        t_min = torch.amin(torch.cat((t_min_seq, t_min_seg)), keepdim=True)

        masks = minima_masks & (tensors < t_min)
        return masks
    

    def rerank(self, sequences: torch.Tensor, input_len: int, batches: List[dict[str, torch.Tensor]], metric: Literal["entropy", "surprisal", "perplexity"] = "entropy", return_score: bool = False, normalise_scores: bool = False):
        compute = self.compute_fn(metric)
        preds = []
        srcs, mts = self.process_sequences(sequences, input_len)
        for i, batch in enumerate(tqdm(batches, desc="Reranking...")):
            logits = batch['logits']
            labels = batch['labels']
            completion_mask = batch['completion_mask']
            scores = compute(logits[i], labels, completion_mask)
            if not normalise_scores:
                if metric == 'entropy':
                    scores = scores / scores.amax(-1, keepdim=True)
                elif metric == 'surprisal':
                    scores = scores / (- (scores.amax(-1, keepdim=True)))
                elif metric == 'perplexity':
                    scores = ((torch.tensor(self.model.vocab_size) - scores) / (torch.tensor(self.model.vocab_size) - 1)).masked_fill(~completion_mask, 0)
            
            if self.per_segment_eval:
                seg_masks = self.compute_mask(scores, completion_mask)
                scores = [self.per_segment(score, seg_mask) for score, seg_mask in zip(scores.masked_fill(~completion_mask, -100), seg_masks)]
                scores = torch.stack(scores)
            else:
                scores = scores.sum(dim=-1) / completion_mask.sum(dim=-1)

            min_idx = scores.argmin().item()
            max_idx = scores.argmax().item()
            max_mt = mts[i][max_idx]
            min_mt = mts[i][min_idx]

            max_score = scores[max_idx].item()
            min_score = scores[min_idx].item()
            res = {
                "src": srcs[i],
                "high": {
                    "mt": max_mt,
                    "score": max_score
                },
                "low": {
                    "mt": min_mt,
                    "score": min_score
                }
            }
            preds.append(res)
        return preds
