from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from typing import List, Literal
from tqdm import tqdm
from functools import partial

import torch
import numpy as np
import os

from segmenter import Segmenter
from rerankers.base_reranker import BaseReranker

code2name = {
    "zho": "Chinese",
    "fra": "French",
    "deu": "German",
    "msa": "Malay",
    "kor": "Korean",
}

class RatioReranker(BaseReranker):
    def __init__(self, model_path: str | os.PathLike, tokenizer: str | os.PathLike | None = None, segmenter: Segmenter | None = None, granularity: Literal['token', 'sequence', 'segment'] = 'token', metric_type: Literal['logps', 'entropy', 'perplexity'] = 'logps', **model_kwargs):
        positive_model = AutoModelForCausalLM.from_pretrained(f"{model_path}/positive", **model_kwargs)
        negative_model = AutoModelForCausalLM.from_pretrained(f"{model_path}/negative", **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/positive")
        super().__init__(
            positive_model,
            tokenizer
            )
        self.granularity = granularity
        if self.granularity == 'segment' and segmenter is None:
            self.segmenter = Segmenter("Qwen/Qwen2.5-0.5B", **model_kwargs)
        else:
            self.segmenter = segmenter
        if metric_type == 'logps':
            self.compute_positive = partial(self.compute_logps, model=positive_model)
            self.compute_negative = partial(self.compute_logps, model=negative_model)
        elif metric_type == 'entropy':
            self.compute_positive = partial(self.compute_entropy, model=positive_model)
            self.compute_negative = partial(self.compute_entropy, model=negative_model)
        
        self.use_perplexity = metric_type == 'perplexity'

    @torch.no_grad()
    def compute_logits(self, inputs, model: PreTrainedModel):
        outputs = model(**inputs)
        logits: torch.Tensor = outputs.logits[:, :-1].cpu()
        return logits

    def compute_logps(self, example, model: PreTrainedModel):
        _exclude_keys = {"completion_mask", "segment_mask"}
        inputs = {
            k: v.to(model.device)
            for k, v in example.items()
            if not k in _exclude_keys
        }
        
        logits = self.compute_logits(inputs=inputs, model=model)
        completion_mask: torch.Tensor = example["completion_mask"]
        labels: torch.Tensor = inputs['input_ids'][:, 1:].cpu()
        completion_mask = completion_mask[:, 1:].bool()
        
        logps = logits.log_softmax(dim=-1)
        logps = logps.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        logps = logps.masked_fill(~completion_mask, 0)

        if self.granularity == "sequence":
            logps = logps.sum(dim=1) / completion_mask.sum(dim=1)

        return logps
    
    def compute_entropy(self, example, model: PreTrainedModel):
        _exclude_keys = {"completion_mask", "segment_mask"}
        inputs = {
            k: v.to(model.device)
            for k, v in example.items()
            if not k in _exclude_keys
        }
        
        logits = self.compute_logits(inputs=inputs, model=model)
        completion_mask: torch.Tensor = example["completion_mask"].clone()
        completion_mask = completion_mask[:, 1:].bool()

        entropy = - (logits.softmax(dim=-1) * logits.log_softmax(dim=-1)).sum(dim=-1)
        entropy = entropy.masked_fill(~completion_mask, 0)
        return entropy
        
    def per_segment_logps(self, pos_tensor: torch.Tensor, neg_tensor: torch.Tensor, seg_mask: torch.Tensor):
        rewards = []
        segment_pos = []
        segment_neg = []
        for i, t in enumerate(seg_mask):
            if t < 0:
                continue
            if t == 0:
                segment_pos.append(pos_tensor[i])
                segment_neg.append(neg_tensor[i])
            if t == 1:
                if segment_pos and segment_neg:
                    logps_pos = torch.tensor(segment_pos).mean()
                    logps_neg = torch.tensor(segment_neg).mean()
                    if self.use_perplexity:
                        ppl_pos = torch.exp(-logps_pos)
                        ppl_neg = torch.exp(-logps_neg)
                        rewards.append(ppl_pos - ppl_neg)
                    else:
                        logps_ratios = logps_pos - logps_neg
                        rewards.append(logps_ratios)
                    segment_pos = [pos_tensor[i]]
                    segment_neg = [neg_tensor[i]]
            if not i == len(seg_mask) - 1:
                if seg_mask[i+1] < 0:
                    logps_pos = torch.tensor(segment_pos).mean()
                    logps_neg = torch.tensor(segment_neg).mean()
                    if self.use_perplexity:
                        ppl_pos = torch.exp(-logps_pos)
                        ppl_neg = torch.exp(-logps_neg)
                        rewards.append(ppl_pos - ppl_neg)
                    else:
                        logps_ratios = logps_pos - logps_neg
                        rewards.append(logps_ratios)
            else:
                if t >= 0:
                    logps_pos = torch.tensor(segment_pos).mean()
                    logps_neg = torch.tensor(segment_neg).mean()
                    if self.use_perplexity:
                        ppl_pos = torch.exp(-logps_pos)
                        ppl_neg = torch.exp(-logps_neg)
                        rewards.append(ppl_pos - ppl_neg)
                    else:
                        logps_ratios = logps_pos - logps_neg
                        rewards.append(logps_ratios)
        try:
            rewards = torch.stack(rewards).mean()
            return rewards
        except:
            print(rewards, len(rewards))
            raise Exception

    def compute_rewards(self, log_lklh_positive: torch.Tensor, log_lklh_negative: torch.Tensor, c_masks: torch.Tensor | None = None, seg_masks: torch.Tensor | None = None):
        if self.granularity == 'segment':
            seg_masks = seg_masks[:, 1:]
            if not c_masks.shape == log_lklh_positive.shape:
                c_masks = c_masks[:, 1:]
            per_segment_rewards = [self.per_segment_logps(logps_pos, logps_neg, seg_mask) for logps_pos, logps_neg, seg_mask in zip(log_lklh_positive, log_lklh_negative, seg_masks)]
            rewards = torch.stack(per_segment_rewards)
            return rewards
        if self.use_perplexity and not self.granularity == 'token':
            ppl_positive = torch.exp(-(log_lklh_positive.sum(dim=1)  / c_masks.sum(dim=1)))
            ppl_negative = torch.exp(-(log_lklh_negative.sum(dim=1) / c_masks.sum(dim=1)))

            rewards = ppl_positive - ppl_negative
            return rewards
        rewards = log_lklh_positive - log_lklh_negative
        if self.granularity == 'token':
            rewards = rewards.sum(dim=1) / c_masks.sum(dim=1)
        return rewards

    def _score(self, batch: dict):
        log_lklh_positive = self.compute_positive(batch)
        log_lklh_negative = self.compute_negative(batch)

        c_mask = batch['completion_mask']
        seg_mask = batch.get("segment_mask")

        rewards = self.compute_rewards(log_lklh_positive, log_lklh_negative, c_mask, seg_mask)
        return rewards
    
    def sigmoid(self, score):
        return 1 / (1 + np.exp(-score))
    
    def rerank(self, srcs: List[str], mts: List[List[str]], tgt_lang: str, return_score: bool = False, normalise_scores: bool = False):
        batches = self.prepare_data(srcs, mts, code2name[tgt_lang])
        if self.granularity == 'segment':
            seg_masks = self.segmenter.compute(srcs, mts, code2name[tgt_lang])
            for i, batch in enumerate(batches):
                batch.update({"segment_mask": seg_masks[i]})
        results = []
        for i, batch in enumerate(tqdm(batches, total=len(batches), desc="Reranking...")):
            rewards = self._score(batch)
            if normalise_scores:
                rewards = 1 - rewards.sigmoid()
                best = rewards.argmax().item()
            else:
                best = rewards.argmin().item()
            res = {
                "src": srcs[i],
                "mt": mts[i][best],
            }

            if return_score:
                score = rewards[best].item()
                res.update({"score": score})
            results.append(res)
        return results