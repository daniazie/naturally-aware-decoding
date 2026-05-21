from transformers import PreTrainedModel, PreTrainedTokenizerBase, AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Literal
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from comet import load_from_checkpoint, download_model
import warnings

from tqdm.asyncio import tqdm_asyncio
import asyncio
import logging
import torch
import numpy as np
import os

from segmenter import Segmenter
from data_utils import prepare_data

code2name = {
    "zho": "Chinese",
    "fra": "French",
    "deu": "German",
    "msa": "Malay",
    "kor": "Korean",
}

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="pytorch_lightning")

@dataclass
class RatioArgs:
    tgt_lang: str | None = None
    return_score: bool = False
    normalise_scores: bool = False

@dataclass
class LikelihoodArgs:
    tgt_lang: str | None = None
    metric: Literal["entropy", "surprisal", "logprobs", "perplexity"] = "entropy"
    return_score: bool = False
    normalise_scores: bool = False

@dataclass
class CometConfig:
    return_score: bool = False

@dataclass
class RerankerConfig(RatioArgs):
    w_nat: float = 1.0
    w_comet: float = 1.0
    return_score: bool = False
    return_nat: bool = False
    return_comet: bool = False

class RatioReranker:
    def __init__(self, model_dir: str | os.PathLike, tokenizer: str | os.PathLike | None = None, segmenter: Segmenter | None = None, granularity: Literal['token', 'sequence', 'segment'] = 'token', metric_type: Literal['logps', 'entropy', 'perplexity'] = 'logps', **model_kwargs):
        positive_model = AutoModelForCausalLM.from_pretrained(f"{model_dir}/positive", **model_kwargs)
        negative_model = AutoModelForCausalLM.from_pretrained(f"{model_dir}/negative", **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}/positive")
        self.prepare_data = partial(prepare_data, tokenizer=tokenizer)
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

    def _score(self, batch):
        seg_mask = None
        if self.granularity == 'segment':
            seg_mask = self.segmenter.compute(batch=batch)
            for i, sample in enumerate(batch):
                sample.update(seg_mask[i])
        samples = self.collate_fn(samples)
        log_lklh_positive = self.compute_positive(batch)
        log_lklh_negative = self.compute_negative(batch)

        c_mask = batch['completion_mask']
        if self.granularity == 'segment':
            seg_mask = batch['segment_mask']

        rewards = self.compute_rewards(log_lklh_positive, log_lklh_negative, c_mask, seg_mask)
        return rewards
    
    def sigmoid(self, score):
        return 1 / (1 + np.exp(-score))
    
    def rerank(self, srcs: List[str], mts: List[List[str]], tgt_lang: str, return_score: bool = False, normalise_scores: bool = False):
        batches = self.prepare_data(srcs, mts, code2name[tgt_lang])
        results = []
        for i, batch in enumerate(tqdm(batches, total=len(batches), desc="Reranking...")):
            rewards = self._score(batch)
            best = rewards.argmin().item()
            res = {
                "src": srcs[i],
                "mt": mts[i][best],
            }

            if normalise_scores:
                rewards = 1 - rewards.sigmoid()

            if return_score:
                score = rewards[best].item()
                res.update({"score": score})
            results.append(res)
        return results
    
class LikelihoodReranker:
    def __init__(self, model: str | PreTrainedModel, tokenizer: PreTrainedTokenizerBase | None = None, **model_kwargs):
        if isinstance(model, str):
            self.model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model)
        else:
            self.model = model

        self.prepare_data = partial(prepare_data, tokenizer=tokenizer)

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
    
    def compute_logps(self, logits: torch.Tensor, labels: torch.Tensor, completion_mask: torch.Tensor):
        return logits.log_softmax(dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1).masked_fill(~completion_mask, 0)
    
    def I(self, logits: torch.Tensor, labels: torch.Tensor, completion_mask: torch.Tensor):
        logps = self.compute_logps(logits, labels, completion_mask)
        return - logps
    
    def H(self, logits: torch.Tensor, labels: torch.Tensor, completion_mask: torch.Tensor):
        probs = logits.softmax(dim=-1)
        logps = logits.log_softmax(dim=-1)
        return - (probs * logps).sum(dim=-1).masked_fill(~completion_mask, 0)
    
    def perplexity(self, logits: torch.Tensor, labels: torch.Tensor, completion_mask: torch.Tensor):
        entropy = self.H(logits, labels, completion_mask)
        return torch.exp(entropy)
    
    def compute_fn(self, metric: Literal["entropy", "surprisal", "logprobs", "perplexity"] = "entropy"):
        if metric == "entropy":
            return self.H
        if metric == "surprisal":
            return self.I
        if metric == "logprobs":
            return self.compute_logps
        if metric == "perplexity":
            return self.perplexity
    
    def rerank(self, srcs: List[str], mts: List[List[str]], tgt_lang: str, metric: Literal["entropy", "surprisal", "logprobs", "perplexity"] = "entropy", return_score: bool = False, normalise_scores: bool = False):
        batches = self.prepare_data(srcs, mts, tgt_lang)
        compute = self.compute_fn(metric)
        preds = []
        for i, batch in enumerate(tqdm(batches, desc="Reranking...")):
            logits, labels, completion_mask = self.model_forward(batch)
            scores = compute(logits, labels, completion_mask)
            scores = scores.sum(dim=1) / completion_mask.sum(dim=1)
            best_idx = scores.argmax().item()
            if metric == "logprobs":
                if normalise_scores:
                    scores = scores.sigmoid()
            else:
                if normalise_scores:
                    scores = 1 - scores.sigmoid()

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
    
class CometReranker:
    def __init__(
        self,
        model_path: str = "Unbabel/wmt22-cometkiwi-da",
    ):
        model_path = download_model(model_path)
        torch.set_float32_matmul_precision("medium")
        self.model = load_from_checkpoint(model_path)
        self.model = torch.compile(self.model, mode="max-autotune")

    def _convert_sample(self, prompt: str, completions: List[str]):
        sample = []
        for completion in completions:
            sample.append({
                "src": prompt,
                "mt": completion
            })
        return sample
    
    def compute(self, prompt: str, completions: List[str]) -> List[float]:
        batch = self._convert_sample(prompt, completions)
        preds = self.model.predict(batch, progress_bar=False, num_workers=4)
        if hasattr(preds, "metadata"):
            scores = preds.metadata.mqm_scores
        else:
            scores = preds.scores
        return scores
    
    async def _rerank(self, prompt: str, completions: List[str], return_score: bool = False):
        scores = self.compute(prompt, completions)
        best = np.argmax(scores)
        res = {
            "src": prompt,
            "mt": completions[best]
        }

        if return_score:
            score = scores[best]
            res.update({"score": score})
        return res
    
    async def _gather(self, prompts, completions, return_score):
        results = [self._rerank(prompt, completions[i], return_score) for i, prompt in enumerate(prompts)]
        results = await tqdm_asyncio.gather(results, desc="Scoring...")
        return results

    def rerank(self, prompts: List[str], completions: List[List[str]], return_score: bool = False):
        results = asyncio.run(self._gather(prompts, completions, return_score))
        return results
    
class Reranker:
    def __init__(self, model_dir: str, hf_kwargs: dict | None, comet_model: str = "Unbabel/wmt23-cometkiwi-da-xl", comet_kwargs: dict | None = None, segmenter: Segmenter | None = None, granularity: Literal['token', 'sequence', 'segment'] = "token"):
        self.nat_reranker = RatioReranker(model_dir, granularity=granularity, **hf_kwargs)
        self.comet_reranker = CometReranker(comet_model)
    
    def get_best(self, prompt: str, completions: List[str], scores: np.ndarray, return_score: bool = False):
        best = scores.argmax().item()
        res = {
            "src": prompt,
            "mt": completions[best]
        }

        if return_score:
            res.update({
                "score": scores[best].item()
            })
        
        return res

    def rerank(self, prompts: List[str], completions: List[List[str]], tgt_lang: str, return_score: bool = False, w_nat: float = 1., w_comet: float = 1., return_nat: bool = False, return_comet: bool = False):
        results = []
        nat_results = []
        comet_results = []
        for i, prompt in enumerate(tqdm(prompts, desc="Reranking...")):
            nat_scores = self.nat_reranker._score(prompt, completions[i], tgt_lang).sigmoid().float().numpy()
            comet_scores = np.array(self.comet_reranker.compute(prompt, completions[i]))

            scores = ((w_nat * nat_scores) + (w_comet * comet_scores)) / 2
            res = self.get_best(prompt, completions[i], scores, return_score=return_score)

            if return_nat:
                nat_res = self.get_best(prompt, completions[i], nat_scores, return_score=return_score)
                nat_results.append(nat_res)
            if return_comet:
                comet_res = self.get_best(prompt, completions[i], comet_scores, return_score=return_score)
                comet_results.append(comet_res)

            results.append(res)

        if not return_comet and not return_nat:
            return results
        elif return_comet and not return_nat:
            return results, comet_results
        elif return_nat and not return_comet:
            return results, nat_results
        else:
            return results, nat_results, comet_results
        
    def tune(self, prompts: List[str], completions: List[List[str]], tgt_lang: str, init_weights: list[float] | None = None, num_epochs: int = 1, learning_rate: float = 1e-4):   
        def score(
            w1, w2,
            prompt, completions, tgt_lang,
        ):
            nat_scores = self.nat_reranker._score(prompt, completions[i], tgt_lang).sigmoid().to(dtype=torch.float32).numpy()
            comet_scores = np.array(self.comet_reranker.compute(prompt, completions[i]))

            scores = ((nat_scores * w1) + (comet_scores * w2)) / 2
            return scores
        
        def calc_gradient(w1, w2, score_fn, eps=1e-5):
            grad_w1: np.ndarray = (score_fn(w1 + eps, w2) - score_fn(w1, w2)) / eps
            grad_w2: np.ndarray = (score_fn(w1, w2 + eps) - score_fn(w1, w2)) / eps
            return grad_w1.mean(), grad_w2.mean()

        if init_weights is None:
            w1, w2 = 0., 0.
        else:
            w1, w2 = init_weights

        w1_new, w2_new = 0., 0.
        for _ in range(num_epochs):
            for i, prompt in enumerate(tqdm(prompts, desc="Tuning...")):
                score_fn = partial(score, prompt=prompt, completions=completions[i], tgt_lang=tgt_lang)
                if i == 0:
                    _score = score_fn(w1, w2)
                g1, g2 = calc_gradient(w1, w2, score_fn)
                w1 += learning_rate * g1
                w2 += learning_rate * g2
                _new = score_fn(w1, w2)
                if _new.mean() > _score.mean():
                    w1_new = w1
                    w2_new = w2
                    _score = _new
                else:
                    w1 = w1_new
                    w2 = w2_new
                
        
        print("w_nat:", w1)
        print("w_comet:", w2)

        return w1_new, w2_new