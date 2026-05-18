from transformers import PreTrainedModel, PreTrainedTokenizerBase, AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Literal
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial

import torch
import numpy as np
import os

from segmenter import Segmenter

code2name = {
    "zho": "Chinese",
    "fra": "French",
    "deu": "German",
    "msa": "Malay",
    "kor": "Korean",
}

@dataclass
class RerankerConfig:
    lang: str
    granularity: str = "token"
    return_score: bool = False
    normalise_scores: bool = False

class TranslationeseReranker:
    def __init__(self, model_dir: str | os.PathLike, tokenizer: str | os.PathLike | None = None, **model_kwargs):
        positive_model = AutoModelForCausalLM.from_pretrained(f"{model_dir}/negative", **model_kwargs)
        negative_model = AutoModelForCausalLM.from_pretrained(f"{model_dir}/positive", **model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer if tokenizer is not None else f"{model_dir}/negative")
        self.compute_positive = partial(self.compute_logps, model=positive_model)
        self.compute_negative = partial(self.compute_logps, model=negative_model)

    def compute_logps(self, inputs, model: PreTrainedModel):
        inputs = {
            k: v.to(model.device)
            for k, v in inputs.items()
        }
        completion_mask: torch.Tensor = inputs.pop("completion_mask")
        with torch.no_grad():
            outputs = model(**inputs)
        logits: torch.Tensor = outputs.logits[:, :-1]
        labels: torch.Tensor = inputs['input_ids'][:, 1:]
        completion_mask = completion_mask[:, 1:].bool()
        
        logps = logits.log_softmax(dim=-1)
        logps = logps.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        logps = logps.masked_fill(~completion_mask, 0)

        if self.granularity == "sequence":
            logps = logps.sum(dim=1) / completion_mask.sum(dim=1)

        return logps.cpu()
    
    def tokenize_fn(self, text, tokenizer: PreTrainedTokenizerBase, **kwargs):
        return tokenizer.apply_chat_template(
            text,
            tokenize=True,
            return_dict=True,
            **kwargs
        )
    
    def format_messages(self, src: str, mt: str, lang: str,):
        return {
            "src": [
                {"role": "user", "content": f"Translate the following text into {lang}.\n\n{src}"}
            ],
            "mt": [
                {"role": "assistant", "content": mt}
            ]
        }
    
    def preprocess_batch(self, src: str, mts: List[str], lang: str):
        data = []
        for mt in mts:
            sample = self.format_messages(src, mt, lang)
            data.append(sample)
        return data
    
    def prepare_data(self, example, tokenizer: PreTrainedTokenizerBase):
        output = {}
        prompt_ids = self.tokenize_fn(example['src'], tokenizer, add_generation_prompt=True)['input_ids']
        prompt_completion_processed = self.tokenize_fn(example['src'] + example['mt'], tokenizer)

        prompt_completion_ids = prompt_completion_processed['input_ids']
        completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))

        output['input_ids'] = torch.tensor([prompt_completion_ids])
        output['attention_mask'] = torch.tensor([prompt_completion_processed['attention_mask']])
        output['completion_mask'] = torch.tensor([completion_mask])
        return output
    
    def compute_per_segment_rewards(self, tensor: torch.Tensor, mask: torch.Tensor):
        rewards = []
        segment = []
        for i, t in enumerate(mask):
            if t < 0:
                continue
            if t:
                segment.append(tensor[i])
            if t == 0:
                #segment.append(tensor[i])
                if segment:
                    reward = torch.tensor(segment).mean()
                    rewards.append(reward)
                segment = [tensor[i]]
            if not i == len(mask) - 1:
                if mask[i+1] < 0:
                    reward = torch.tensor(segment).mean()
                    rewards.append(reward)
            else:
                if t > 0:
                    reward = torch.tensor(segment).mean()
                    rewards.append(reward)
        return torch.tensor(rewards).mean()

    def compute_rewards(self, log_lklh_positive: torch.Tensor, log_lklh_negative: torch.Tensor, c_mask: torch.Tensor | None = None, seg_mask: torch.Tensor | None = None):
        logps_ratios = log_lklh_positive - log_lklh_negative
        if self.granularity == 'token':
            logps_ratios = logps_ratios.sum(dim=1) / c_mask.sum(dim=1)
        if self.granularity == "segment":
            per_segment_logps_ratios = [self.compute_per_segment_rewards(ratio, mask) for ratio, mask in zip(logps_ratios, seg_mask)]
            logps_ratios = torch.stack(per_segment_logps_ratios)
        return logps_ratios
    
    def _score_loop_segment(self, samples, seg_mask: torch.Tensor):
        rewards = []
        for sample, mask in zip(samples, seg_mask):
            log_lklh_positive = self.compute_positive(sample)
            log_lklh_negative = self.compute_negative(sample)

            if self.granularity == 'token':
                c_mask = sample['completion_mask']
            else:
                c_mask = None

            reward = self.compute_rewards(log_lklh_positive, log_lklh_negative, c_mask, mask)
            rewards.append(reward)
        return torch.stack(rewards)

    def _score_loop(self, samples):
        rewards = []
        for sample in samples:
            log_lklh_positive = self.compute_positive(sample)
            log_lklh_negative = self.compute_negative(sample)

            if self.granularity == 'token':
                c_mask = sample['completion_mask']
            else:
                c_mask = None

            reward = self.compute_rewards(log_lklh_positive, log_lklh_negative, c_mask)
            rewards.append(reward)
        return torch.stack(rewards)

    def _score(self, src: str, mts: List[str], lang: str, seg_mask: torch.Tensor | None = None):
        samples = [self.prepare_data(sample, self.tokenizer) for sample in self.preprocess_batch(src, mts, lang)]
        if self.granularity == 'segment':
            rewards = self._score_loop_segment(samples, seg_mask)
        else:
            rewards = self._score_loop(samples)
        return rewards
    
    def sigmoid(self, score):
        return 1 / (1 + np.exp(-score))
    
    def _rerank_loop(self, srcs: List[str], mts: List[List[str]], lang: str, return_score: bool = False, normalise_scores: bool = False):
        results = []
        for src, mt in tqdm(zip(srcs, mts), total=len(srcs), desc="Reranking..."):
            rewards = self._score(src, mt, code2name[lang])
            best = rewards.argmax().item()
            res = {
                "src": src,
                "mt": mt[best],
            }

            if normalise_scores:
                rewards = rewards.sigmoid(-1).item()

            if return_score:
                score = rewards[best].item()
                res.update({"score": score})
            results.append(res)
        return results

    def _rerank_loop_segment(self, srcs: List[str], mts: List[List[str]], lang: str, return_score: bool = False, normalise_scores: bool = False, seg_masks: list[torch.Tensor] | None = None):
        results = []
        for src, mt, seg_mask in tqdm(zip(srcs, mts, seg_masks), total=len(srcs), desc="Reranking..."):
            rewards = self._score(src, mt, code2name[lang], seg_mask)
            best = rewards.argmax().item()
            res = {
                "src": src,
                "mt": mt[best],
            }

            if normalise_scores:
                rewards = rewards.softmax(-1).item()

            if return_score:
                score = rewards[best].item()
                res.update({"score": score})
            results.append(res)
        return torch.stack(results)

    def rerank(self, srcs: List[str], mts: List[List[str]], lang: str, granularity: Literal['token', 'sequence', 'segment'] = 'token', return_score: bool = False, normalise_scores: bool = False, seg_masks: list[torch.Tensor] | None = None):
        if granularity == "segment":
            assert seg_masks is not None, "Segment-level reranking requires `seg_masks`"
        self.granularity = granularity
        if self.granularity == 'segment':
            results = self._rerank_loop_segment(
                srcs,
                mts,
                lang=lang,
                return_score=return_score,
                normalise_scores=normalise_scores,
                seg_masks=seg_masks
            )
        else:
            results = self._rerank_loop(
                srcs,
                mts,
                lang=lang,
                return_score=return_score,
                normalise_scores=normalise_scores
            )
        return results