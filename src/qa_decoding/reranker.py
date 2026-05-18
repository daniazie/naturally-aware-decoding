from transformers import PreTrainedModel, PreTrainedTokenizerBase, AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Literal
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from comet import load_from_checkpoint, download_model

import torch
import numpy as np
import os

from segmenter import Segmenter
from utils import pad

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
        self.positive_model = AutoModelForCausalLM.from_pretrained(f"{model_dir}/negative", **model_kwargs)
        negative_model = AutoModelForCausalLM.from_pretrained(f"{model_dir}/positive", **model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer if tokenizer is not None else f"{model_dir}/negative")
        self.compute_positive = partial(self.compute_logps, model=self.positive_model)
        self.compute_negative = partial(self.compute_logps, model=negative_model)

    def compute_logps(self, example, model: PreTrainedModel):
        _exclude_keys = {"completion_mask", "segment_mask"}
        inputs = {
            k: v.to(model.device)
            for k, v in example.items()
            if not k in _exclude_keys
        }
        completion_mask: torch.Tensor = example["completion_mask"].clone()
        with torch.no_grad():
            outputs = model(**inputs)
        logits: torch.Tensor = outputs.logits[:, :-1].cpu()
        labels: torch.Tensor = inputs['input_ids'][:, 1:].cpu()
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
    
    def format_messages(self, src: str, mt: str, lang: str, seg_mask: torch.Tensor | None = None):
        message = {
            "src": [
                {"role": "user", "content": f"Translate the following text into {lang}.\n\n{src}"}
            ],
            "mt": [
                {"role": "assistant", "content": mt}
            ]
        }

        if self.granularity == "segment":
            message.update({
                "segment_mask": seg_mask
            })
        return message
    
    def preprocess_batch(self, src: str, mts: List[str], lang: str, seg_masks: torch.Tensor | None = None):
        data = []
        if self.granularity == 'segment':
            for mt, seg_mask in zip(mts, seg_masks):
                sample = self.format_messages(src, mt, lang, seg_mask)
                data.append(sample)
        else:
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

        output['input_ids'] = torch.tensor(prompt_completion_ids)
        output['attention_mask'] = torch.tensor(prompt_completion_processed['attention_mask'])
        output['completion_mask'] = torch.tensor(completion_mask)
        if self.granularity == 'segment':
            output['segment_mask'] = example['segment_mask']
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

        return torch.stack(rewards).mean()

    def compute_rewards(self, log_lklh_positive: torch.Tensor, log_lklh_negative: torch.Tensor, c_mask: torch.Tensor | None = None, seg_masks: torch.Tensor | None = None):
        logps_ratios = log_lklh_positive - log_lklh_negative
        if self.granularity == 'token':
            logps_ratios = logps_ratios.sum(dim=1) / c_mask.sum(dim=1)
        if self.granularity == "segment":
            seg_masks = seg_masks[:, 1:]
            per_segment_logps_ratios = [self.compute_per_segment_rewards(ratio, seg_mask) for ratio, seg_mask in zip(logps_ratios, seg_masks)]
            logps_ratios = torch.stack(per_segment_logps_ratios)
        return logps_ratios
    

    def _score(self, src: str, mts: List[str], lang: str, seg_masks: torch.Tensor | None = None):
        samples = [self.prepare_data(sample, self.tokenizer) for sample in self.preprocess_batch(src, mts, lang, seg_masks)]
        samples = self.collate_fn(samples)
        log_lklh_positive = self.compute_positive(samples)
        log_lklh_negative = self.compute_negative(samples)

        if self.granularity == 'token':
            c_mask = samples['completion_mask']
            seg_mask = None
        elif self.granularity == 'segment':
            seg_mask = samples['segment_mask']
            c_mask = None
        else:
            c_mask = None
            seg_mask = None

        rewards = self.compute_rewards(log_lklh_positive, log_lklh_negative, c_mask, seg_mask)
        return rewards
    
    def collate_fn(self, examples):
        input_ids = [example['input_ids'] for example in examples]
        attention_mask = [example['attention_mask'] for example in examples]
        completion_mask = [example['completion_mask'] for example in examples]

        input_ids = pad(
            input_ids,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = pad(
            attention_mask,
            padding_value=0,
        )
        completion_mask = pad(
            completion_mask,
            padding_value=0,
        )

        outputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
        }

        if self.granularity == "segment":
            segment_mask = torch.stack([example['segment_mask'] for example in examples])
            
            assert input_ids.shape == segment_mask.shape, f"Mask size `{segment_mask.shape}` does not match inputs size `{input_ids.shape}`."
            outputs.update({"segment_mask": segment_mask})
        
        return outputs
    
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
                rewards = rewards.sigmoid()

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
                rewards = rewards.sigmoid()

            if return_score:
                score = rewards[best].item()
                res.update({"score": score})
            results.append(res)
        return results

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
    

