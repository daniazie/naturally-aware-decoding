from transformers import PreTrainedModel, PreTrainedTokenizerBase, AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Literal
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from comet import load_from_checkpoint, download_model
from warnings import WarningMessage

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
class NatArgs:
    lang: str
    return_score: bool = False
    normalise_scores: bool = False

@dataclass
class CometConfig:
    return_score: bool = False

@dataclass
class RerankerConfig:
    w_nat: float = 1.0
    w_comet: float = 1.0
    return_score: bool = False
    return_nat: bool = False
    return_comet: bool = False

class TranslationeseReranker:
    def __init__(self, model_dir: str | os.PathLike, tokenizer: str | os.PathLike | None = None, segmenter: Segmenter | None = None, granularity: Literal['token', 'sequence', 'segment'] = 'token', metric_type: Literal['logps', 'entropy'] = 'logps', **model_kwargs):
        positive_model = AutoModelForCausalLM.from_pretrained(f"{model_dir}/negative", **model_kwargs)
        negative_model = AutoModelForCausalLM.from_pretrained(f"{model_dir}/positive", **model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer if tokenizer is not None else f"{model_dir}/negative")
        self.granularity = granularity
        if self.granularity == 'segment' and segmenter is None:
            self.segmenter = Segmenter(positive_model, self.tokenizer)
        else:
            self.segmenter = segmenter
        if metric_type == 'logps':
            self.compute_positive = partial(self.compute_logps, model=positive_model)
            self.compute_negative = partial(self.compute_logps, model=negative_model)
        elif metric_type == 'entropy':
            self.compute_positive = partial(self.compute_entropy, model=positive_model)
            self.compute_negative = partial(self.compute_entropy, model=negative_model)

    @torch.no_grad()
    def compute_logits(self, inputs, model: PreTrainedModel):
        outputs = model(**inputs)
        logits: torch.Tensor = outputs.logits[:, :-1].cpu()
        inputs = {
            k: v.cpu()
            for k, v in inputs.items()
        }
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
    
    def tokenize_fn(self, text, tokenizer: PreTrainedTokenizerBase, **kwargs):
        return tokenizer.apply_chat_template(
            text,
            tokenize=True,
            return_dict=True,
            **kwargs
        )
    
    def format_messages(self, src: str, mt: str, lang: str):
        message = {
            "src": [
                {"role": "user", "content": f"Translate the following text into {lang}.\n\n{src}"}
            ],
            "mt": [
                {"role": "assistant", "content": mt}
            ]
        }
        return message
    
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
        attention_mask = prompt_completion_processed['attention_mask']
        completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))

        if self.granularity == 'segment':
            forward_input, backward_input = {}, {}
            forward_input['input_ids'] = torch.tensor(prompt_completion_ids)
            forward_input['attention_mask'] = torch.tensor(attention_mask)
            forward_input['completion_mask'] = torch.tensor(completion_mask)

            backward_input['input_ids'] = torch.tensor(prompt_completion_ids[::-1])
            backward_input['attention_mask'] = torch.tensor(attention_mask[::-1])
            backward_input['completion_mask'] = torch.tensor(completion_mask[::-1])
            return forward_input, backward_input

        output['input_ids'] = torch.tensor(prompt_completion_ids)
        output['attention_mask'] = torch.tensor(attention_mask)
        output['completion_mask'] = torch.tensor(completion_mask)
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
    
    def per_segment_logps(self, pos_tensor: torch.Tensor, neg_tensor: torch.Tensor, seg_mask: torch.Tensor):
        rewards = []
        segment_pos = []
        segment_neg = []
        for i, t in enumerate(seg_mask):
            if t < 0:
                continue
            if t:
                segment_pos.append(pos_tensor[i])
                segment_neg.append(neg_tensor[i])
            if t == 0:
                if segment_pos and segment_neg:
                    logps_pos = torch.tensor(segment_pos).sum(dim=-1)
                    logps_neg = torch.tensor(segment_neg).sum(dim=-1)
                    logps_ratios = logps_pos - logps_neg
                    rewards.append(logps_ratios)
                    segment_pos = [pos_tensor[i]]
                    segment_neg = [neg_tensor[i]]
            if not i == len(seg_mask) - 1:
                if seg_mask[i+1] < 0:
                    logps_pos = torch.tensor(segment_pos).sum(dim=-1)
                    logps_neg = torch.tensor(segment_neg).sum(dim=-1)
                    logps_ratios = logps_pos - logps_neg
                    rewards.append(logps_ratios)
            else:
                if t > 0:
                    logps_pos = torch.tensor(segment_pos).sum(dim=-1)
                    logps_neg = torch.tensor(segment_neg).sum(dim=-1)
                    logps_ratios = logps_pos - logps_neg
                    rewards.append(logps_ratios)
        return torch.stack(rewards).mean()

    def compute_rewards(self, log_lklh_positive: torch.Tensor, log_lklh_negative: torch.Tensor, c_masks: torch.Tensor | None = None, seg_masks: torch.Tensor | None = None):
        if self.granularity == 'segment':
            seg_masks = seg_masks[:, 1:]
            if not c_masks.shape == log_lklh_positive.shape:
                c_masks = c_masks[:, 1:]
            per_segment_rewards = [self.per_segment_logps(logps_pos, logps_neg, seg_mask) for logps_pos, logps_neg, seg_mask in zip(log_lklh_positive, log_lklh_negative, seg_masks)]
            rewards = torch.stack(per_segment_rewards)
            return rewards
        rewards = log_lklh_positive - log_lklh_negative
        if self.granularity == 'token':
            rewards = rewards.sum(dim=1) / c_masks.sum(dim=1)
        # if self.granularity == "segment":
        #     seg_masks = seg_masks[:, 1:]
        #     per_segment_logps_ratios = [self.compute_per_segment_rewards(ratio, seg_mask) for ratio, seg_mask in zip(logps_ratios, seg_masks)]
        #     logps_ratios = torch.stack(per_segment_logps_ratios)
        return rewards

    def _score(self, src: str, mts: List[str], lang: str):
        seg_mask = None
        if self.granularity == 'segment':
            forward_batch, backward_batch = [], []
            _samples = [self.prepare_data(sample, self.tokenizer) for sample in self.preprocess_batch(src, mts, lang)]
            for sample in _samples:
                forward_batch.append(sample[0])
                backward_batch.append(sample[1])
            seg_mask = self.segmenter.compute(batch=(forward_batch, backward_batch))
            samples = forward_batch
            for sample, mask in zip(samples, seg_mask):
                sample.update(mask)
        else:
            samples = [self.prepare_data(sample, self.tokenizer) for sample in self.preprocess_batch(src, mts, lang)]
        samples = self.collate_fn(samples)
        log_lklh_positive = self.compute_positive(samples)
        log_lklh_negative = self.compute_negative(samples)

        c_mask = samples['completion_mask']
        if self.granularity == 'segment':
            seg_mask = samples['segment_mask']

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


    def rerank(self, srcs: List[str], mts: List[List[str]], lang: str, return_score: bool = False, normalise_scores: bool = False):
        results = self._rerank_loop(
            srcs,
            mts,
            lang=lang,
            return_score=return_score,
            normalise_scores=normalise_scores
        )
        return results
    
class CometReranker:
    def __init__(
        self,
        model_path: str = "Unbabel/wmt23-cometkiwi-da-xl",
    ):
        model_path = download_model(model_path)
        self.model = load_from_checkpoint(model_path)

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
        scores = self.model.predict(batch, progress_bar=False)
        return scores.scores
    
    def rerank(self, prompts: List[str], completions: List[List[str]], return_score: bool = False):
        results = []
        for i, prompt in enumerate(tqdm(prompts, desc='Reranking...')):
            scores = self.compute(prompt, completions[i])
            best = np.argmax(scores)
            res = {
                "src": prompt,
                "mt": completions[i][best]
            }

            if return_score:
                score = scores[best]
                res.update({"score": score})
            results.append(res)
        return results
    
class Reranker:
    def __init__(self, nat_eval_model_dir: str, hf_kwargs: dict | None, comet_model: str = "Unbabel/wmt23-cometkiwi-da-xl", comet_kwargs: dict | None = None, segmenter: Segmenter | None = None, granularity: Literal['token', 'sequence', 'segment'] = "token"):
        self.nat_reranker = TranslationeseReranker(nat_eval_model_dir, granularity=granularity, **hf_kwargs)
        self.comet_reranker = CometReranker(comet_model, **comet_kwargs)
    
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

    def rerank(self, prompts: List[str], completions: List[List[str]], lang: str, return_score: bool = False, w_nat: float = 1., w_comet: float = 1., return_nat: bool = False, return_comet: bool = False):
        results = []
        nat_results = []
        comet_results = []
        for i, prompt in enumerate(tqdm(prompts, desc="Reranking...")):
            nat_scores = self.nat_reranker._score(prompt, completions[i], lang).sigmoid().float().numpy()
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