from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from functools import partial
import torch

from dataclasses import dataclass
from tqdm import tqdm
from typing_extensions import List, Literal, Optional
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
class RewardOutputs:
    mean: Optional[list] = None
    z_score: Optional[list] = None


@dataclass
class TranslationeseResults:
    token_level: Optional[RewardOutputs]
    seq_level: Optional[RewardOutputs]


class TranslationeseIndex:
    def __init__(
        self,
        model_dir: str | os.PathLike,
        tokenizer: str | os.PathLike | None = None,
        metric_type: Literal["logps", "entropy"] = "logps",
        **model_kwargs,
    ):
        positive_model = AutoModelForCausalLM.from_pretrained(
            f"{model_dir}/positive", **model_kwargs
        )
        negative_model = AutoModelForCausalLM.from_pretrained(
            f"{model_dir}/negative", **model_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer if tokenizer is not None else f"{model_dir}/positive"
        )
        self.model_kwargs = model_kwargs
        if metric_type == "logps":
            self.compute_positive = partial(self.compute_logps, model=positive_model)
            self.compute_negative = partial(self.compute_logps, model=negative_model)
        elif metric_type == "entropy":
            self.compute_positive = partial(self.compute_entropy, model=positive_model)
            self.compute_negative = partial(self.compute_entropy, model=negative_model)

    @torch.no_grad()
    def compute_logits(self, inputs, model: PreTrainedModel):
        outputs = model(**inputs)
        logits: torch.Tensor = outputs.logits[:, :-1].cpu()
        return logits

    def compute_logps(self, example, model: PreTrainedModel):
        _exclude_keys = {"completion_mask", "segment_mask"}
        inputs = {
            k: v.to(model.device) for k, v in example.items() if k not in _exclude_keys
        }

        logits = self.compute_logits(inputs=inputs, model=model)
        completion_mask: torch.Tensor = example["completion_mask"]
        labels: torch.Tensor = inputs["input_ids"][:, 1:].cpu()
        completion_mask = completion_mask[:, 1:].bool()

        logps = logits.log_softmax(dim=-1)
        logps = logps.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        logps = logps.masked_fill(~completion_mask, 0)

        if self.granularity == "sequence":
            logps = logps.sum(dim=1) / completion_mask.sum(dim=1)

        return logps

    def per_segment_logps(
        self,
        log_lklh_positive: torch.Tensor,
        log_lklh_negative: torch.Tensor,
        mask: torch.Tensor,
    ):
        rewards = []
        logps_pos_segments = []
        logps_neg_segments = []
        for i, m in enumerate(mask):
            if m < 0:
                continue
            if i == len(mask) - 1:
                logps_pos_segments.append(log_lklh_positive[i])
                logps_neg_segments.append(log_lklh_negative[i])
                logps_pos_segments = torch.stack(logps_pos_segments)
                logps_neg_segments = torch.stack(logps_neg_segments)
                segment_rewards = logps_pos_segments.mean() - logps_neg_segments.mean()
                rewards.append(segment_rewards)
                continue
            is_end = (m == 1) or (mask[i + 1] < 0)
            logps_pos_segments.append(log_lklh_positive[i])
            logps_neg_segments.append(log_lklh_negative[i])
            if is_end:
                logps_pos_segments = torch.stack(logps_pos_segments)
                logps_neg_segments = torch.stack(logps_neg_segments)
                segment_rewards = logps_pos_segments.mean() - logps_neg_segments.mean()
                rewards.append(segment_rewards)
                logps_pos_segments = []
                logps_neg_segments = []
                continue
        rewards = torch.stack(rewards)
        return rewards.mean()

    def compute_entropy(self, example, model: PreTrainedModel):
        _exclude_keys = {"completion_mask", "segment_mask"}
        inputs = {
            k: v.to(model.device) for k, v in example.items() if k not in _exclude_keys
        }

        logits = self.compute_logits(inputs=inputs, model=model)
        completion_mask: torch.Tensor = example["completion_mask"].clone()
        completion_mask = completion_mask[:, 1:].bool()

        entropy = -(logits.softmax(dim=-1) * logits.log_softmax(dim=-1)).sum(dim=-1)
        entropy = entropy.masked_fill(~completion_mask, 0)
        return entropy

    def tokenize_fn(self, text, tokenizer: PreTrainedTokenizerBase, **kwargs):
        return tokenizer.apply_chat_template(
            text, tokenize=True, return_dict=True, **kwargs
        )

    def format_messages(
        self,
        src: str,
        mt: str,
        lang: str,
    ):
        return {
            "src": [
                {
                    "role": "user",
                    "content": f"Translate the following text into {lang}.\n\n{src}",
                }
            ],
            "mt": [{"role": "assistant", "content": mt}],
        }

    def _prepare_sample(self, example):
        output = {}
        prompt_ids = self.tokenize_fn(
            example["src"], self.tokenizer, add_generation_prompt=True
        )["input_ids"]
        prompt_completion_processed = self.tokenize_fn(
            example["src"] + example["mt"], self.tokenizer
        )

        prompt_completion_ids = prompt_completion_processed["input_ids"]
        completion_mask = [0] * len(prompt_ids) + [1] * (
            len(prompt_completion_ids) - len(prompt_ids)
        )
        attention_mask = prompt_completion_processed["attention_mask"]

        output["input_ids"] = torch.tensor(prompt_completion_ids)
        output["attention_mask"] = torch.tensor(attention_mask)
        output["completion_mask"] = torch.tensor(completion_mask)
        return output

    def prepare_batch(self, examples):
        outputs = [self._prepare_sample(example) for example in examples]
        return outputs

    def collate_fn(self, examples):
        input_ids = [example["input_ids"] for example in examples]
        attention_mask = [example["attention_mask"] for example in examples]
        completion_mask = [example["completion_mask"] for example in examples]

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
            segment_mask = torch.stack(
                [example["segment_mask"] for example in examples]
            )
            outputs.update({"segment_mask": segment_mask})

        return outputs

    def compute_rewards(
        self,
        log_lklh_positive: torch.Tensor,
        log_lklh_negative: torch.Tensor,
        c_masks: torch.Tensor | None = None,
        seg_masks: torch.Tensor | None = None,
    ):
        if self.granularity == "segment":
            per_segment_rewards = [
                self.per_segment_logps(logps_pos, logps_neg, seg_mask)
                for logps_pos, logps_neg, seg_mask in zip(
                    log_lklh_positive, log_lklh_negative, seg_masks
                )
            ]
            rewards = torch.stack(per_segment_rewards)
            return rewards
        if self.use_perplexity and not self.granularity == "token":
            ppl_positive = torch.exp(
                -(log_lklh_positive.sum(dim=1) / c_masks.sum(dim=1))
            )
            ppl_negative = torch.exp(
                -(log_lklh_negative.sum(dim=1) / c_masks.sum(dim=1))
            )

            rewards = ppl_positive - ppl_negative
            return rewards
        rewards = log_lklh_positive - log_lklh_negative
        if self.granularity == "token":
            rewards = rewards.sum(dim=-1) / c_masks.sum(dim=-11)
        return rewards

    def compute_rewards(
        self,
        log_lklh_positive: torch.Tensor,
        log_lklh_negative: torch.Tensor,
        c_masks: torch.Tensor | None = None,
        seg_masks: torch.Tensor | None = None,
    ):
        if self.granularity == "segment":
            seg_masks = seg_masks[:, 1:]
            if not c_masks.shape == log_lklh_positive.shape:
                c_masks = c_masks[:, 1:]
            per_segment_rewards = [
                self.per_segment_logps(logps_pos, logps_neg, seg_mask)
                for logps_pos, logps_neg, seg_mask in zip(
                    log_lklh_positive, log_lklh_negative, seg_masks
                )
            ]
            rewards = torch.stack(per_segment_rewards)
            return rewards
        rewards = log_lklh_positive - log_lklh_negative
        if self.granularity == "token":
            rewards = rewards.sum(dim=1) / c_masks.sum(dim=1)
        return rewards

    def score(
        self,
        srcs: List[str],
        mts: List[str],
        lang: str,
        batch_size: int = 4,
        reward_normalise_type: Literal["mean", "z_score", "all"] = "mean",
        segmenter: Segmenter | None = None,
        granularity: Literal["token", "sequence", "segment"] = "token",
    ):
        self.granularity = granularity
        if self.granularity == "segment" and segmenter is None:
            self.segmenter = Segmenter("Qwen/Qwen2.5-0.5B", **self.model_kwargs)
        else:
            self.segmenter = segmenter
        self.reward_normalise_type = reward_normalise_type
        rewards = []
        hyps = [self.format_messages(src, mt, lang) for src, mt in zip(srcs, mts)]
        for i in tqdm(
            range(0, len(hyps), batch_size),
            desc=f"Evaluating {self.granularity}-level...",
        ):
            seg_mask = None
            batch = self.prepare_batch(hyps[i : i + batch_size])
            if self.granularity == "segment":
                seg_mask = self.segmenter.compute(batch=batch)
                for i, sample in enumerate(batch):
                    sample.update(seg_mask[i])
            batch = self.collate_fn(batch)

            log_lklh_positive = self.compute_positive(batch)
            log_lklh_negative = self.compute_negative(batch)

            c_mask = batch["completion_mask"]
            if self.granularity == "segment":
                seg_mask = batch["segment_mask"]

            per_batch_rewards = self.compute_rewards(
                log_lklh_positive, log_lklh_negative, c_mask, seg_mask
            )
            per_batch_rewards = 1 - per_batch_rewards.sigmoid()
            rewards += per_batch_rewards.tolist()
        return {"score": rewards, "mean_score": np.mean(rewards).item()}
