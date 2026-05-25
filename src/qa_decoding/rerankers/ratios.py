from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from typing import List, Literal
from tqdm import tqdm
from functools import partial

import torch
import numpy as np
import os

from segmenter import Segmenter
from rerankers.base_reranker import BaseReranker
from rerankers.qe_rerank import CometReranker


code2name = {
    "zho": "Chinese",
    "fra": "French",
    "deu": "German",
    "msa": "Malay",
    "kor": "Korean",
}


class QAReranker(BaseReranker):
    def __init__(
        self,
        model_path: str | os.PathLike,
        tokenizer: str | os.PathLike | None = None,
        segmenter: Segmenter | None = None,
        comet_model: str = "Unbabel/wmt22-cometkiwi-da",
        granularity: Literal["token", "sequence", "segment"] = "token",
        metric_type: Literal["logprobs", "entropy", "perplexity"] = "logprobs",
        **model_kwargs,
    ):
        positive_model = AutoModelForCausalLM.from_pretrained(
            f"{model_path}/positive", **model_kwargs
        )
        negative_model = AutoModelForCausalLM.from_pretrained(
            f"{model_path}/negative", **model_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/positive")
        self.comet_reranker = CometReranker(comet_model)
        super().__init__(positive_model, tokenizer)
        self.granularity = granularity
        if self.granularity == "segment" and segmenter is None:
            self.segmenter = Segmenter("Qwen/Qwen3-0.6B", **model_kwargs)
        else:
            self.segmenter = segmenter
        if metric_type == "logprobs":
            self.compute_positive = partial(self.compute_logps, model=positive_model)
            self.compute_negative = partial(self.compute_logps, model=negative_model)
        elif metric_type == "entropy":
            self.compute_positive = partial(self.compute_entropy, model=positive_model)
            self.compute_negative = partial(self.compute_entropy, model=negative_model)

        self.use_perplexity = metric_type == "perplexity"

    @torch.no_grad()
    def compute_logits(self, inputs, model: PreTrainedModel):
        outputs = model(**inputs, use_cache=True)
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

    def compute_entropy(self, example, model: PreTrainedModel):
        _exclude_keys = {"completion_mask", "segment_mask"}
        inputs = {
            k: v.to(model.device) for k, v in example.items() if k not in _exclude_keys
        }

        logits = self.compute_logits(inputs=inputs, model=model)
        labels = example["input_ids"][:, 1:]
        completion_mask: torch.Tensor = example["completion_mask"].clone()
        completion_mask = completion_mask[:, 1:].bool()

        entropy = (
            (logits.softmax(dim=-1) * -logits.log_softmax(dim=-1))
            .gather(-1, labels.unsqueeze(-1))
            .squeeze(-1)
        )
        entropy = entropy.masked_fill(~completion_mask, 0)
        return entropy

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

    def _score(self, batch: dict):
        log_lklh_positive = self.compute_positive(batch)
        log_lklh_negative = self.compute_negative(batch)

        c_mask = batch["completion_mask"]
        seg_mask = batch.get("segment_mask")

        rewards = self.compute_rewards(
            log_lklh_positive, log_lklh_negative, c_mask, seg_mask
        )
        return rewards

    def sigmoid(self, score):
        return 1 / (1 + np.exp(-score))

    def get_unique_seqs_idxs(self, mts: List[List[str]]):
        unique = set()
        unique_sequences = []
        idxs = []
        for i, mt in enumerate(mts):
            if mt not in unique:
                idxs.append(i)
                unique_sequences.append(mt)
                unique.add(mt)
        return torch.tensor(idxs), unique_sequences

    def rerank(
        self,
        srcs: List[str],
        mts: List[List[str]],
        refs: List[str],
        tgt_lang: str,
        return_score: bool = False,
        normalise_scores: bool = False,
    ):
        comet_batches = []
        for i, src in enumerate(srcs):
            comet_batches += self.comet_reranker._convert_sample(src, mts[i])
        comet_scores = self.comet_reranker.compute(comet_batches)
        comet_scores = [
            comet_scores[i * len(mts[i]) : (i + 1) * len(mts[i])]
            for i in range(len(srcs))
        ]

        batches = self.prepare_data(srcs, mts, code2name[tgt_lang])
        if self.granularity == "segment":
            seg_masks = self.segmenter.compute(srcs, mts, code2name[tgt_lang])
            for i, batch in enumerate(batches):
                batch.update({"segment_mask": seg_masks[i]})
        results = []
        for i, batch in enumerate(
            tqdm(batches, total=len(batches), desc="Reranking...")
        ):
            unique_seqs, per_batch_mts = self.get_unique_seqs_idxs(mts[i])
            per_batch = {
                k: torch.stack([v[idx] for idx in unique_seqs])
                for k, v in batch.items()
            }

            per_batch_comet_scores = torch.tensor(
                [comet_scores[i][idx] for idx in unique_seqs]
            )
            batch_len = per_batch_comet_scores.shape[0]
            if batch_len >= 6:
                k = batch_len // 2
                comet_rewards, candidate_idxs = torch.topk(per_batch_comet_scores, k=k)
                per_batch = {
                    k: torch.stack([v[idx.item()] for idx in candidate_idxs])
                    for k, v in per_batch.items()
                }
            else:
                comet_rewards = per_batch_comet_scores

            rewards = self._score(per_batch)
            ratios_rewards = (rewards - rewards.mean()) / rewards.std()

            all_rewards = comet_rewards - ratios_rewards
            scaled_rewards = (all_rewards - all_rewards.mean()) / all_rewards.std()

            scaled_best = scaled_rewards.argmax(-1).item()
            ratio_best = ratios_rewards.argmin(-1).item()
            comet_best = comet_rewards.argmax(-1).item()

            res = {
                "src": srcs[i],
                "ref": refs[i],
                "mts": {
                    "ratio": per_batch_mts[ratio_best],
                    "comet": per_batch_mts[comet_best],
                    "scaled": per_batch_mts[scaled_best],
                },
                "scores": {
                    "ratio": 1 - (ratios_rewards[ratio_best].sigmoid().item()),
                    "comet": comet_rewards[comet_best].item(),
                    "scaled": scaled_rewards[scaled_best].item(),
                },
            }

            results.append(res)
        return results
