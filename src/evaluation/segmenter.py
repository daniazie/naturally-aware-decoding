from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
import torch
from tqdm import tqdm

from typing import List

from utils import pad


class Segmenter:
    def __init__(
        self,
        model: str | PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **model_kwargs,
    ):
        if isinstance(model, str):
            self.model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(model)
        else:
            self.model = model
            self.tokenizer = tokenizer

    def H(self, logits: torch.Tensor):
        probs = logits.softmax(-1)
        logps = logits.log_softmax(-1)
        return -(probs * logps).sum(-1)

    def I(self, logits: torch.Tensor, labels: torch.Tensor):
        return -logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    def compute_mask(self, tensors: torch.Tensor, c_masks: torch.Tensor):
        minima_masks = []
        for tensor in tensors:
            minima_mask = []
            for i in range(len(tensor)):
                if i + 1 == len(tensor):
                    minima_mask.append(tensor[i] < tensor[i - 1])
                elif i == 0:
                    minima_mask.append(tensor[i] < tensor[i + 1])
                else:
                    minima_mask.append(
                        tensor[i] < tensor[i - 1] and tensor[i] < tensor[i + 1]
                    )
            minima_masks.append(torch.stack(minima_mask))
        minima_masks = torch.stack(minima_masks)

        t_min_seq = (
            tensors.masked_fill(~minima_masks, 0).masked_fill(~c_masks, 0).sum(dim=-1)
            / c_masks.sum(dim=-1)
        ).unsqueeze(-1)
        t_min_seg = (
            tensors.masked_fill(~minima_masks, 0).masked_fill(~c_masks, 0).sum(dim=-1)
            / minima_masks.int().masked_fill(~c_masks, 0).sum(dim=-1)
        ).unsqueeze(-1)

        t_min = torch.amin(torch.cat((t_min_seq, t_min_seg)), keepdim=True)

        masks = minima_masks & (tensors < t_min)
        return masks

    def segment(self, tensors: torch.Tensor, masks: torch.Tensor):
        all_segments = []
        for tensor, mask in zip(tensors, masks):
            segments = []
            for i, m in enumerate(mask):
                if m < 0:
                    continue
                if i == len(mask) - 1:
                    segment.append(tensor[i])
                    segments.append(segment)
                    continue
                is_end = (m == 1) or (mask[i + 1] < 0)
                segment.append(tensor[i])
                if is_end:
                    segments.append(segment)
                    segment = []
                    continue
            all_segments.append(segments)
        return all_segments

    @torch.no_grad()
    def model_forward(self, batch):
        _exclude_keys = {"completion_mask"}
        forward_inputs = {
            k: v.to(self.model.device)
            for k, v in batch.items()
            if k not in _exclude_keys
        }

        logits: torch.Tensor = self.model(**forward_inputs).logits[:, :-1].cpu()
        labels: torch.Tensor = forward_inputs["input_ids"][:, 1:].cpu()
        completion_mask: torch.Tensor = batch["completion_mask"][:, 1:].bool().cpu()

        forward_inputs = {
            k: v.cpu() for k, v in batch.items() if k not in _exclude_keys
        }

        outputs = {
            "logits": logits,
            "labels": labels,
            "completion_mask": completion_mask,
        }

        return outputs

    def compute_segment(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        completion_mask: torch.Tensor,
    ):
        entropy = self.H(logits)
        entropy_scaled = entropy / entropy.amax(-1, keepdim=True)
        masks = self.compute_mask(entropy_scaled, completion_mask)
        return masks

    def format_messages(
        self,
        prompt: str,
        completion: str,
        lang: str,
    ):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": f"Translate the following text into {lang}.\n\n{prompt}",
                }
            ],
            "completion": [{"role": "assistant", "content": completion}],
        }

    def tokenize_fn(self, text, tokenizer: PreTrainedTokenizerBase, **kwargs):
        return tokenizer.apply_chat_template(
            text, tokenize=True, return_dict=True, **kwargs
        )

    def preprocess_batch(self, prompt: str, completions: List[str], lang: str):
        data = []
        for completion in completions:
            sample = self.format_messages(prompt, completion, lang)
            data.append(sample)
        return data

    def prepare_sample(self, sample):
        output = {}
        prompt_ids = self.tokenize_fn(
            sample["prompt"],
            tokenizer=self.tokenizer,
            add_generation_prompt=True,
        )["input_ids"]

        prompt_completion_processed = self.tokenize_fn(
            sample["prompt"] + sample["completion"],
            tokenizer=self.tokenizer,
        )

        prompt_completion_ids = prompt_completion_processed["input_ids"]
        attention_mask = prompt_completion_processed["attention_mask"]
        completion_mask = [0] * len(prompt_ids) + [1] * (
            len(prompt_completion_ids) - len(prompt_ids)
        )

        output["input_ids"] = torch.tensor(prompt_completion_ids)
        output["attention_mask"] = torch.tensor(attention_mask)
        output["completion_mask"] = torch.tensor(completion_mask)

        return output

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

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
        }

    def get_batch(
        self,
        prompt: str | None = None,
        completions: List[str] | None = None,
        lang: str | None = None,
        samples: List[List[dict]] | List[dict] | None = None,
    ):
        if samples is None:
            samples = self.preprocess_batch(prompt, completions, lang)
        batch = self.collate_fn(samples)
        return batch

    def compute(
        self,
        prompts: List[str] | None = None,
        completions: List[List[str]] | None = None,
        lang: str | None = None,
        batch: List[List[dict]]
        | List[tuple[dict, dict]]
        | tuple[list, list]
        | None = None,
    ):
        masks = []
        if batch:
            batch = self.get_batch(samples=batch)
            outputs = self.model_forward(batch)
            seg_masks = [
                {"segment_mask": seg_mask}
                for seg_mask in self.compute_segment(**outputs)
            ]
            return seg_masks
        for prompt, completion in tqdm(
            zip(prompts, completions), total=len(prompts), desc="Segmenting..."
        ):
            batch = self.get_batch(prompt, completion, lang)
            outputs = self.model_forward(batch)
            seg_masks = self.compute_segment(**outputs)
            masks.append(seg_masks)
        return masks

    def cuda(self):
        return self.model.cuda()

    def cpu(self):
        return self.model.cpu()
