from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from functools import partial
import torch.nn.functional as F
import torch

from dataclasses import dataclass
from tqdm import tqdm
from typing_extensions import List, Literal, Optional
import numpy as np

code2name = {
    "zho": "Chinese",
    "fra": "French",
    "deu": "German",
    "msa": "Malay"
}

def pad(
    tensors: list[torch.Tensor],
    padding_value: int = 0,
    padding_side: str = 'left',
    pad_to_multiple_of: int | None = None,
) -> torch.Tensor:
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    if pad_to_multiple_of is not None:
        remainder = output_shape[0] % pad_to_multiple_of
        if remainder != 0:
            output_shape[0] += pad_to_multiple_of - remainder

    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        if padding_side == 'left':
            seq_start = output_shape[0] - t.shape[0]
        elif padding_side == 'right':
            seq_start = 0
        else:
            raise ValueError("Invalid padding_side.")
        
        seq_slice = slice(seq_start, seq_start + t.shape[0])
        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output

@dataclass
class RewardOutputs:
    mean: Optional[list] = None
    z_score: Optional[list] = None

@dataclass
class TranslationeseResults:
    token_level: Optional[RewardOutputs]
    seq_level: Optional[RewardOutputs]

class TranslationeseIndex:
    def __init__(self, model_path, **model_kwargs):
        positive_model = AutoModelForCausalLM.from_pretrained(f"{model_path}/positive", **model_kwargs)
        negative_model = AutoModelForCausalLM.from_pretrained(f"{model_path}/negative", **model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/positive")
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
            if self.reward_normalise_type == "mean":
                seq_logps = logps.sum(dim=1) / completion_mask.sum(dim=1)
                return seq_logps.cpu()
            elif self.reward_normalise_type == "z_score":
                z_scores = (logps - logps.mean(dim=1).unsqueeze(1)) / logps.var(dim=1).unsqueeze(1)
                return z_scores.mean(dim=1).cpu()
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
    
    def _prepare_sample(self, example):
        output = {}
        prompt_ids = self.tokenize_fn(example['src'], self.tokenizer, add_generation_prompt=True)['input_ids']
        prompt_completion_processed = self.tokenize_fn(example['src'] + example['mt'], self.tokenizer)

        prompt_completion_ids = prompt_completion_processed['input_ids']
        completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
        attention_mask = prompt_completion_processed['attention_mask']

        output['input_ids'] = prompt_completion_ids
        output['attention_mask'] = attention_mask
        output['completion_mask'] = completion_mask
        return output

    def prepare_batch(self, examples):
        outputs = [self._prepare_sample(example) for example in examples]
        return outputs
    
    def collate_fn(self, examples: list[dict]):
        input_ids = [example['input_ids'] for example in examples]
        attention_mask = [example['attention_mask'] for example in examples]
        completion_mask = [example['completion_mask'] for example in examples]

        input_ids = [torch.tensor(ids) for ids in input_ids]
        attention_mask = [torch.tensor(mask) for mask in attention_mask]
        completion_mask = [torch.tensor(mask) for mask in completion_mask]

        input_ids = pad(
            input_ids,
            padding_value=self.tokenizer.pad_token_id,
            padding_side="right",
        )
    
        attention_mask = pad(
            attention_mask,
            padding_value=0,
            padding_side="right",
        )

        completion_mask = pad(
            completion_mask,
            padding_value=0,
            padding_side="right",
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask
        }

    def compute_rewards(self, log_lklh_positive: torch.Tensor, log_lklh_negative: torch.Tensor, mask: torch.Tensor | None = None):
        logps_ratios = log_lklh_positive - log_lklh_negative
        if self.granularity == 'token':
            if self.reward_normalise_type == "mean":
                rewards = logps_ratios.sum(dim=1) / mask.sum(dim=1)
            elif self.reward_normalise_type == "z_score":
                rewards = (logps_ratios - logps_ratios.mean(dim=1).unsqueeze(1)) / logps_ratios.var(dim=1).unsqueeze(1)
                rewards = rewards.mean(dim=1)
        return rewards

    def score(self, srcs: List[str], mts: List[str], lang: str, granularity: Literal['token', 'sequence', 'all'] = 'token', normalise_score: bool = False, batch_size: int = 4, reward_normalise_type: Literal["mean", "z_score", "all"] = "mean"):
        self.granularity = granularity
        self.reward_normalise_type = reward_normalise_type
        rewards = []
        hyps = [self.format_messages(src, mt, lang) for src, mt in zip(srcs, mts)]
        for i in tqdm(range(0, len(hyps), batch_size), desc="Evaluating..."):
            batch = self.prepare_batch(hyps[i:i+batch_size])
            batch = self.collate_fn(batch)
            log_lklh_high = self.compute_positive(batch)
            log_lklh_low = self.compute_negative(batch)

            if self.granularity == "token":
                mask = batch["completion_mask"]
            else:
                mask = None

            per_batch_rewards = self.compute_rewards(log_lklh_low, log_lklh_high, mask=mask)
            if normalise_score:
                per_batch_rewards = per_batch_rewards.sigmoid()
            rewards += per_batch_rewards.tolist()
        return {"score": rewards, "mean_score": np.mean(rewards).item()}