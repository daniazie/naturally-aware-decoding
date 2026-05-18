from transformers.data.data_collator import DataCollatorMixin
from torch.utils.data import Dataset
import torch
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os

def pad(
    tensors: list[torch.Tensor],
    padding_value: int = 0,
    padding_side: str = 'right',
    pad_to_multiple_of: int | None = None,
    max_length: int | None = None
) -> torch.Tensor:
    output_shape = [max_length] if max_length else np.max([t.shape for t in tensors], 0).tolist()

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

def extract_prompt(example, chosen_key, rejected_key):
    for idx in range(min(len(example[chosen_key]), len(example[rejected_key]))):
        if example[chosen_key][idx] != example[rejected_key][idx]:
            if example[chosen_key][idx - 1] == " ":
                idx -= 1
            break
    return {
        "instruction": example[chosen_key][:idx],
        chosen_key: example[chosen_key][idx:],
        rejected_key: example[rejected_key][idx:],
    }

def is_conversational(example):
    for key, value in example.items():
        if isinstance(value, list):
            if isinstance(value[0], dict) and "role" in value[0].keys():
                return True
    return False

@dataclass
class DataCollatorForPairedPreference(DataCollatorMixin):
    pad_token_id: int
    max_length: int | None = None
    truncation_mode: str = "keep_start"
    padding_side: str = 'right'
    pad_to_multiple_of: int | None = None
    return_tensors: str = "pt"

    def prepare_data(self, examples):
        prompt_chosen_ids = [example["prompt_ids"] + example['chosen_ids'] for example in examples]
        prompt_rejected_ids = [example['prompt_ids'] + example['rejected_ids'] for example in examples]
        chosen_mask = [[0] * len(example['prompt_ids']) +  [1] * len(example['chosen_ids']) for example in examples]
        rejected_mask = [[0] * len(example['prompt_ids']) +  [1] * len(example['rejected_ids']) for example in examples]

        if self.max_length is not None:
            if self.truncation_mode == "keep_start":
                sl = slice(None, self.max_length)
            elif self.truncation_mode == 'keep_end':
                sl = slice(-self.max_length, None)
            else:
                raise ValueError(f"Unsupported truncation type.")
            
            prompt_chosen_ids = [ids[sl] for ids in prompt_chosen_ids]
            prompt_rejected_ids = [ids[sl] for ids in prompt_rejected_ids]
            chosen_mask = [mask[sl] for mask in chosen_mask]
            rejected_mask = [mask[sl] for mask in rejected_mask]

        chosen_attention_mask = [[1] * len(ids) for ids in prompt_chosen_ids]
        rejected_attention_mask = [[1] * len(ids) for ids in prompt_rejected_ids]

        input_ids = prompt_chosen_ids + prompt_rejected_ids
        completion_mask = chosen_mask + rejected_mask
        attention_mask = chosen_attention_mask + rejected_attention_mask
        
        return input_ids, attention_mask, completion_mask
    

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids, attention_mask, completion_mask = self.prepare_data(examples)
        
        input_ids = [torch.tensor(ids) for ids in input_ids]
        attention_mask = [torch.tensor(mask) for mask in attention_mask]
        completion_mask = [torch.tensor(mask) for mask in completion_mask]

        output = {}
        output['input_ids'] = pad(
            input_ids,
            padding_value=self.pad_token_id,
            padding_side=self.padding_side,
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_length
        )

        output['attention_mask'] = pad(
            attention_mask,
            padding_value=0,
            padding_side=self.padding_side,
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_length
        )

        output['completion_mask'] = pad(
            completion_mask,
            padding_value=0,
            padding_side=self.padding_side,
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_length
        )
        
        return output
    

@dataclass
class DataCollatorForUnpairedPreference(DataCollatorMixin):
    pad_token_id: int
    max_length: int | None = None
    truncation_mode: str = "keep_start"
    padding_side: str = 'right'
    pad_to_multiple_of: int | None = None
    return_tensors: str = "pt"

    def prepare_data(self, examples):
        prompt_completion_ids = [example["prompt_ids"] + example['completion_ids'] for example in examples]
        completion_mask = [[0] * len(example['prompt_ids']) +  [1] * len(example['completion_ids']) for example in examples]
        labels = [example['labels'] for example in examples]

        if self.max_length is not None:
            if self.truncation_mode == "keep_start":
                sl = slice(None, self.max_length)
            elif self.truncation_mode == 'keep_end':
                sl = slice(-self.max_length, None)
            else:
                raise ValueError(f"Unsupported truncation type.")
            
            prompt_completion_ids = [ids[sl] for ids in prompt_completion_ids]
            completion_mask = [mask[sl] for mask in completion_mask]

        attention_mask = [[1] * len(ids) for ids in prompt_completion_ids]

        return prompt_completion_ids, completion_mask, attention_mask, labels
    

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_completion_ids, completion_mask, attention_mask, labels = self.prepare_data(examples)
        
        prompt_completion_ids = [torch.tensor(ids) for ids in prompt_completion_ids]
        attention_mask = [torch.tensor(mask) for mask in attention_mask]
        completion_mask = [torch.tensor(mask) for mask in completion_mask]
        labels = [torch.tensor(label) for label in labels]

        prompt_completion_ids = pad(
            prompt_completion_ids,
            padding_value=self.pad_token_id,
            padding_side=self.padding_side,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        attention_mask = pad(
            attention_mask,
            padding_value=0,
            padding_side=self.padding_side,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        completion_mask = pad(
            completion_mask,
            padding_value=0,
            padding_side=self.padding_side,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        labels = pad(
            labels,
            padding_value=self.pad_token_id,
            padding_side=self.padding_side, 
            pad_to_multiple_of=self.pad_to_multiple_of
        )

        output = {}
        output['input_ids'] = prompt_completion_ids
        output['attention_mask'] = attention_mask
        output['completion_mask'] = completion_mask
        output['labels'] = labels
    
        return output