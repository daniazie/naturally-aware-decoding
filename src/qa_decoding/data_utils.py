from transformers import PreTrainedTokenizerBase
from dataclasses import dataclass
from datasets import Dataset
from datasets import load_dataset as load_ds
from typing import List
from pathlib import Path
import random
import torch
import numpy as np

@dataclass
class GenerationConfig:
    top_k: int = 20
    num_beams: int | None = None
    do_sample: bool = True
    top_p: float = 0.95
    temperature: float = 0.6
    max_new_tokens: int = 1024

@dataclass
class vLLMGenerationConfig:
    top_k: int = 20
    top_p: float = 0.95
    temperature: float = 0.6
    max_tokens: int = 1024

@dataclass
class ModelArgs:
    attn_implementation: str = "sdpa"
    dtype: torch.dtype = torch.bfloat16
    device_map: str = "auto"

code2name = {
    "zho": "Chinese",
    "fra": "French",
    "deu": "German",
    "msa": "Malay",
    "kor": "Korean",
}

flores_names = {
    "zsm_Latn": "Malay",
    "kor_Hang": "Korean",
    "cmn_Hans": "Chinese"
}

flores_codes = {
    v: k
    for k, v in flores_names.items()
}

name2code = {
    v: k 
    for v, k in code2name.items()
}

def load_dataset(data_path="NTREX/NTREX-128", tgt_lang: str = "zho", convert_chat_template: bool = False, split: str | None = None, tokenizer: PreTrainedTokenizerBase | None = None):
    random.seed(42)
    if Path(data_path).exists():
        with open(f"{data_path}/newstest2019-src.eng.txt", "r") as file:
            srcs = file.readlines()
        if tgt_lang == "zho":
            lang_code = tgt_lang + "-CN"
        else:
            lang_code = tgt_lang
        with open(f"{data_path}/newstest2019-ref.{lang_code}.txt", "r", encoding='utf-8') as file:
            refs = file.readlines()
        data = []
        for src, ref in zip(srcs, refs):
            data.append({
                "src": src,
                "ref": ref
            })
        data = random.sample(data, 100)
        dataset =  Dataset.from_list(data)
    else:
        src_ds = load_ds(data_path, "eng_Latn", split=split)
        lang_code = flores_codes[code2name[tgt_lang]]
        ref_ds = load_ds(data_path, lang_code, split=split)
        data = []
        for src, ref in zip(src_ds, ref_ds):
            data.append({
                "src": src['text'],
                "ref": ref['text']
            })
        data = random.sample(data, 100)
        dataset = Dataset.from_list(data)
    dataset = dataset.map(format_messages, fn_kwargs={"lang": code2name[tgt_lang]}, batched=True)
    if convert_chat_template:
        dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer}, batched=True)
        
    return dataset

def format_messages(examples, lang):
    messages = []
    for src in examples['src']:
        messages.append([
            {"role": "user", "content": f"Translate the following text into {lang}. Return ONLY the translation, without commentary or explaination.\n\n{src}"}
        ])

    return {"messages": messages}

def apply_chat_template(examples, tokenizer: PreTrainedTokenizerBase):
    prompts = []
    for text in examples['messages']:
        prompt = tokenizer.apply_chat_template(
            text,
            enable_thinking=False,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)
    return {"prompt": prompts}

def pad(
    tensors: list[torch.Tensor],
    padding_value: int = 0,
    padding_side: str = 'right',
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

def prepare_data(prompts: List[str] | str | None = None, completions: List[List[str]] | List[str] | None = None, lang: str | None = None, batches: List[List[dict]] | None = None, tokenizer: PreTrainedTokenizerBase | None = None):
    def format_messages(prompt: str, completion: str, lang: str,):
        return {
            "prompt": [
                {"role": "user", "content": f"Translate the following text into {lang}.\n\n{prompt}"}
            ],
            "completion": [
                {"role": "assistant", "content": completion}
            ]
        }
    
    def tokenize_fn(text, tokenizer: PreTrainedTokenizerBase, **kwargs):
        return tokenizer.apply_chat_template(
            text,
            tokenize=True,
            return_dict=True,
            **kwargs
        )

    def _prepare_sample(prompt: str, completions: List[str], lang: str):
        data = []
        for completion in completions:
            sample = format_messages(prompt, completion, lang)
            data.append(sample)
        return data
    
    def _preprocess(sample):
        output = {}
        prompt_ids = tokenize_fn(
            sample['prompt'],
            tokenizer=tokenizer,
            add_generation_prompt=True,
        )['input_ids']

        prompt_completion_processed = tokenize_fn(
            sample['prompt'] + sample['completion'],
            tokenizer=tokenizer,
        )

        prompt_completion_ids = prompt_completion_processed['input_ids']
        attention_mask = prompt_completion_processed['attention_mask']
        completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))

        output['input_ids'] = torch.tensor(prompt_completion_ids)
        output['attention_mask'] = torch.tensor(attention_mask)
        output['completion_mask'] = torch.tensor(completion_mask)

        return output
    
    def collate_fn(examples):
        input_ids = [example['input_ids'] for example in examples]
        attention_mask = [example['attention_mask'] for example in examples]
        completion_mask = [example['completion_mask'] for example in examples]

        input_ids = pad(
            input_ids,
            padding_value=tokenizer.pad_token_id,
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
            "completion_mask": completion_mask
        }
    
    def get_batch(prompt: str | None = None, completions: List[str] | None = None, lang: str | None = None, batch: List[List[dict]] | List[dict] | None = None):
        if batch is None:
            batch = [_preprocess(sample) for sample in _prepare_sample(prompt, completions, lang)]
        batch = collate_fn(batch)
        return batch
    
    if prompts is not None:
        if isinstance(prompts, str):
            return get_batch(prompts, completions, lang)
        else:
            batches = [get_batch(prompt, completions[i], lang) for i, prompt in enumerate(prompts)]
            return batches
    elif batches is not None:
        return [get_batch(batch=batch) for batch in batches]
    return

