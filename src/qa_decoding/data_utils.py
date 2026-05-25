from transformers import PreTrainedTokenizerBase
from datasets import Dataset
from datasets import load_dataset as load_ds
from functools import partial
from typing import List, Literal
from pathlib import Path
import random
import torch
import numpy as np


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
    "cmn_Hans": "Chinese",
}

flores_codes = {v: k for k, v in flores_names.items()}

name2code = {v: k for v, k in code2name.items()}


def load_dataset(
    data_path="NTREX/NTREX-128",
    tgt_lang: str = "zho",
    format_message: Literal["translategemma", "messages"] = "messages",
    convert_chat_template: bool = False,
    tokenizer: PreTrainedTokenizerBase | None = None,
):
    random.seed(42)
    if Path(data_path).exists():
        with open(f"{data_path}/newstest2019-src.eng.txt", "r") as file:
            srcs = file.readlines()
        if tgt_lang == "zho":
            lang_code = tgt_lang + "-CN"
        else:
            lang_code = tgt_lang
        with open(
            f"{data_path}/newstest2019-ref.{lang_code}.txt",
            "r",
            encoding="utf-8",
        ) as file:
            refs = file.readlines()
        data = []
        for src, ref in zip(srcs, refs):
            data.append({"src": src, "ref": ref})
        # data = random.sample(data, 100)
        dataset = Dataset.from_list(data)
    else:
        src_ds = load_ds(data_path, "eng_Latn", split="devtest")
        lang_code = flores_codes[code2name[tgt_lang]]
        ref_ds = load_ds(data_path, lang_code, split="devtest")
        data = []
        for src, ref in zip(src_ds, ref_ds):
            data.append({"src": src["text"], "ref": ref["text"]})
        # data = random.sample(data, 100)
        dataset = Dataset.from_list(data)
    if format_message == "messages":
        format_fn = partial(format_messages, lang=code2name[tgt_lang])
    elif format_message == "translategemma":
        format_fn = partial(format_prompts_for_translategemma, lang=tgt_lang)
    dataset = dataset.map(format_fn, batched=True)
    if convert_chat_template:
        dataset = dataset.map(
            apply_chat_template,
            fn_kwargs={"tokenizer": tokenizer},
            batched=True,
        )

    return dataset


def format_messages(examples, lang):
    messages = []
    for src in examples["src"]:
        messages.append(
            [
                {
                    "role": "user",
                    "content": f"Translate the following text into {lang}. Return ONLY the translation, without commentary or explaination.\n\n{src}",
                }
            ]
        )

    return {"messages": messages}


def format_prompts_for_translategemma(examples, lang):
    prompts = []
    lang_map = {"msa": "ms", "zho": "zh-CN", "kor": "ko-KR"}
    for example in examples["src"]:
        prompts.append(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "source_lang_code": "en",
                            "target_lang_code": lang_map[lang],
                            "text": example,
                        }
                    ],
                }
            ]
        )
    return {"messages": prompts}


def apply_chat_template(examples, tokenizer: PreTrainedTokenizerBase):
    prompts = []
    for text in examples["messages"]:
        prompt = tokenizer.apply_chat_template(
            text,
            enable_thinking=False,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
    return {"prompt": prompts}


def pad(
    tensors: list[torch.Tensor],
    padding_value: int = 0,
    padding_side: str = "right",
    pad_to_multiple_of: int | None = None,
) -> torch.Tensor:
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    if pad_to_multiple_of is not None:
        remainder = output_shape[0] % pad_to_multiple_of
        if remainder != 0:
            output_shape[0] += pad_to_multiple_of - remainder

    output = torch.full(
        (len(tensors), *output_shape),
        padding_value,
        dtype=tensors[0].dtype,
        device=tensors[0].device,
    )

    for i, t in enumerate(tensors):
        if padding_side == "left":
            seq_start = output_shape[0] - t.shape[0]
        elif padding_side == "right":
            seq_start = 0
        else:
            raise ValueError("Invalid padding_side.")

        seq_slice = slice(seq_start, seq_start + t.shape[0])
        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output


def collate_fn(examples, pad_token_id):
        input_ids = [example["input_ids"] for example in examples]
        attention_mask = [example["attention_mask"] for example in examples]
        completion_mask = [example["completion_mask"] for example in examples]

        input_ids = pad(
            input_ids,
            padding_value=pad_token_id,
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

def prepare_data(
    prompts: List[str] | str | None = None,
    completions: List[List[str]] | List[str] | None = None,
    lang: str | None = None,
    batches: List[List[dict]] | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
    return_dict: bool = True,
):
    def format_messages(
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

    def tokenize_fn(text, tokenizer: PreTrainedTokenizerBase, **kwargs):
        return tokenizer.apply_chat_template(
            text, tokenize=True, return_dict=True, **kwargs
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
            sample["prompt"],
            tokenizer=tokenizer,
            add_generation_prompt=True,
        )["input_ids"]

        prompt_completion_processed = tokenize_fn(
            sample["prompt"] + sample["completion"],
            tokenizer=tokenizer,
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

    def get_batch(
        prompt: str | None = None,
        completions: List[str] | None = None,
        lang: str | None = None,
        batch: List[List[dict]] | List[dict] | None = None,
    ):
        if batch is None:
            batch = [
                _preprocess(sample)
                for sample in _prepare_sample(prompt, completions, lang)
            ]
        if return_dict:
            return collate_fn(batch, tokenizer.pad_token_id)
        return batch

    if prompts is not None:
        if isinstance(prompts, str):
            return get_batch(prompts, completions, lang)
        else:
            batches = [
                get_batch(prompt, completions[i], lang)
                for i, prompt in enumerate(prompts)
            ]
            return batches
    elif batches is not None:
        return [get_batch(batch=batch) for batch in batches]
    return

def unbatch(dataset: Dataset):
    ds = {}
    for batch in dataset:
        for key, values in batch.items():
            if ds.get(key) is None:
                ds[key] = []
            ds[key] += values
    return Dataset.from_dict(ds)