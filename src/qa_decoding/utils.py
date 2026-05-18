from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from dataclasses import dataclass
from datasets import Dataset
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

code2name = {
    "zho": "Chinese",
    "fra": "French",
    "deu": "German",
    "msa": "Malay",
    "kor": "Korean",
}

def load_dataset(data_path="NTREX/NTREX-128", tgt_lang: str = "zho", convert_chat_template: bool = False, tokenizer: PreTrainedTokenizerBase | None = None):
    with open(f"{data_path}/newstest2019-src.eng.txt", "r") as file:
        src = file.readlines()
    if tgt_lang == "zho":
        lang_code = tgt_lang + "-CN"
    else:
        lang_code = tgt_lang
    with open(f"{data_path}/newstest2019-ref.{lang_code}.txt", "r") as file:
        ref = file.readlines()
    dataset =  Dataset.from_dict({"src": src, "ref": ref})
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