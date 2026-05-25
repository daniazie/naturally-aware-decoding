from dataclasses import asdict, dataclass
from functools import partial
import torch
import numpy as np
import json
import os

from vllm import LLM, SamplingParams
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)

from qa_decode import vllm_pipeline, hf_pipeline, flush
from data_utils import load_dataset


@dataclass
class GenerationConfig:
    top_k: int | None = None
    num_beams: int | None = None
    do_sample: bool = False
    temperature: float | None = None
    max_new_tokens: int = 1024
    output_logits: bool = False
    return_dict_in_generate: bool = False
    top_h: float | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    use_cache: bool = True


@dataclass
class vLLMGenerationConfig:
    top_k: int = None
    temperature: float = None
    top_p: float | None = None
    max_tokens: int = 1024
    repetition_penalty: float = 0.5
    presence_penalty: float = 0.5
    frequency_penalty: float = 0.5


@dataclass
class ModelArgs:
    attn_implementation: str = "sdpa"
    dtype: torch.dtype = torch.bfloat16
    device_map: str = "auto"


def vllm_generator(dataset_loader, args, generation_kwargs, rerank_args=None):
    model = LLM(
        args.model,
        seed=42,
        quantization="bitsandbytes",
        gpu_memory_utilization=0.80 if args.reranker_type is not None else 0.92,
        dtype="bfloat16",
        cpu_offload_gb=4,
        distributed_executor_backend="mp",
    )

    dataset = dataset_loader(tokenizer=model.get_tokenizer())
    print(dataset[:3])
    generation_kwargs = {
        k: v for k, v in asdict(generation_kwargs).items() if v is not None
    }

    sampling_params = SamplingParams(
        n=args.best_of if args.reranker_type is not None else 1,
        seed=42,
        max_tokens=512,
        # **generation_kwargs,
    )

    preds = vllm_pipeline(
        model,
        dataset,
        batch_size=args.batch_size,
        granularity=args.granularity,
        per_segment_eval=args.per_segment_eval,
        reranker_args=rerank_args,
        reranker_type=args.reranker_type,
        sampling_params=sampling_params
    )

    flush()
    return preds


def hf_generator(dataset_loader, args, generation_kwargs, rerank_args=None):
    set_seed(42)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_method="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_4",
    )
    model = torch.compile(model, mode="max-autotune")

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")

    dataset = dataset_loader()
    preds = hf_pipeline(
        model,
        tokenizer,
        dataset,
        batch_size=16,
        best_of=args.best_of,
        reranker_args=rerank_args,
        reranker_type=args.reranker_type,
        generation_kwargs=asdict(generation_kwargs),
    )

    return preds