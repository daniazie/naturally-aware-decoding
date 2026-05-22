from dataclasses import asdict
from functools import partial
import torch
import numpy as np
import json
import os

from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from qa_decode import vllm_pipeline, hf_pipeline, tune_pipeline, flush
from data_utils import load_dataset

def vllm_generator(dataset_loader, args, generation_kwargs, rerank_args=None):
    model = LLM(
        args.model,
        seed=42,
        quantization="bitsandbytes",
        gpu_memory_utilization=0.82 if args.reranker_type is not None else 0.92,
        dtype="bfloat16",
        distributed_executor_backend="mp",
        cpu_offload_gb=4,
    )

    dataset = dataset_loader(tokenizer=model.get_tokenizer())
    print(dataset[:3])
    sampling_params = SamplingParams(
        n=args.best_of if args.reranker_type is not None else 1,
        seed=42,
        **asdict(generation_kwargs)
    )
    
    preds = vllm_pipeline(
        model,
        dataset,
        batch_size=args.batch_size,
        granularity=args.granularity,
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

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto', quantization_config=quantization_config, dtype=torch.bfloat16)
    model = torch.compile(model, mode="max-autotune")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    dataset = dataset_loader()
    preds = hf_pipeline(
        model,
        tokenizer,
        dataset,
        batch_size=16,
        best_of=args.best_of,
        reranker_args=asdict(rerank_args),
        reranker_type=args.reranker_type,
        generation_kwargs=asdict(generation_kwargs)
    )

    output_file = args.output_file if not "none" in args.output_file else args.output_file.replace(f"none", "unranked")

    return preds

def tune(args, generation_kwargs):
    dataset_loader = partial(load_dataset, "openlanguagedata/flores_plus", args.tgt_lang, convert_chat_template=args.vllm, split='dev')
    model = LLM(
        args.model,
        seed=42,
        quantization="bitsandbytes",
        gpu_memory_utilization=0.82,
        dtype="bfloat16",
        distributed_executor_backend="mp",
        cpu_offload_gb=4,
    )

    dataset = dataset_loader(tokenizer=model.get_tokenizer())
    sampling_params = SamplingParams(
        n=args.best_of,
        seed=42,
        **asdict(generation_kwargs)
    )
    W_nat, W_comet = tune_pipeline(
        model,
        dataset,
        tgt_lang=args.tgt_lang,
        batch_size=args.batch_size,
        sampling_params=sampling_params,
        granularity=args.granularity
    )

    W_nat, W_comet = np.array(W_nat), np.array(W_comet)

    weights = {
        "W_nat": {
            "mean": W_nat.mean().item(),
            "median": W_nat.median().item()
        },
        "W_comet": np.array(W_comet).mean().item(),

    }
    os.makedirs('/'.join(args.output_file.split('/')[:-1]), exist_ok=True)
    with open(args.output_file, "w", encoding='utf-8') as file:
        json.dump(weights, file, indent=2, ensure_ascii=False)