from datasets import Dataset
from typing import List, Dict
from dataclasses import asdict

from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, BitsAndBytesConfig, set_seed
from functools import partial
from tqdm import tqdm
import torch

import argparse
import json
import os
import gc

from reranker import NatArgs, CometConfig, RerankerConfig
from segmenter import Segmenter
from qa_decode import vllm_pipeline, hf_pipeline
from utils import GenerationConfig, vLLMGenerationConfig, load_dataset

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--tgt_lang', type=str, default=None)
    parser.add_argument('--best_of', type=int, default=None)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--vllm', action='store_true', default=False)
    parser.add_argument('--reranker_type', choices=['natural', 'comet', 'combined', 'none'], default=None)
    parser.add_argument('--granularity', choices=["token", "segment", "sequence"], default=None)
    return parser

if __name__ == "__main__":
    parser = init_parser()
    args, kwargs = parser.parse_known_args()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    gen_config = vLLMGenerationConfig if args.vllm else GenerationConfig
    reranker_type = args.reranker_type if (args.reranker_type != "none") else None
    if reranker_type is not None:
        if reranker_type == 'natural':
            reranker_config = NatArgs
        elif reranker_type == 'comet':
            reranker_config = CometConfig
        elif reranker_type == 'combined':
            reranker_config = RerankerConfig
        hf_parser = HfArgumentParser([gen_config, reranker_config])
        generation_kwargs, rerank_args = hf_parser.parse_args_into_dataclasses(args=kwargs)
    else:
        rerank_args = None
        hf_parser = HfArgumentParser([gen_config])
        generation_kwargs = hf_parser.parse_args_into_dataclasses(args=kwargs)
        if isinstance(generation_kwargs, tuple):
            generation_kwargs = generation_kwargs[0]

    dataset_loader = partial(load_dataset, args.data_path, args.tgt_lang, convert_chat_template=args.vllm)
    if args.vllm:
        model = LLM(
            args.model,
            seed=42,
            quantization="bitsandbytes",
            gpu_memory_utilization=0.7849 if args.granularity == 'segment' or reranker_type != 'natural' else 0.92,
            dtype="bfloat16",
            distributed_executor_backend="mp",
            cpu_offload_gb=4,
        )

        dataset = dataset_loader(tokenizer=model.get_tokenizer())
        print(dataset[:3])
        sampling_params = SamplingParams(
            n=args.best_of if reranker_type is not None else 1,
            seed=42,
            **asdict(generation_kwargs)
        )

        preds = vllm_pipeline(
            model,
            dataset,
            batch_size=args.batch_size,
            granularity=args.granularity,
            reranker_args=rerank_args,
            reranker_type=reranker_type,
            sampling_params=sampling_params
        )
    else:
        set_seed(42)
        quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_method="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

        dataset = dataset_loader()

        model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto', quantization_config=quantization_config, dtype=torch.bfloat16)
        model = torch.compile(model, mode="max-autotune")
        
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        dataset = load_dataset(args.data_path, args.tgt_lang)
        preds = hf_pipeline(
            model,
            tokenizer,
            dataset,
            batch_size=16,
            best_of=args.best_of,
            reranker_args=asdict(rerank_args),
            reranker_type=reranker_type,
            generation_kwargs=asdict(generation_kwargs)
        )

    output_file = args.output_file if not "none" in args.output_file else args.output_file.replace(f"none", "unranked")

    os.makedirs('/'.join(output_file.split('/')[:-1]), exist_ok=True)
    with open(args.output_file, "w", encoding='utf-8') as file:
        json.dump(preds, file, indent=2, ensure_ascii=False)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()