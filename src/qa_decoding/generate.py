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

from reranker import RerankerConfig
from segmenter import Segmenter
from qa_decode import vllm_pipeline, hf_pipeline, seg_pipeline
from utils import GenerationConfig, vLLMGenerationConfig, load_dataset

torch.cuda.empty_cache()
gc.collect()
torch.cuda.reset_peak_memory_stats()

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--tgt_lang', type=str, default=None)
    parser.add_argument('--best_of', type=int, default=None)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--vllm', action='store_true', default=False)
    parser.add_argument('--segment_level', action='store_true', default=False)
    return parser

def load_segmenter():
    model_path = "t_index_reproduce/models/sft/qwen2.5-0.5b-mixture-5000-10/negative"
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    segmenter = Segmenter(model=model, tokenizer=tokenizer)
    return segmenter

if __name__ == "__main__":
    parser = init_parser()
    args, kwargs = parser.parse_known_args()
    gen_config = vLLMGenerationConfig if args.vllm else GenerationConfig
    hf_parser = HfArgumentParser([gen_config, RerankerConfig])
    generation_kwargs, reranker_args = hf_parser.parse_args_into_dataclasses(args=kwargs)

    dataset_loader = partial(load_dataset, args.data_path, args.tgt_lang, convert_chat_template=args.vllm)
    if args.segment_level:
        segmenter = load_segmenter()

    if args.vllm or args.segment_level:
        model = LLM(
            args.model,
            seed=42,
            quantization="bitsandbytes",
            gpu_memory_utilization=0.85 if args.segment_level else 0.92,
            dtype="bfloat16",
            distributed_executor_backend="mp",
            cpu_offload_gb=4,
        )

        dataset = dataset_loader(tokenizer=model.get_tokenizer())
        print(dataset[:3])
        sampling_params = SamplingParams(
            n=args.best_of,
            seed=42,
            **asdict(generation_kwargs)
        )

        if args.segment_level:
            preds = seg_pipeline(
                model,
                texts=dataset,
                batch_size=args.batch_size,
                reranker_args=asdict(reranker_args),
                sampling_params=sampling_params
            )

        else:
            preds = vllm_pipeline(
                model,
                dataset,
                batch_size=args.batch_size,
                reranker_args=asdict(reranker_args),
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
        
        tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left')

        dataset = load_dataset(args.data_path, args.tgt_lang)
        preds = hf_pipeline(
            model,
            tokenizer,
            dataset,
            batch_size=16,
            best_of=args.best_of,
            reranker_args=asdict(reranker_args),
            generation_kwargs=asdict(generation_kwargs)
        )

    os.makedirs('/'.join(args.output_file.split('/')[:-1]), exist_ok=True)
    with open(args.output_file, "w", encoding='utf-8') as file:
        json.dump(preds, file, indent=2, ensure_ascii=False)