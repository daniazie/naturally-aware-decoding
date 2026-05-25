from dataclasses import asdict

from transformers import HfArgumentParser, set_seed
from functools import partial
import torch
import warnings
import argparse
import json
import os
import gc

from rerankers import RatioArgs, LikelihoodArgs, CometArgs, RerankerArgs
from data_utils import load_dataset
from gen_utils import (
    GenerationConfig,
    vLLMGenerationConfig,
    vllm_generator,
    hf_generator,
    flush
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--tgt_lang", type=str, default=None)
    parser.add_argument("--best_of", type=int, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--vllm", action="store_true", default=False)
    parser.add_argument(
        "--reranker_type",
        choices=["ratios", "likelihood", "self", "comet", "combined", "none"],
        default=None,
    )
    parser.add_argument(
        "--granularity", choices=["token", "segment", "sequence"], default=None
    )
    parser.add_argument("--per_segment_eval", action="store_true", default=False)
    return parser


def parse_args(args):
    gen_config = vLLMGenerationConfig if args.vllm else GenerationConfig
    reranker_type = args.reranker_type if (args.reranker_type != "none") else None
    setattr(args, "reranker_type", reranker_type)
    if reranker_type is not None:
        if reranker_type == "ratios":
            reranker_config = RatioArgs
        elif reranker_type == "likelihood" or reranker_type == "self":
            reranker_config = LikelihoodArgs
        elif reranker_type == "comet":
            reranker_config = CometArgs
        elif reranker_type == "combined":
            reranker_config = RerankerArgs
        hf_parser = HfArgumentParser([gen_config, reranker_config])
        generation_kwargs, rerank_args = hf_parser.parse_args_into_dataclasses(
            args=kwargs
        )
        if (
            hasattr(rerank_args, "tgt_lang")
            and asdict(rerank_args).get("tgt_lang") is None
        ):
            setattr(rerank_args, "tgt_lang", args.tgt_lang)
        rerank_args = asdict(rerank_args)
    else:
        rerank_args = None
        hf_parser = HfArgumentParser([gen_config])
        generation_kwargs = hf_parser.parse_args_into_dataclasses(args=kwargs)
        if isinstance(generation_kwargs, tuple):
            generation_kwargs = generation_kwargs[0]
    return args, generation_kwargs, rerank_args


if __name__ == "__main__":
    import vllm

    parser = init_parser()
    args, kwargs = parser.parse_known_args()
    generate = vllm_generator if args.vllm else hf_generator
    torch.cuda.empty_cache()
    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    gc.enable()

    args, generation_kwargs, rerank_args = parse_args(args)
    print(args, generation_kwargs, rerank_args)
    if not args.vllm:
        set_seed(42)

    output_file = args.output_file
    if "none" in output_file:
        output_file = output_file.replace("none", "unranked")
    output_file = output_file.replace(args.tgt_lang, args.tgt_lang[:2])
    output_file = output_file.replace(
        args.data_path, args.data_path.split("/")[-1].lower()
    )
    output_file = output_file.split('/')[-1]

    output_dir = (
        "/".join(args.output_file.split("/")[:2])
        + "/"
        + args.model.split("/")[-1].lower()
        + "/"
        + args.output_file.split("/")[2]
    )

    format_message = "translategemma" if "translategemma" in args.model else "messages"
    dataset_loader = partial(
        load_dataset,
        args.data_path,
        args.tgt_lang,
        format_message=format_message,
        convert_chat_template=args.vllm,
    )

    flush()
    preds = generate(dataset_loader, args, generation_kwargs, rerank_args)

    flush()

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{output_file}", "w", encoding="utf-8") as file:
        json.dump(preds, file, indent=2, ensure_ascii=False)
