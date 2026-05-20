from transformers import PreTrainedModel, PreTrainedTokenizerBase, HfArgumentParser, BitsAndBytesConfig, set_seed
from datasets import Dataset
from typing import List, Dict, Literal

from vllm import LLM, SamplingParams
from tqdm import tqdm
import torch
import gc

from reranker import TranslationeseReranker, CometReranker, Reranker
from segmenter import Segmenter
torch.cuda.empty_cache()
gc.collect()
torch.cuda.reset_peak_memory_stats()

def hf_pipeline(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: List[dict] | Dict[str, List[str]] | Dataset,
    batch_size: int = 4,
    device_map: str = "auto",
    best_of: int = 8,
    reranker_type: Literal["natural", "comet", "combined"] | None = None,
    granularity: Literal['token', 'segment', 'sequence'] | None = None,
    reranker_args: dict | None = None,
    generation_kwargs: dict | None = None,
    enable_tqdm: bool = True
):
    if reranker_type == "natural":
        reranker = TranslationeseReranker(
            model_dir="t_index_reproduce/models/sft/qwen2.5-0.5b-mixture-5000-10",
            granularity=granularity,
            device_map=device_map
        )
    elif reranker_type == "comet":
        reranker = CometReranker()
    elif reranker_type == "combined":
        reranker = Reranker(
            nat_eval_model_dir="t_index_reproduce/models/sft/qwen2.5-0.5b-mixture-5000-10",
            hf_kwargs={"device_map": device_map},
        )

    if isinstance(texts, list):
        texts = Dataset.from_list(texts)
    elif isinstance(texts, dict):
        texts = Dataset.from_dict(texts)
    
    texts = texts.batch(batch_size)
    
    if "num_beams" in generation_kwargs.keys() and generation_kwargs.get("num_beams") is None:
        generation_kwargs.pop("num_beams")

    generation_kwargs["num_return_sequences"] = best_of

    preds = []
    if enable_tqdm:
        texts = tqdm(texts, desc="Generating...")

    for i, batch in enumerate(texts):
        model_inputs = tokenizer.apply_chat_template(
            batch['messages'],
            add_generation_prompt=True,
            tokenize=True,
            max_length=1024,
            padding='max_length',
            return_tensors='pt'
        ).to(model.device)

        input_len = [len(input_ids.flatten()) for input_ids in model_inputs['input_ids']]
        
        model_inputs = {
            k: v.to(model.device)
            for k, v in model_inputs.items()
        }

        outputs = model.generate(**model_inputs, **generation_kwargs)
        outputs = torch.stack([torch.stack([seq for seq in outputs[i:i+best_of]]) for i in range(0, len(outputs), best_of)])
        mts = [tokenizer.batch_decode([seq[input_len[i]:] for seq in outputs[i]], skip_special_tokens=True) for i in range(len(outputs))]

        best = reranker.rerank(batch['src'], mts, **reranker_args)
        if i % 4 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        preds += best
    return preds

def vllm_pipeline(
    model: LLM,
    texts: List[dict] | Dict[str, List[str]] | Dataset,
    batch_size: int = 4,
    device_map: str = "auto",
    sampling_params: SamplingParams | None = None,
    reranker_type: Literal["natural", "comet", "combined"] | None = None,
    granularity: Literal['token', 'segment', 'sequence'] | None = None,
    reranker_args: dict | None = None,
):
    if reranker_type == "natural":
        reranker = TranslationeseReranker(
            model_dir="t_index_reproduce/models/sft/qwen2.5-0.5b-mixture-5000-10",
            granularity=granularity,
            device_map=device_map
        )
    elif reranker_type == "comet":
        reranker = CometReranker()
    elif reranker_type == "combined":
        reranker = Reranker(
            model_dir="t_index_reproduce/models/sft/qwen2.5-0.5b-mixture-5000-10",
            hf_kwargs={"device_map": device_map},
            granularity=granularity
        )

    if isinstance(texts, list):
        texts = Dataset.from_list(texts)
    elif isinstance(texts, dict):
        texts = Dataset.from_dict(texts)
    
    texts = texts.batch(batch_size)

    preds = []


    if reranker_type is None:
        for i, batch in enumerate(tqdm(texts, total=len(texts), desc="Generating...")):
            outputs = model.generate(batch['prompt'], sampling_params=sampling_params, use_tqdm=False)
            mts = [output.outputs[0].text for output in outputs]
            preds += mts
        return preds

    for i, batch in enumerate(tqdm(texts, total=len(texts), desc="Generating...")):
        outputs = model.generate(batch['prompt'], sampling_params=sampling_params, use_tqdm=False)
        mts = [[seq.text for seq in output.outputs] for output in outputs]

        best = reranker.rerank(batch['src'], mts, **reranker_args)
        if i % 4 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        preds += best
    return preds

