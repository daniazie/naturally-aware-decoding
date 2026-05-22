from transformers import PreTrainedModel, PreTrainedTokenizerBase
from datasets import Dataset
from typing import List, Dict, Literal

from vllm import LLM, SamplingParams
from tqdm import tqdm
import torch
import gc

from rerankers import RatioReranker, LikelihoodReranker, CometReranker, MultiReranker

torch.cuda.empty_cache()
gc.collect()
torch.cuda.reset_peak_memory_stats()

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def load_reranker(reranker_type, granularity, device_map):
    if reranker_type == "ratios":
        return RatioReranker(
            model_path="t_index_reproduce/models/sft/qwen2.5-0.5b-mixture-5000-10",
            granularity=granularity,
            device_map=device_map,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_4"
        )
    elif reranker_type == "likelihood":
        return LikelihoodReranker(
            model="Qwen/Qwen2.5-0.5B",
            device_map=device_map,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_4"
        )
    elif reranker_type == "comet":
        return CometReranker("Unbabel/XCOMET-XL")
    elif reranker_type == "combined":
        return MultiReranker(
            model_dir="t_index_reproduce/models/sft/qwen2.5-0.5b-mixture-5000-10",
            hf_kwargs={"dtype": torch.bfloat16, "device_map": device_map, "attn_implementation": "flash_attention_4"},
            comet_model="Unbabel/XCOMET-XL",
            granularity=granularity
        )
    return

def _generate(texts, model, sampling_params):
    preds = []
    for i, batch in enumerate(tqdm(texts, total=len(texts), desc="Generating...")):
        outputs = model.generate(batch['prompt'], sampling_params=sampling_params, use_tqdm=False)
        mts = [output.outputs[0].text for output in outputs]
        for src, mt, ref in zip(batch['src'], mts, batch['ref']):
            preds.append({
                "src": src,
                "ref": ref,
                "mt": mt
            })
    return preds

def hf_pipeline(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: List[dict] | Dict[str, List[str]] | Dataset,
    batch_size: int = 4,
    device_map: str = "auto",
    best_of: int = 8,
    reranker_type: Literal["ratios", "comet", "combined", "likelihood"] | None = None,
    granularity: Literal['token', 'segment', 'sequence'] | None = None,
    reranker_args: dict | None = None,
    generation_kwargs: dict | None = None,
    enable_tqdm: bool = True
):
    reranker = load_reranker(reranker_type, granularity=granularity, device_map=device_map)

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
    reranker_type: Literal["ratios", "likelihood", "comet", "combined"] | None = None,
    granularity: Literal['token', 'segment', 'sequence'] | None = None,
    reranker_args: dict | None = None,
):
    
    reranker = load_reranker(reranker_type, granularity=granularity, device_map=device_map)

    if isinstance(texts, list):
        texts = Dataset.from_list(texts)
    elif isinstance(texts, dict):
        texts = Dataset.from_dict(texts)
    
    texts = texts.batch(batch_size)

    preds = []
    
    if reranker_type is None:
        return _generate(texts, model=model, sampling_params=sampling_params)

    for i, batch in enumerate(tqdm(texts, total=len(texts), desc="Generating...")):
        outputs = model.generate(batch['prompt'], sampling_params=sampling_params, use_tqdm=False)
        mts = [[seq.text for seq in output.outputs] for output in outputs]

        best = reranker.rerank(batch['src'], mts, **reranker_args)
        if i % 4 == 0:
            flush()
        for res, ref in zip(best, batch['ref']):
            res.update({
                "ref": ref,
            })
        preds += best
    return preds

def tune_pipeline(
    model: LLM,
    texts: List[dict] | Dict[str, List[str]] | Dataset,
    tgt_lang: str,
    batch_size: int = 4,
    device_map: str = "auto",
    sampling_params: SamplingParams | None = None,
    granularity: Literal['token', 'segment', 'sequence'] | None = None,
):
    reranker = MultiReranker(
        model_dir="t_index_reproduce/models/sft/qwen2.5-0.5b-mixture-5000-10",
        hf_kwargs={"device_map": device_map},
        comet_model="Unbabel/XCOMET-XL",
        granularity=granularity
    )

    if isinstance(texts, list):
        texts = Dataset.from_list(texts)
    elif isinstance(texts, dict):
        texts = Dataset.from_dict(texts)
    
    texts = texts.batch(batch_size)


    W_nat, W_comet = [], []

    for i, batch in enumerate(tqdm(texts, total=len(texts), desc="Generating...")):
        outputs = model.generate(batch['prompt'], sampling_params=sampling_params, use_tqdm=False)
        mts = [[seq.text for seq in output.outputs] for output in outputs]

        w_nat, w_comet = reranker.tune(batch['src'], mts, tgt_lang, learning_rate=1e-4, num_epochs=2)
        W_nat.append(w_nat)
        W_comet.append(w_comet)
    
    return W_nat, W_comet