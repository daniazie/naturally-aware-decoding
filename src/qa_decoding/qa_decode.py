from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.generation.utils import (
    GenerateDecoderOnlyOutput,
    GenerateBeamDecoderOnlyOutput,
)
from datasets import Dataset
from typing import List, Dict, Literal
from functools import partial

from vllm import LLM, SamplingParams
from tqdm import tqdm
import torch
import gc

from rerankers import (
    QAReranker,
    LikelihoodReranker,
    CometReranker,
    SelfReranker,
)


def flush():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    gc.collect()
    
def compute_logits(
    input_ids: torch.Tensor,
    outputs: GenerateDecoderOnlyOutput | GenerateBeamDecoderOnlyOutput,
    best_of,
):
    sequences = outputs.sequences.cpu()
    logits_tup = outputs.logits
    logits = []
    for seq in range(sequences.shape[0]):
        _logits = torch.stack([token[seq] for token in logits_tup]).cpu()
        logits.append(_logits)
    logits = torch.stack([torch.stack(logits[i * best_of: (i+1) * best_of]) for i in range(input_ids.shape[0])])
    flush()
    return logits


def load_reranker(reranker_type, granularity, per_segment_eval, device_map):
    if reranker_type == "ratios":
        return QAReranker(
            model_path="t_index_reproduce/models/sft/qwen3-0.6b-multilingual",
            granularity=granularity,
            device_map=device_map,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_4",
            comet_model="Unbabel/XCOMET-XL",
        )
    elif reranker_type == "likelihood":
        return LikelihoodReranker(
            model="Qwen/Qwen3-0.6B",
            per_segment_eval=per_segment_eval,
            device_map=device_map,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_4",
        )
    elif reranker_type == "comet":
        return CometReranker("Unbabel/XCOMET-XL")
    return


def _generate(texts, model, sampling_params):
    preds = []
    for i, batch in enumerate(tqdm(texts, total=len(texts), desc="Generating...")):
        outputs = model.generate(
            batch["prompt"], sampling_params=sampling_params, use_tqdm=False
        )
        mts = [output.outputs[0].text for output in outputs]
        for src, mt, ref in zip(batch["src"], mts, batch["ref"]):
            preds.append({"src": src, "ref": ref, "mt": mt})
    return preds


def hf_pipeline(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: List[dict] | Dict[str, List[str]] | Dataset,
    batch_size: int = 4,
    device_map: str = "auto",
    best_of: int = 8,
    reranker_type: Literal["ratios", "comet", "combined", "likelihood", "self"]
    | None = None,
    granularity: Literal["token", "segment", "sequence"] | None = None,
    reranker_args: dict | None = None,
    generation_kwargs: dict | None = None,
    per_segment_eval: bool = False,
):
    if reranker_type == "self":
        if hasattr(model, "vocab_size"):
            vocab_size = model.vocab_size
        else:
            vocab_size = model.config.get_text_config().vocab_size
        reranker = SelfReranker(
            tokenizer, best_of=best_of, vocab_size=vocab_size, per_segment_eval=per_segment_eval
        )
    else:
        reranker = load_reranker(
            reranker_type,
            granularity=granularity,
            per_segment_eval=per_segment_eval,
            device_map=device_map,
        )

    if isinstance(texts, list):
        texts = Dataset.from_list(texts)
    elif isinstance(texts, dict):
        texts = Dataset.from_dict(texts)

    texts = texts.batch(batch_size)

    generation_kwargs = {
        k: v
        for k, v in generation_kwargs.items()
        if v is not None
    }

    generation_kwargs["num_return_sequences"] = best_of
    compute_per_batch_logits = partial(
        compute_logits, best_of=best_of
    )

    preds = []

    for i, batch in enumerate(tqdm(texts, desc="Generating...")):
        model_inputs = tokenizer.apply_chat_template(
            batch["messages"],
            add_generation_prompt=True,
            tokenize=True,
            max_length=1024,
            padding="max_length",
            return_tensors="pt",
        ).to(model.device)

        input_ids = model_inputs["input_ids"].clone().cpu()
        input_len = model_inputs["input_ids"].shape[1]

        outputs = model.generate(**model_inputs, **generation_kwargs)
        flush()
        if reranker_type == "self":
            logits = compute_per_batch_logits(input_ids, outputs)
            sequences = outputs.sequences.cpu()
            best = reranker.rerank(sequences, logits, input_len, best_of, refs=batch['ref'] **reranker_args)
        else:
            sequences = outputs.cpu()
            outputs = torch.stack(
                [
                    torch.stack([seq for seq in sequences[i : i + best_of]])
                    for i in range(0, len(sequences), best_of)
                ]
            )
            mts = [
                tokenizer.batch_decode(
                    [seq[input_len:] for seq in sequences[i]],
                    skip_special_tokens=True,
                )
                for i in range(len(sequences))
            ]
            best = reranker.rerank(batch["src"], mts, refs=batch['src'], **reranker_args)
        preds += best
    return preds


def vllm_pipeline(
    model: LLM,
    texts: List[dict] | Dict[str, List[str]] | Dataset,
    batch_size: int = 4,
    device_map: str = "auto",
    sampling_params: SamplingParams | None = None,
    reranker_type: Literal["ratios", "likelihood", "comet", "combined"] | None = None,
    granularity: Literal["token", "segment", "sequence"] | None = None,
    per_segment_eval: bool = False,
    reranker_args: dict | None = None,
):
    reranker = load_reranker(
        reranker_type,
        granularity=granularity,
        per_segment_eval=per_segment_eval,
        device_map=device_map,
    )

    if isinstance(texts, list):
        texts = Dataset.from_list(texts)
    elif isinstance(texts, dict):
        texts = Dataset.from_dict(texts)

    texts = texts.batch(batch_size)

    preds = []

    if reranker_type is None:
        return _generate(texts, model=model, sampling_params=sampling_params)

    for i, batch in enumerate(tqdm(texts, total=len(texts), desc="Generating...")):
        outputs = model.generate(
            batch["prompt"], sampling_params=sampling_params, use_tqdm=False
        )
        mts = [[seq.text for seq in output.outputs] for output in outputs]
        best = reranker.rerank(batch["src"], mts, refs=batch['ref'], **reranker_args)
        for res, ref in zip(best, batch["ref"]):
            res.update(
                {
                    "ref": ref,
                }
            )
        flush()
        preds += best
    return preds
