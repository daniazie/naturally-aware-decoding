from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
import torch
from tqdm import tqdm

from typing import List

from utils import pad


class Segmenter:
    def __init__(self, model: str | PreTrainedModel, tokenizer: PreTrainedTokenizerBase | None = None, **model_kwargs):
        if isinstance(model, str):
            self.model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(model)
        else:
            self.model = model
            self.tokenizer = tokenizer

    def H(self, logits: torch.Tensor):
        probs = logits.softmax(-1)
        logps = logits.log_softmax(-1)
        return - (probs * logps).sum(-1)

    def I(self, logits: torch.Tensor, labels: torch.Tensor):
        return - logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    def tau(self, tensors: torch.Tensor, c_masks: torch.Tensor):
        maxima_masks = []
        minima_masks = []
        for tensor in tensors:
            maxima_mask = []
            minima_mask = []
            for i in range(len(tensor)):
                if i + 1 == len(tensor):
                    maxima_mask.append(tensor[i] > tensor[i-1])
                    minima_mask.append(tensor[i] < tensor[i-1])
                elif i == 0:
                    maxima_mask.append(tensor[i] > tensor[i+1])
                    minima_mask.append(tensor[i] < tensor[i+1])
                else:
                    maxima_mask.append(tensor[i] > tensor[i-1] and tensor[i] > tensor[i+1])
                    minima_mask.append(tensor[i] < tensor[i-1] and tensor[i] < tensor[i+1])
            maxima_masks.append(torch.stack(maxima_mask))
            minima_masks.append(torch.stack(minima_mask))
        maxima_masks = torch.stack(maxima_masks)
        minima_masks = torch.stack(minima_masks)

        t_all = tensors.masked_fill(~c_masks, 0).sum(dim=1) / c_masks.sum(dim=1)
        t_max = tensors.masked_fill(~maxima_masks, 0).sum(dim=1) / c_masks.sum(dim=1)
        t_all = t_all.unsqueeze(-1)
        t_max = t_max.unsqueeze(-1)
        return t_max, t_all, maxima_masks, minima_masks

    def get_bs(self, tensor: torch.Tensor, c_mask: torch.Tensor):
        t_max, t_all, maximas, minimas = self.tau(tensor, c_mask)
        b0_mask = tensor >= t_max
        b1_mask = tensor >= t_all
        b2_mask = maximas
        b3_mask = ~minimas
        return b0_mask, b1_mask, b2_mask, b3_mask

    def get_boundaries(self, H: torch.Tensor, I: torch.Tensor, c_mask: torch.Tensor):
        H_bs = self.get_bs(H, c_mask)
        I_bs = self.get_bs(I, c_mask)

        strong = (H_bs[0] & H_bs[2]) & (I_bs[1] | I_bs[3])
        weak = (H_bs[1] & H_bs[2]) & (I_bs[1] | I_bs[3])
        bp = (H_bs[1] & H_bs[2]) | (I_bs[1] & I_bs[2])
        obg = (H_bs[0] & H_bs[2]) & (I_bs[0] & I_bs[2])
        return strong, weak, bp, obg
    
    def segment(self, tensor: torch.Tensor, mask: torch.Tensor):
        segments = []
        segment = []
        for i, t in enumerate(mask):
            if t < 0:
                continue
            if t == 0:
                segment.append(tensor[i])
            if t == 1:
                if segment:
                    segments.append(segment)
                segment = [tensor[i]]
            if not i == len(mask) - 1:
                if mask[i+1] < 0:
                    segments.append(segment)
            else:
                if t >= 0:
                    segments.append(segment)

        return segments

    @torch.no_grad()
    def model_forward(self, batch):
        _exclude_keys = {"completion_mask"}
        forward_inputs = {
            k: v.to(self.model.device)
            for k, v in batch.items()
            if not k in _exclude_keys
        }

        logits: torch.Tensor = self.model(**forward_inputs).logits[:, :-1].cpu()
        labels: torch.Tensor = forward_inputs['input_ids'][:, 1:].cpu()
        completion_mask: torch.Tensor = batch['completion_mask'][:, 1:].bool().cpu()


        forward_inputs = {
            k: v.cpu()
            for k, v in batch.items()
            if not k in _exclude_keys
        }

        outputs = {
            "logits": logits,
            "labels": labels,
            "completion_mask": completion_mask,
        }

        return outputs

    def compute_segment(self, logits: torch.Tensor, labels: torch.Tensor, completion_mask: torch.Tensor):
        entropy = self.H(logits).masked_fill(~completion_mask, 0)
        surprisal = self.I(logits, labels).masked_fill(~completion_mask, 0)

        end, start, bp, obg = self.get_boundaries(entropy, surprisal, completion_mask)

        masks = (((start | bp) & (end & obg))).int().masked_fill(~completion_mask, -100) 
        return masks

    def format_messages(self, prompt: str, completion: str, lang: str,):
        return {
            "prompt": [
                {"role": "user", "content": f"Translate the following text into {lang}.\n\n{prompt}"}
            ],
            "completion": [
                {"role": "assistant", "content": completion}
            ]
        }
    
    def tokenize_fn(self, text, tokenizer: PreTrainedTokenizerBase, **kwargs):
        return tokenizer.apply_chat_template(
            text,
            tokenize=True,
            return_dict=True,
            **kwargs
        )

    def preprocess_batch(self, prompt: str, completions: List[str], lang: str):
        data = []
        for completion in completions:
            sample = self.format_messages(prompt, completion, lang)
            data.append(sample)
        return data
    
    def prepare_sample(self, sample):
        output = {}
        prompt_ids = self.tokenize_fn(
            sample['prompt'],
            tokenizer=self.tokenizer,
            add_generation_prompt=True,
        )['input_ids']

        prompt_completion_processed = self.tokenize_fn(
            sample['prompt'] + sample['completion'],
            tokenizer=self.tokenizer,
        )

        prompt_completion_ids = prompt_completion_processed['input_ids']
        attention_mask = prompt_completion_processed['attention_mask']
        completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))

        output['input_ids'] = torch.tensor(prompt_completion_ids)
        output['attention_mask'] = torch.tensor(attention_mask)
        output['completion_mask'] = torch.tensor(completion_mask)

        return output

    def collate_fn(self, examples):
        input_ids = [example['input_ids'] for example in examples]
        attention_mask = [example['attention_mask'] for example in examples]
        completion_mask = [example['completion_mask'] for example in examples]

        input_ids = pad(
            input_ids,
            padding_value=self.tokenizer.pad_token_id,
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
            "completion_mask": completion_mask
        }
    
    def get_batch(self, prompt: str | None = None, completions: List[str] | None = None, lang: str | None = None, samples: List[List[dict]] | List[dict] | None = None):
        if samples is None:
            samples = self.preprocess_batch(prompt, completions, lang)
        batch = self.collate_fn(samples)
        return batch

    def compute(self, prompts: List[str] | None = None, completions: List[List[str]] | None = None, lang: str | None = None, batch: List[List[dict]] | List[tuple[dict, dict]] | tuple[list, list] | None = None):
        masks = []
        if batch:
            batch = self.get_batch(samples=batch)
            outputs = self.model_forward(batch)
            seg_masks = [{"segment_mask": seg_mask} for seg_mask in self.compute_segment(**outputs)]
            return seg_masks
        for prompt, completion in tqdm(zip(prompts, completions), total=len(prompts), desc="Segmenting..."):
            batch = self.get_batch(prompt, completion, lang)
            outputs = self.model_forward(batch)
            seg_masks = self.compute_segment(**outputs)
            masks.append(seg_masks)
        return masks
    
    def cuda(self):
        return self.model.cuda()
    
    def cpu(self):
        return self.model.cpu()