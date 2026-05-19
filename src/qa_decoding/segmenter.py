from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel
import torch

from tqdm import tqdm
from utils import pad

from typing import List

code2name = {
    "zho": "Chinese",
    "fra": "French",
    "deu": "German",
    "msa": "Malay",
    "kor": "Korean",
}

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

        t_all = tensors.unsqueeze(-1).sum(dim=1) / c_masks.dum(dim=1)
        t_max = tensors.masked_fill(~maxima_masks, 0).unsqueeze(-1).sum(dim=1) / c_masks.sum(dim=1)
        return t_max, t_all, maxima_masks, minima_masks

    def get_bs(self, tensor: torch.Tensor):
        t_max, t_all, maximas, minimas = self.tau(tensor)
        b0_mask = tensor >= t_max
        b1_mask = tensor >= t_all
        b2_mask = maximas
        b3_mask = ~minimas
        return b0_mask, b1_mask, b2_mask, b3_mask

    def get_boundaries(self, H: torch.Tensor, I: torch.Tensor):
        H_bs = self.get_bs(H)
        I_bs = self.get_bs(I)

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
            if t:
                segment.append(tensor[i])
            if t == 0:
                if segment:
                    segments.append(segment)
                segment = [tensor[i]]
            if not i == len(mask) - 1:
                if mask[i+1] < 0:
                    segments.append(segment)
            else:
                if t > 0:
                    segments.append(segment)

        return segments

    @torch.no_grad()
    def model_forward(self, forward_batch, backward_batch):
        _exclude_keys = {"completion_mask"}
        forward_inputs = {
            k: v.to(self.model.device)
            for k, v in forward_batch.items()
            if not k in _exclude_keys
        }

        backward_inputs = {
            k: v.fliplr().to(self.model.device)
            for k, v in backward_batch.items()
            if not k in _exclude_keys
        }

        forward_logits: torch.Tensor = self.model(**forward_inputs).logits[:, :-1].cpu()
        backward_logits: torch.Tensor = self.model(**backward_inputs).logits[:, 1:].cpu()

        forward_labels: torch.Tensor = forward_inputs['input_ids'][:, 1:].cpu()
        backward_labels = backward_inputs['input_ids'][:, :-1].cpu()

        forward_mask: torch.Tensor = forward_batch['completion_mask'][:, 1:].bool().cpu()
        backward_mask: torch.Tensor = backward_batch['completion_mask'][:, :-1].bool().cpu()

        outputs = {
            "forward_logits": forward_logits,
            "forward_labels": forward_labels,
            "forward_completion_mask": forward_mask,
            "backward_logits": backward_logits,
            "backward_labels": backward_labels,
            "backward_completion_mask": backward_mask
        }

        return outputs
    
    def flip(self, tensors: torch.Tensor):
        flipped = []
        for tensor in tensors:
            flipped.append(torch.tensor(tensor.tolist()[::-1]))
        return torch.stack(flipped)

    def compute_segment(self, forward_logits: torch.Tensor, backward_logits: torch.Tensor, forward_labels: torch.Tensor, backward_labels: torch.Tensor, forward_completion_mask: torch.Tensor, backward_completion_mask: torch.Tensor):
        H_forward = self.H(forward_logits).masked_fill(~forward_completion_mask, 0)
        I_forward = self.I(forward_logits, forward_labels).masked_fill(~forward_completion_mask, 0)
        H_backward = self.flip(self.H(backward_logits)).masked_fill(~backward_completion_mask, 0)
        I_backward = self.flip(self.I(backward_logits, backward_labels)).masked_fill(~backward_completion_mask, 0)

        end_new, start, forward_bp, forward_obg = self.get_boundaries(H_forward, I_forward)
        start_new, end, backward_bp, backward_obg = self.get_boundaries(H_backward, I_backward)

        masks = (((start_new | backward_obg) | (end | backward_bp)) | ((start | forward_bp) | (end_new | forward_obg))).int().masked_fill(~forward_completion_mask, -100)
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
        forward_input = {}
        backward_input = {}
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

        forward_input['input_ids'] = torch.tensor(prompt_completion_ids)
        forward_input['attention_mask'] = torch.tensor(attention_mask)
        forward_input['completion_mask'] = torch.tensor(completion_mask)

        backward_input['input_ids'] = torch.tensor(prompt_completion_ids[::-1])
        backward_input['attention_mask'] = torch.tensor(attention_mask[::-1])
        backward_input['completion_mask'] = torch.tensor(completion_mask[::-1])

        return forward_input, backward_input

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
    
    def get_batch(self, prompt, completion, lang):
        samples = [self.prepare_sample(sample) for sample in self.preprocess_batch(prompt, completion, lang)]
        forward_batch, backward_batch = [], []
        for sample in samples:
            forward_batch.append(sample[0])
            backward_batch.append(sample[1])
        forward_batch = self.collate_fn(forward_batch)
        backward_batch = self.collate_fn(backward_batch)
        return forward_batch, backward_batch

    def compute(self, prompts: List[str], completions: List[List[str]], lang: str):
        masks = []
        for prompt, completion in tqdm(zip(prompts, completions), total=len(prompts), desc="Segmenting..."):
            forward_batch, backward_batch = self.get_batch(prompt, completion, lang)
            outputs = self.model_forward(forward_batch, backward_batch)
            seg_masks = self.compute_segment(**outputs)
            masks.append(seg_masks)
        return masks
    
    def cuda(self):
        return self.model.cuda()
    
    def cpu(self):
        return self.model.cpu()