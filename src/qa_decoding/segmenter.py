from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel
import torch

from typing import List
from tqdm import tqdm
from functools import partial

from data_utils import prepare_data

class Segmenter:
    def __init__(self, model: str | PreTrainedModel, tokenizer: PreTrainedTokenizerBase | None = None, **model_kwargs):
        if isinstance(model, str):
            self.model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model)
        else:
            self.model = model
            tokenizer = tokenizer
        self.prepare_data = partial(prepare_data, tokenizer=tokenizer)

    def H(self, logits: torch.Tensor):
        probs = logits.softmax(-1)
        logps = logits.log_softmax(-1)
        return - (probs * logps).sum(-1)

    def I(self, logits: torch.Tensor, labels: torch.Tensor):
        return - logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    def compute_mask(self, tensors: torch.Tensor, c_masks: torch.Tensor):
        minima_masks = []
        for tensor in tensors:
            minima_mask = []
            for i in range(len(tensor)):
                if i + 1 == len(tensor):
                    minima_mask.append(tensor[i] < tensor[i-1])
                elif i == 0:
                    minima_mask.append(tensor[i] < tensor[i+1])
                else:
                    minima_mask.append(tensor[i] < tensor[i-1] and tensor[i] < tensor[i+1])
            minima_masks.append(torch.stack(minima_mask))
        minima_masks = torch.stack(minima_masks)

        t_min_seq = (tensors.masked_fill(~minima_masks, 0).masked_fill(~c_masks, 0).sum(dim=-1) / c_masks.sum(dim=-1)).unsqueeze(-1)
        t_min_seg = (tensors.masked_fill(~minima_masks, 0).masked_fill(~c_masks, 0).sum(dim=-1) / minima_masks.int().masked_fill(~c_masks, 0).sum(dim=-1)).unsqueeze(-1)
        
        t_min = torch.amin(torch.cat((t_min_seq, t_min_seg)), keepdim=True)

        masks = minima_masks & (tensors < t_min)
        return masks
    
    def segment(self, tensors: torch.Tensor, masks: torch.Tensor):
        all_segments = []
        for tensor, mask in zip(tensors, masks):
            segments = []
            for i, m in enumerate(mask):
                if m < 0:
                    continue
                if (i == len(mask) - 1):
                    segment.append(tensor[i])
                    segments.append(segment)
                    continue
                is_end = (m == 1) or (mask[i+1] < 0)
                segment.append(tensor[i])
                if is_end:
                    segments.append(segment)
                    segment = []
                    continue
            all_segments.append(segments)
        return all_segments

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
        entropy = self.H(logits)
        entropy_scaled = entropy / entropy.amax(-1, keepdim=True)
        masks = self.compute_mask(entropy_scaled, completion_mask)
        return masks

    
    def compute(self, prompts: List[str] | None = None, completions: List[List[str]] | None = None, lang: str | None = None, batches: List[List[List[dict]]] | List[List[dict]] | None = None):
        masks = []
        if batches:
            batches = self.prepare_data(batches)
            outputs = self.model_forward(batch)
            seg_masks = [{"segment_mask": seg_mask} for seg_mask in self.compute_segment(**outputs)]
            return seg_masks
        
        batches = self.prepare_data(prompts, completions, lang)
        for i, batch in enumerate(tqdm(batches, total=len(prompts), desc="Segmenting...")):
            outputs = self.model_forward(batch)
            seg_masks = self.compute_segment(**outputs)
            masks.append(seg_masks)
        return masks
    
    def cuda(self):
        return self.model.cuda()
    
    def cpu(self):
        return self.model.cpu()