from typing import List
from comet import load_from_checkpoint, download_model

from tqdm.asyncio import tqdm_asyncio
import asyncio
import torch
import numpy as np
from rerankers.base_reranker import BaseReranker

class CometReranker(BaseReranker):
    def __init__(
        self,
        model_path: str = "Unbabel/wmt22-cometkiwi-da",
    ):
        model_path = download_model(model_path)
        torch.set_float32_matmul_precision("medium")
        self.model = load_from_checkpoint(model_path)
        self.model = torch.compile(self.model, mode="max-autotune")
        super().__init__(
            self.model
        )

    def _convert_sample(self, src: str, mts: List[str]):
        sample = []
        for mt in mts:
            sample.append({
                "src": src,
                "mt": mt
            })
        return sample
    
    def compute(self, src: str, mts: List[str]) -> List[float]:
        batch = self._convert_sample(src, mts)
        preds = self.model.predict(batch, progress_bar=False, num_workers=4)
        if hasattr(preds, "metadata"):
            scores = preds.metadata.mqm_scores
        else:
            scores = preds.scores
        return scores
    
    async def _rerank(self, src: str, mts: List[str], return_score: bool = False):
        scores = self.compute(src, mts)
        best = np.argmax(scores)
        res = {
            "src": src,
            "mt": mts[best]
        }

        if return_score:
            score = scores[best]
            res.update({"score": score})
        return res
    
    async def _gather(self, srcs, mts, return_score):
        results = [self._rerank(src, mts[i], return_score) for i, src in enumerate(srcs)]
        results = await tqdm_asyncio.gather(results, desc="Scoring...")
        return results

    def rerank(self, srcs: List[str], mts: List[List[str]], return_score: bool = False):
        results = asyncio.run(self._gather(srcs, mts, return_score))
        return results