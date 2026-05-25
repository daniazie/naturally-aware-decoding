from typing import List
from comet import load_from_checkpoint, download_model

import torch
import numpy as np
from rerankers.base_reranker import BaseReranker

import logging
import warnings

warnings.filterwarnings("ignore", module="pytorch_lightning")
warnings.filterwarnings("ignore", module="lightning")
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logger = logging.getLogger("comet")
logger.setLevel(logging.ERROR)


class CometReranker(BaseReranker):
    def __init__(
        self,
        model_path: str = "Unbabel/wmt22-cometkiwi-da",
    ):
        model_path = download_model(model_path)
        torch.set_float32_matmul_precision("medium")
        self.model = load_from_checkpoint(model_path)
        self.model = torch.compile(self.model, mode="max-autotune")
        super().__init__(self.model)

    def _convert_sample(self, src: str, mts: List[str]):
        sample = []
        for mt in mts:
            sample.append({"src": src, "mt": mt})
        return sample

    def compute(self, batch: List[dict[str, str]]) -> List[float]:
        preds = self.model.predict(batch, progress_bar=False)
        if hasattr(preds, "metadata"):
            mqm_scores = np.array(preds.metadata.mqm_scores)
            src_scores = np.array(preds.metadata.src_scores)
            pred_scores = np.array(preds.scores)
            # error_spans = np.array([len([span for span in error_spans if span['confidence'] >= 0.6]) for error_spans in preds.metadata.error_spans])

            scores = mqm_scores + src_scores
            scores = (scores - scores.mean()) / scores.std()
            scores = scores.tolist()
        else:
            scores = preds.scores
        return scores

    def _rerank(
        self,
        src: str,
        mts: List[str],
        scores: List[float],
        return_score: bool = False,
    ):
        best = np.argmax(scores)
        res = {"src": src, "mt": mts[best]}

        if return_score:
            score = scores[best]
            res.update({"score": score})
        return res

    def rerank(self, srcs: List[str], mts: List[List[str]], return_score: bool = False):
        batch = []
        n = len(mts[0])
        for i, src in enumerate(srcs):
            batch += self._convert_sample(src, mts[i])
        all_scores = self.compute(batch)

        scores = [all_scores[i * n : (i + 1) * n] for i in range(len(srcs))]
        results = [
            self._rerank(src, mts[i], scores[i], return_score)
            for i, src in enumerate(srcs)
        ]
        return results
