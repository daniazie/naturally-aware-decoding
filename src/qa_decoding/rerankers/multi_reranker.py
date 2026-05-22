from typing import List, Literal
from tqdm import tqdm
from functools import partial

import torch
import numpy as np

from segmenter import Segmenter
from rerankers.qe_rerank import CometReranker
from rerankers.ratios import RatioReranker

class MultiReranker():
    def __init__(self, model_dir: str, hf_kwargs: dict | None, comet_model: str = "Unbabel/wmt23-cometkiwi-da-xl", comet_kwargs: dict | None = None, segmenter: Segmenter | None = None, granularity: Literal['token', 'sequence', 'segment'] = "token"):
        self.nat_reranker = RatioReranker(model_dir, granularity=granularity, **hf_kwargs)
        self.comet_reranker = CometReranker(comet_model)
    
    def get_best(self, src: str, mts: List[str], scores: np.ndarray, return_score: bool = False):
        best = scores.argmax().item()
        res = {
            "src": src,
            "mt": mts[best]
        }

        if return_score:
            res.update({
                "score": scores[best].item()
            })
        
        return res

    def rerank(self, srcs: List[str], mts: List[List[str]], tgt_lang: str, return_score: bool = False, w_nat: float = 1., w_comet: float = 1., return_nat: bool = False, return_comet: bool = False):
        results = []
        nat_results = []
        comet_results = []
        for i, src in enumerate(tqdm(srcs, desc="Reranking...")):
            batch = self.nat_reranker.prepare_data(src, mts[i], tgt_lang)
            nat_scores: torch.Tensor = self.nat_reranker._score(batch)
            nat_scores = nat_scores.sigmoid().to(dtype=torch.float32).numpy()
            comet_scores = np.array(self.comet_reranker.compute(src, mts[i]))

            scores = ((w_nat * nat_scores) + (w_comet * comet_scores)) / 2
            res = self.get_best(src, mts[i], scores, return_score=return_score)

            if return_nat:
                nat_res = self.get_best(src, mts[i], nat_scores, return_score=return_score)
                nat_results.append(nat_res)
            if return_comet:
                comet_res = self.get_best(src, mts[i], comet_scores, return_score=return_score)
                comet_results.append(comet_res)

            results.append(res)

        if not return_comet and not return_nat:
            return results
        elif return_comet and not return_nat:
            return results, comet_results
        elif return_nat and not return_comet:
            return results, nat_results
        else:
            return results, nat_results, comet_results
        
    def tune(self, srcs: List[str], mts: List[List[str]], tgt_lang: str, init_weights: list[float] | None = None, num_epochs: int = 1, learning_rate: float = 1e-4):   
        def score(
            w1, w2,
            src, mts, tgt_lang,
        ):
            nat_scores = self.nat_reranker._score(src, mts[i], tgt_lang).sigmoid().to(dtype=torch.float32).numpy()
            comet_scores = np.array(self.comet_reranker.compute(src, mts[i]))

            scores = ((nat_scores * w1) + (comet_scores * w2)) / 2
            return scores
        
        def calc_gradient(w1, w2, score_fn, eps=1e-5):
            grad_w1: np.ndarray = (score_fn(w1 + eps, w2) - score_fn(w1, w2)) / eps
            grad_w2: np.ndarray = (score_fn(w1, w2 + eps) - score_fn(w1, w2)) / eps
            return grad_w1.mean(), grad_w2.mean()

        if init_weights is None:
            w1, w2 = 0., 0.
        else:
            w1, w2 = init_weights

        w1_new, w2_new = 0., 0.
        for _ in range(num_epochs):
            for i, src in enumerate(tqdm(srcs, desc="Tuning...")):
                score_fn = partial(score, src=src, mts=mts[i], tgt_lang=tgt_lang)
                if i == 0:
                    _score = score_fn(w1, w2)
                g1, g2 = calc_gradient(w1, w2, score_fn)
                w1 += learning_rate * g1
                w2 += learning_rate * g2
                _new = score_fn(w1, w2)
                if _new.mean() > _score.mean():
                    w1_new = w1
                    w2_new = w2
                    _score = _new
                else:
                    w1 = w1_new
                    w2 = w2_new
                
        
        print("w_nat:", w1)
        print("w_comet:", w2)

        return w1_new, w2_new