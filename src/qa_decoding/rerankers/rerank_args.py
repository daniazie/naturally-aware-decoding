from typing import Literal
from dataclasses import dataclass

@dataclass
class RatioArgs:
    tgt_lang: str | None = None
    return_score: bool = False
    normalise_scores: bool = False

@dataclass
class LikelihoodArgs:
    tgt_lang: str | None = None
    metric: Literal["entropy", "surprisal", "perplexity"] = "entropy"
    return_score: bool = False
    normalise_scores: bool = False

@dataclass
class CometArgs:
    return_score: bool = False

@dataclass
class RerankerArgs:
    tgt_lang: str | None = None
    w_nat: float = 1.0
    w_comet: float = 1.0
    return_score: bool = False
    return_nat: bool = False
    return_comet: bool = False