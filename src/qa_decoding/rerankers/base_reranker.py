from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from typing import List
from functools import partial

from abc import ABCMeta, abstractmethod


from data_utils import prepare_data


class BaseReranker(metaclass=ABCMeta):
    def __init__(
        self,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **kwargs,
    ):
        if tokenizer:
            self.prepare_data = partial(prepare_data, tokenizer=tokenizer, **kwargs)

    @abstractmethod
    def rerank(self, prompts: List[str], completions: List[List[str]], *args, **kwargs):
        pass
