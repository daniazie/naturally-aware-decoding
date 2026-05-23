from transformers import PreTrainedModel, PreTrainedTokenizerBase, AutoModelForCausalLM, AutoTokenizer
from typing import List
from functools import partial

from abc import ABCMeta, abstractmethod

import numpy as np

from data_utils import prepare_data

class BaseReranker(metaclass=ABCMeta):
    def __init__(self, model: PreTrainedModel | None = None, tokenizer: PreTrainedTokenizerBase | None = None, *args, **kwargs):
        if tokenizer:
            self.prepare_data = partial(prepare_data, tokenizer=tokenizer)

    @abstractmethod
    def rerank(self, prompts: List[str], completions: List[List[str]], *args, **kwargs):
        pass