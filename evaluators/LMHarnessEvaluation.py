from lm_eval import evaluator, tasks
from lm_eval.models.huggingface import HFLM
import os, json
import torch
import numpy as np
import copy

class LMHarnessEvaluation:
    def __init__(self, model, tokenizer, harness_eval_config):
        try:
            self.model = model
            self.tokenizer = tokenizer
            self.harness_eval_config = harness_eval_config
        except Exception as e:
            print(e)


    def eval(self):
        self.model.to("cuda")
        language_model = HFLM(
            pretrained=self.model,
            tokenizer=self.tokenizer,
            device="cuda"
        )
        
        print(language_model.device)

        eval_results = evaluator.simple_evaluate(
            model=language_model,
            **self.harness_eval_config
        )

        return str(eval_results)
