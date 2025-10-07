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

        with open('harness_results.json', 'w', encoding='utf-8') as file:
            json.dump(eval_results['results'], file, indent=4, ensure_ascii=False)

        return json.dumps(eval_results['results'], indent=4, ensure_ascii=False)

