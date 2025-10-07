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
            
            # Base line:
            # https://huggingface.co/google/gemma-3-270m-it
            # Reasoning and factuality evaluataion

            self.task_list = [
                "bbh",
                "gpqa",
                "ifeval"
            ]

        except Exception as e:
            print(e)


    def eval(self):
        language_model = HFLM(
            pretrained=self.model,
            tokenizer=self.tokenizer,
            device="cuda"
        )

        print(language_model.device)

        eval_results = evaluator.simple_evaluate(
            model=language_model,
            tasks=self.task_list,
            **self.harness_eval_config
        )['results']

        
        selected_results = {
            'bbh' : eval_results['bbh']['exact_match,get-answer'],
            'gpqa' : eval_results['gpqa_diamond_zeroshot']['acc,none'],
            # 'ifeval' : eval_results['ifeval']
        }
        
        with open('harness_result_reference.json', 'w', encoding='utf-8') as file:
            json.dump(eval_results, file, indent=4, ensure_ascii=False)

        print(selected_results)
        return selected_results

