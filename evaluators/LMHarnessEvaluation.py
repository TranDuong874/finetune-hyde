from lm_eval import evaluator, tasks
from lm_eval.models.huggingface import HFLM
import os, json
import torch
import numpy as np

import copy

class LMHarnessEvaluation:
    def __init__(self, cfg, model=None, tokenizer=None):
        """
        cfg: dict like your example
        model: optional in-memory HuggingFace model
        tokenizer: required if model is in-memory
        """
        self.cfg = cfg
        self.model_obj = model
        self.tokenizer = tokenizer

        self.model_name_or_obj = cfg.get("model_name_or_obj")
        self.device = cfg.get("device", "cuda")
        self.output_dir = cfg.get("output_dir", "results")
        self.task_names = cfg.get("task_names", ["mmlu", "hellaswag", "arc_challenge", "truthfulqa_mc2"])
        self.num_fewshot = cfg.get("num_fewshot", 5)
        self.batch_size = cfg.get("batch_size", "auto")
        self.limit = cfg.get("limit", None)
        self.apply_chat_template = cfg.get("apply_chat_template", True)
        self.fewshot_as_multiturn = cfg.get("fewshot_as_multiturn", True)
        self.system_instruction = cfg.get("system_instruction", "")

        os.makedirs(self.output_dir, exist_ok=True)
        self.task_manager = tasks.get_task_dict(self.task_names)
        
    def eval(self, output_filename="lm_harness_result"):
        """Run evaluation using built-in output saving"""
        
        # Set output path - use output_filename or default
        output_path = None
        if output_filename:
            output_path = os.path.join(self.output_dir, output_filename.replace('.json', ''))
        else:
            output_path = self.output_dir
        
        if self.model_obj is not None:
            lm = HFLM(
                pretrained=self.model_obj,
                tokenizer=self.tokenizer,
                device=self.device
            )
            results = evaluator.simple_evaluate(
                model=lm,
                tasks=self.task_names,
                output_path=output_path,  # Built-in saving
                log_samples=True,         # Optional: save individual samples
                batch_size=self.batch_size,
                apply_chat_template=self.apply_chat_template,
                fewshot_as_multiturn=self.fewshot_as_multiturn,
                num_fewshot=self.num_fewshot,
                system_instruction=self.system_instruction,
                limit=self.limit
            )
        else:
            results = evaluator.simple_evaluate(
                model="hf",
                model_args=f"pretrained={self.model_name_or_obj},trust_remote_code=True",
                tasks=self.task_names,
                output_path=output_path,
                log_samples=True,
                batch_size=self.batch_size,
                apply_chat_template=self.apply_chat_template,
                fewshot_as_multiturn=self.fewshot_as_multiturn,
                num_fewshot=self.num_fewshot,
                system_instruction=self.system_instruction,
                limit=self.limit
            )
        
        print(f"Results automatically saved to {output_path}")
        return results