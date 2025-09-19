from lm_eval import evaluator, tasks
from lm_eval.models.huggingface import HFLM
import os, json
import torch
import numpy as np

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

    def _to_json_serializable(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, torch.Tensor):
            # Convert to list of Python scalars
            return obj.detach().cpu().tolist()
        elif isinstance(obj, dict):
            return {k: self._to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_json_serializable(x) for x in obj]
        else:
            return obj

    
    def eval(self, output_filename=None):
        """Run evaluation on either model name or in-memory model"""
        # Wrap in-memory model if provided
        if self.model_obj is not None:
            lm = HFLM(
                pretrained=self.model_obj,
                tokenizer=self.tokenizer,
                device=self.device
            )
            results = evaluator.simple_evaluate(
                model=lm,
                tasks=self.task_names,
                batch_size=self.batch_size,
                apply_chat_template=self.apply_chat_template,
                fewshot_as_multiturn=self.fewshot_as_multiturn,
                num_fewshot=self.num_fewshot,
                system_instruction=self.system_instruction,
                limit=self.limit
            )
        else:
            # Load by model name
            results = evaluator.simple_evaluate(
                model="hf",
                model_args=f"pretrained={self.model_name_or_obj},trust_remote_code=True",
                tasks=self.task_names,
                batch_size=self.batch_size,
                apply_chat_template=self.apply_chat_template,
                fewshot_as_multiturn=self.fewshot_as_multiturn,
                num_fewshot=self.num_fewshot,
                system_instruction=self.system_instruction,
                limit=self.limit
            )

        print("="*50)
        print(results)
        print(type(results))
        print("="*50)
        serializable_results = self._to_json_serializable(results)

        # Save results
        if output_filename:
            path = os.path.join(self.output_dir, output_filename)
            with open(path, "w") as f:
                json.dump(serializable_results, f, indent=2)
            print(f"Results saved to {path}")
        return results
