from lm_eval import evaluator, tasks
from lm_eval.models.huggingface import HFLM
import json, os

class LMHarnessEvaluation:
    def __init__(self,
                 model_name_or_obj="google/gemma-3-270m-it",
                 tokenizer=None,
                 task_names=None,
                 device="cuda",
                 output_dir="results"):
        """
        Wrapper for EleutherAI's lm-evaluation-harness.
        Supports either a model name string or an in-memory model+tokenizer.
        """
        self.model_name_or_obj = model_name_or_obj
        self.tokenizer = tokenizer
        self.task_names = task_names or ["mmlu", "hellaswag", "arc_challenge", "truthfulqa_mc2"]
        self.device = device
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)
        self.task_manager = tasks.get_task_dict(self.task_names)

    def eval(self, output_filename=None, num_fewshot=5, batch_size="auto"):
        """
        Run evaluation on either a HF model name or in-memory model.
        """
        # Case 1: model_name_or_obj is a string → load by name
        if isinstance(self.model_name_or_obj, str):
            results = evaluator.simple_evaluate(
                model="hf",
                model_args=f"pretrained={self.model_name_or_obj},trust_remote_code=True",
                tasks=self.task_names,
                batch_size=batch_size,
                device=self.device,
                apply_chat_template=True,
                fewshot_as_multiturn=True,
                num_fewshot=num_fewshot,
                system_instruction=""
            )
        else:
            # Case 2: model_name_or_obj is a HuggingFace model → wrap with HFLM
            lm = HFLM(
                pretrained=self.model_name_or_obj,
                tokenizer=self.tokenizer,
                device=self.device
            )
            results = evaluator.simple_evaluate(
                model=lm,
                tasks=self.task_names,
                batch_size=batch_size,
                apply_chat_template=True,
                fewshot_as_multiturn=True,
                num_fewshot=num_fewshot,
                system_instruction=""
            )

        # Save results
        if output_filename:
            path = os.path.join(self.output_dir, output_filename)
            with open(path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {path}")

        return results
