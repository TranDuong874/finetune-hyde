from lm_eval import simple_evaluator, tasks
from lm_eval.models.huggingface import HFLM

class LMHarnessEvaluation:
    def __init__(self, model, tokenizer, harness_eval_config):
        try:
            self.model = model
            self.tokenizer = tokenizer
            self.harness_eval_config = harness_eval_config
        except Exception as e:
            print(e)


    def eval(self):
        language_model = HFLM(
            pretrained=self.model,
            tokenizer=self.tokenizer,
            device="cuda"
        )

        eval_results = simple_evaluator(
            model=language_model,
            **self.harness_eval_config['harness_eval_config']
        )

        return eval_results