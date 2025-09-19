from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

class FullEvaluation:
    def __init__(self, model, tokenizer):
        """
        model: a loaded HuggingFace AutoModelForCausalLM
        tokenizer: corresponding tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def evaluate(self, dataset, max_length=128):
        """
        dataset: HuggingFace Dataset
        max_length: max tokens to generate
        Returns a list of dicts: input, target, prediction, and optionally loss
        """
        results = []

        self.model.eval()
        for sample in tqdm(dataset, desc="Evaluating"):
            # Prepare input
            text = " ".join([m["content"] for m in sample["messages"]])
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

            # Generate prediction
            output_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=False  # deterministic
            )
            prediction = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Compute loss if labels available
            labels = inputs.input_ids
            with torch.no_grad():
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss.item()

            results.append({
                "input": text,
                "target": " ".join([m["content"] for m in sample["messages"] if m["role"]=="assistant"]),
                "prediction": prediction,
                "loss": loss
            })

        return results
