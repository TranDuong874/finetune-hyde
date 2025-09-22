from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import math

class PreSavingEvaluation:
    def __init__(self, model, tokenizer, device=None):
        """
        model: a loaded HuggingFace AutoModelForCausalLM
        tokenizer: corresponding tokenizer
        device: torch device, default to model's device
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)

    @torch.no_grad()
    def evaluate(self, dataset, batch_size=8, max_length=128):
        """
        dataset: HuggingFace Dataset
        batch_size: number of samples per batch
        max_length: max tokens to generate
        Returns a list of dicts: input, target, prediction, loss, perplexity
        """
        results = []
        self.model.eval()

        # Create batches
        for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
            batch = dataset[i:i+batch_size]
            def split_by_flag(text: str, flag: str = "<start_of_turn>model"):
                i = text.find(flag)
                if i == -1:
                    return text, ""
                return text[:i + len(flag)], text[i + len(flag):]

            batch_texts = []
            batch_targets = []
            for item in batch["texts"]:
                text, target = split_by_flag(item)
                batch_texts.append(text)
                batch_targets.append(target)

            # Tokenize batch
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)

            # Batched generation
            output_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=False
            )
            predictions = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

            # Compute per-sample loss & perplexity
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs.input_ids)
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs.input_ids[..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                per_token_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                per_sample_loss = per_token_loss.view(inputs.input_ids.size(0), -1).mean(dim=1)

            # Store results per sample
            for j in range(len(batch)):
                loss_val = per_sample_loss[j].item()
                results.append({
                    "question": batch_texts[j],
                    "answer": batch_targets[j],
                    "prediction": predictions[j],
                    "loss": loss_val,
                    "perplexity": math.exp(loss_val) if loss_val < 20 else float("inf")  # safe guard
                })

        return results
