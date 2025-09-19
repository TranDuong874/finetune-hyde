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
        dataset: HuggingFace Dataset with 'messages' field
        batch_size: number of samples per batch
        max_length: max tokens to generate
        Returns a list of dicts: question, answer, prediction, loss, perplexity
        """
        results = []
        self.model.eval()

        # Process dataset in batches
        for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
            # Get batch slice
            batch_end = min(i + batch_size, len(dataset))
            batch_samples = [dataset[j] for j in range(i, batch_end)]
            
            # Extract questions, answers, and full conversations
            batch_questions = []
            batch_answers = []
            batch_full_texts = []
            
            for sample in batch_samples:
                messages = sample["messages"]
                
                # Extract question (user message)
                question = ""
                answer = ""
                for msg in messages:
                    if msg["role"] == "user":
                        question = msg["content"]
                    elif msg["role"] == "assistant":
                        answer = msg["content"]
                
                batch_questions.append(question)
                batch_answers.append(answer)
                
                # Create full conversation text
                full_text = ""
                for msg in messages:
                    full_text += msg["content"] + " "
                batch_full_texts.append(full_text.strip())

            # Tokenize inputs for generation (questions only)
            question_inputs = self.tokenizer(
                batch_questions, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=max_length//2  # Leave room for generation
            ).to(self.device)

            # Generate predictions
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=question_inputs.input_ids,
                    attention_mask=question_inputs.attention_mask,
                    max_length=max_length,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode only the generated part (exclude input)
                predictions = []
                for j, (input_ids, output_ids_sample) in enumerate(zip(question_inputs.input_ids, output_ids)):
                    # Find where generation starts
                    input_length = len(input_ids[input_ids != self.tokenizer.pad_token_id])
                    generated_ids = output_ids_sample[input_length:]
                    prediction = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    predictions.append(prediction)

            # Compute loss on full conversations
            full_text_inputs = self.tokenizer(
                batch_full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)

            # Calculate per-sample loss
            with torch.no_grad():
                outputs = self.model(**full_text_inputs, labels=full_text_inputs.input_ids)
                
                # Get per-token losses
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = full_text_inputs.input_ids[..., 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=self.tokenizer.pad_token_id)
                per_token_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                # Reshape and get mean loss per sample (excluding padding)
                per_token_loss = per_token_loss.view(shift_labels.size())
                
                # Create mask to ignore padding tokens
                loss_mask = (shift_labels != self.tokenizer.pad_token_id).float()
                
                # Calculate mean loss per sample
                per_sample_loss = (per_token_loss * loss_mask).sum(dim=1) / loss_mask.sum(dim=1)

            # Store results for each sample in the batch
            for j in range(len(batch_samples)):
                loss_val = per_sample_loss[j].item()
                
                # Handle potential NaN or infinite values
                if math.isnan(loss_val) or math.isinf(loss_val):
                    loss_val = float('inf')
                    perplexity_val = float('inf')
                else:
                    perplexity_val = math.exp(min(loss_val, 20))  # Cap to prevent overflow
                
                results.append({
                    "question": batch_questions[j],
                    "answer": batch_answers[j],
                    "prediction": predictions[j],
                    "loss": loss_val,
                    "perplexity": perplexity_val
                })

        return results