import os
import pandas as pd
from rouge_score import rouge_scorer
import sacrebleu

class PostSavingEvaluation:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def evaluate(self, df: pd.DataFrame, output_path: str):
        rouge_l_scores, sacrebleu_scores = [], []

        # Assuming df has columns: "reference" and "prediction"
        for _, row in df.iterrows():
            reference = str(row["answer"])
            prediction = str(row["prediction"])

            # ROUGE-L
            rouge_score = self.rouge_scorer.score(reference, prediction)
            rouge_l_scores.append(rouge_score["rougeL"].fmeasure)

            # SacreBLEU (single reference per sample)
            bleu_score = sacrebleu.sentence_bleu(prediction, [reference])
            sacrebleu_scores.append(bleu_score.score)

        # Append new columns
        df["rougeL"] = rouge_l_scores
        df["sacreBLEU"] = sacrebleu_scores

        # Save back
        df.to_csv(output_path, index=False)
        print(f"Evaluation saved to {output_path}")