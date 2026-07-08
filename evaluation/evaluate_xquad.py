"""
Zero-shot cross-dataset evaluation on XQuAD.

XQuAD (Artetxe et al. 2020) is 1,190 SQuAD v1.1 questions professionally
translated into 10 languages. It shares the same schema as MLQA
(id / context / question / answers) so the same preprocessing and metric
pipeline applies with no changes.

This is a ZERO-SHOT transfer evaluation -- models are evaluated on XQuAD
without any XQuAD fine-tuning. Results show whether MLQA training
generalizes to a different multilingual QA dataset.

Update MODEL_PATH and RESULTS_PATH to evaluate each trained variant.
"""

import json
import os

import evaluate

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    XLMRobertaForQuestionAnswering,
    Trainer,
)

from evaluation.preprocess_mlqa import preprocess_validation_examples
from evaluation.metrics import compute_metrics
from evaluation.cjk_metrics import compute_cjk_metrics

# ---------------------------------------------------------------------------
# Change these to evaluate a different model variant:
#   "outputs/centralized/final_model"
#   "outputs/fl/raw_xlmr/global/final_model"
#   "outputs/fl_weighted/raw_xlmr/global/final_model"
# ---------------------------------------------------------------------------
MODEL_PATH   = "outputs/fl_weighted/raw_xlmr/global/final_model"
RESULTS_PATH = "results/xquad/FL_weighted.json"

# XQuAD configs for the 5 languages matching your training setup.
# Note: XQuAD only has a validation split (no test split).
EVAL_CONFIGS = {
    "en": "xquad.en",
    "hi": "xquad.hi",
    "es": "xquad.es",
    "de": "xquad.de",
    "zh": "xquad.zh",
}

CHAR_LEVEL_LANGUAGES = {"zh"}


def evaluate_language(lang, config, model, tokenizer):

    print(f"\nEvaluating {lang}")
    print("-" * 30)

    raw_dataset = load_dataset(
        "google/xquad",
        config,
    )

    # XQuAD only has a validation split
    examples = raw_dataset["validation"]

    features = examples.map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=examples.column_names,
    )

    trainer = Trainer(model=model)

    predictions = trainer.predict(features)
    start_logits, end_logits = predictions.predictions

    predicted_answers = compute_metrics(
        start_logits,
        end_logits,
        features,
        examples,
    )

    formatted_predictions = [
        {
            "id": ex["id"],
            "prediction_text": predicted_answers[ex["id"]],
        }
        for ex in examples
    ]

    references = [
        {
            "id": ex["id"],
            "answers": ex["answers"],
        }
        for ex in examples
    ]

    if lang in CHAR_LEVEL_LANGUAGES:
        results = compute_cjk_metrics(formatted_predictions, references)
        print(f"(using character-level F1 for {lang})")
    else:
        squad_metric = evaluate.load("squad")
        results = squad_metric.compute(
            predictions=formatted_predictions,
            references=references,
        )

    print(f"EM: {results['exact_match']:.2f}")
    print(f"F1: {results['f1']:.2f}")

    return results


def main():

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    print(f"Loading model from {MODEL_PATH}...")
    model = XLMRobertaForQuestionAnswering.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    all_results = {}

    for lang, config in EVAL_CONFIGS.items():
        results = evaluate_language(lang, config, model, tokenizer)
        all_results[lang] = results

    # Summary
    avg_em = sum(r["exact_match"] for r in all_results.values()) / len(all_results)
    avg_f1 = sum(r["f1"] for r in all_results.values()) / len(all_results)
    gap_f1 = (
        max(r["f1"] for r in all_results.values()) -
        min(r["f1"] for r in all_results.values())
    )

    all_results["average"] = {"exact_match": avg_em, "f1": avg_f1}
    all_results["max_min_gap_f1"] = gap_f1

    print("\n" + "=" * 40)
    print(f"Average EM: {avg_em:.2f}  Average F1: {avg_f1:.2f}")
    print(f"Max-Min F1 gap: {gap_f1:.2f}")
    print("=" * 40)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()