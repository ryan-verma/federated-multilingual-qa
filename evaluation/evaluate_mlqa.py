import json
import os
import numpy as np
import evaluate

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    XLMRobertaForQuestionAnswering,
    Trainer,
)

from evaluation.preprocess_mlqa import (
    preprocess_validation_examples,
)

from evaluation.metrics import (
    compute_metrics,
)

from evaluation.cjk_metrics import (
    compute_cjk_metrics,
)

MODEL_PATH = "outputs/fl_dp/raw_xlmr/global/final_model"

EVAL_CONFIGS = {
    "en": "mlqa.en.en",
    "hi": "mlqa.hi.hi",
    "es": "mlqa.es.es",
    "de": "mlqa.de.de",
    "zh": "mlqa.zh.zh",
}

# Languages where the standard whitespace-tokenized SQuAD F1 metric
# understates performance, since they aren't whitespace-delimited.
# Character-level F1 is used for these instead -- see
# evaluation/cjk_metrics.py for why.
CHAR_LEVEL_LANGUAGES = {"zh"}


def evaluate_language(lang, config):

    print(f"\nEvaluating {lang}")
    print("-" * 30)

    raw_dataset = load_dataset(
        "facebook/mlqa",
        config,
    )

    validation_examples = raw_dataset["validation"]

    validation_features = validation_examples.map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=validation_examples.column_names,
    )

    model = (
        XLMRobertaForQuestionAnswering
        .from_pretrained(MODEL_PATH)
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    trainer = Trainer(
        model=model,
    )

    predictions = trainer.predict(
        validation_features
    )

    start_logits, end_logits = (
        predictions.predictions
    )

    predicted_answers = compute_metrics(
        start_logits,
        end_logits,
        validation_features,
        validation_examples,
    )

    formatted_predictions = []

    references = []

    for example in validation_examples:

        example_id = example["id"]

        formatted_predictions.append(
            {
                "id": example_id,
                "prediction_text":
                predicted_answers[example_id],
            }
        )

        references.append(
            {
                "id": example_id,
                "answers": example["answers"],
            }
        )

    if lang in CHAR_LEVEL_LANGUAGES:
        # Whitespace-split F1 understates Chinese performance -- use
        # character-level comparison instead (see evaluation/cjk_metrics.py).
        results = compute_cjk_metrics(
            formatted_predictions,
            references,
        )
        print(f"(using character-level F1 for {lang})")
    else:
        squad_metric = evaluate.load("squad")
        results = squad_metric.compute(
            predictions=formatted_predictions,
            references=references,
        )

    print(
        f"EM: {results['exact_match']:.2f}"
    )

    print(
        f"F1: {results['f1']:.2f}"
    )

    return results


def main():

    os.makedirs(
        "results/dp",
        exist_ok=True,
    )

    all_results = {}

    for lang, config in EVAL_CONFIGS.items():

        results = evaluate_language(
            lang,
            config,
        )

        all_results[lang] = results

    with open(
        "results/dp/metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            all_results,
            f,
            indent=4,
        )

    print("\nEvaluation complete.")
    print(
        "Results saved to "
        "results/centralized/"
    )


if __name__ == "__main__":
    main()