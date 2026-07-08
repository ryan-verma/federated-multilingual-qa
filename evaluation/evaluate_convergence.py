# ============================================================
# evaluate_convergence.py
# ============================================================
import json
import os
import re

from datasets import load_dataset
from transformers import (
    XLMRobertaForQuestionAnswering,
    Trainer,
)

from evaluation.preprocess_eval_loss import (
    preprocess_with_labels,
)

METHOD_CHECKPOINT_DIRS = {
    "fedavg": "outputs/fl/raw_xlmr/global",
    "weighted_fedavg": "outputs/fl_weighted/raw_xlmr/global",
}

# Which language to track loss on. Pick one representative language
# (or loop over all EVAL_CONFIGS if you want per-language convergence too).
EVAL_CONFIG = "mlqa.en.en"

ROUND_DIR_PATTERN = re.compile(r"^checkpoint-round-(\d+)$")


def discover_rounds(checkpoint_root):
    """Finds checkpoint-round-N subdirectories under a checkpoint root
    and returns a sorted list of (round_number, path) tuples."""

    rounds = []

    for name in os.listdir(checkpoint_root):

        match = ROUND_DIR_PATTERN.match(name)

        if match:
            round_num = int(match.group(1))
            rounds.append(
                (round_num, os.path.join(checkpoint_root, name))
            )

    rounds.sort(key=lambda x: x[0])

    return rounds


def evaluate_checkpoint_loss(checkpoint_path, validation_features):

    model = (
        XLMRobertaForQuestionAnswering
        .from_pretrained(checkpoint_path)
    )

    trainer = Trainer(
        model=model,
    )

    predictions = trainer.predict(
        validation_features
    )

    eval_loss = predictions.metrics.get(
        "test_loss"
    )

    return eval_loss


def evaluate_method(method_name, checkpoint_root, validation_features):

    print(f"\n=== {method_name} ===")

    rounds = discover_rounds(checkpoint_root)

    if not rounds:
        print(
            f"No checkpoint-round-N dirs found under {checkpoint_root}"
        )
        return {}

    round_losses = {}

    for round_num, checkpoint_path in rounds:

        loss = evaluate_checkpoint_loss(
            checkpoint_path,
            validation_features,
        )

        round_losses[round_num] = loss

        print(f"Round {round_num}: loss = {loss:.4f}")

    # also evaluate final_model, labeled as one round past the last checkpoint
    final_model_path = os.path.join(checkpoint_root, "final_model")

    if os.path.isdir(final_model_path):

        loss = evaluate_checkpoint_loss(
            final_model_path,
            validation_features,
        )

        final_round_num = rounds[-1][0] + 1

        round_losses[final_round_num] = loss

        print(f"Round {final_round_num} (final_model): loss = {loss:.4f}")

    return round_losses


def main():

    os.makedirs(
        "results/convergence",
        exist_ok=True,
    )

    raw_dataset = load_dataset(
        "facebook/mlqa",
        EVAL_CONFIG,
    )

    validation_examples = raw_dataset["validation"]

    validation_features = validation_examples.map(
        preprocess_with_labels,
        batched=True,
        remove_columns=validation_examples.column_names,
    )

    all_results = {}

    for method_name, checkpoint_root in METHOD_CHECKPOINT_DIRS.items():

        round_losses = evaluate_method(
            method_name,
            checkpoint_root,
            validation_features,
        )

        all_results[method_name] = round_losses

    with open(
        "results/convergence/round_losses.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            all_results,
            f,
            indent=4,
        )

    print("\nConvergence evaluation complete.")
    print(
        "Results saved to "
        "results/convergence/round_losses.json"
    )


if __name__ == "__main__":
    main()