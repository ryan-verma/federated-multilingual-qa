import os
import re

from flwr.app import Context
from flwr.server import (
    ServerAppComponents,
    ServerConfig,
)

from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import FedAvg
from flwr.serverapp import ServerApp

from federated import model_utils

"""
Flower server configuration for the federated learning setup.

TARGET_TOTAL_ROUNDS=3 chosen to roughly match the centralized baseline's
total training exposure per language (3 rounds x 1
local epoch ~= 3 epochs over each language's data, comparable to the
centralized script's 3 epochs over pooled data).

RESUME SUPPORT (opt-in via FL_RESUME=1):
If set, the server looks for the latest outputs/fl/<init>/global/
checkpoint-round-N folder, loads those weights as the starting point
(instead of letting Flower default-initialize from a fresh client), and
only runs the REMAINING rounds needed to reach TARGET_TOTAL_ROUNDS.
Round numbers in saved filenames continue counting up from where the
previous run left off, rather than restarting at 1.

Example:
  python -m training.train_federated            # rounds 1-3, fresh start
  Ctrl+C after round 2 finishes saving
  $env:FL_RESUME=1
  python -m training.train_federated            # resumes, runs round 3 only
"""

TARGET_TOTAL_ROUNDS = 3
OUTPUT_ROOT = f"outputs/fl/{model_utils.FL_INIT_FROM}/global"

RESUME = os.environ.get("FL_RESUME") == "1"


def find_latest_completed_round(output_root):
    """Scans for checkpoint-round-N folders and returns the highest N found,
    or 0 if none exist (i.e. nothing to resume from)."""
    if not os.path.isdir(output_root):
        return 0

    round_numbers = []
    for name in os.listdir(output_root):
        match = re.match(r"checkpoint-round-(\d+)$", name)
        if match:
            round_numbers.append(int(match.group(1)))

    return max(round_numbers) if round_numbers else 0


COMPLETED_ROUNDS = find_latest_completed_round(OUTPUT_ROOT) if RESUME else 0
ROUNDS_TO_RUN = TARGET_TOTAL_ROUNDS - COMPLETED_ROUNDS

if RESUME and COMPLETED_ROUNDS > 0:
    print(
        f"[server] Resuming: found checkpoint-round-{COMPLETED_ROUNDS}, "
        f"running {ROUNDS_TO_RUN} more round(s) to reach "
        f"{TARGET_TOTAL_ROUNDS} total."
    )
elif RESUME:
    print("[server] FL_RESUME=1 set, but no existing checkpoint found -- starting fresh.")


def load_initial_parameters():
    """Loads weights from the last completed round's checkpoint, in the
    same flat ndarray format Flower's Parameters object expects (matches
    the format model_utils.get_parameters() produces)."""
    checkpoint_dir = f"{OUTPUT_ROOT}/checkpoint-round-{COMPLETED_ROUNDS}"

    from transformers import XLMRobertaForQuestionAnswering
    model = XLMRobertaForQuestionAnswering.from_pretrained(checkpoint_dir)
    ndarrays = model_utils.get_parameters(model)

    return ndarrays_to_parameters(ndarrays)


class SaveModelFedAvg(FedAvg):
    """Identical aggregation behavior to standard FedAvg -- this only adds
    a side effect of writing the aggregated global model to disk after each
    round. Round numbers are offset by COMPLETED_ROUNDS so resumed runs
    continue the same numbering instead of restarting at 1.
    """

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            absolute_round = COMPLETED_ROUNDS + server_round
            self._save_checkpoint(aggregated_parameters, absolute_round)

        return aggregated_parameters, aggregated_metrics

    def _save_checkpoint(self, parameters, absolute_round):
        ndarrays = parameters_to_ndarrays(parameters)

        model = model_utils.get_model()
        model_utils.set_parameters(model, ndarrays)

        is_final_round = absolute_round == TARGET_TOTAL_ROUNDS
        save_dir = (
            f"{OUTPUT_ROOT}/final_model"
            if is_final_round
            else f"{OUTPUT_ROOT}/checkpoint-round-{absolute_round}"
        )

        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)

        from transformers import AutoTokenizer

        # Load tokenizer locally if any checkpoint exists, to avoid a network call.
        # Falls back to model name only on the very first checkpoint save (round 1)
        # when no prior checkpoint folder exists yet.
        existing = [
            f"{OUTPUT_ROOT}/checkpoint-round-{i}"
            for i in range(1, absolute_round)
            if os.path.isdir(f"{OUTPUT_ROOT}/checkpoint-round-{i}")
        ]
        tokenizer_source = existing[0] if existing else model_utils.MODEL_NAME
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            local_files_only=bool(existing),
        )

        tokenizer.save_pretrained(save_dir)

        print(f"[server] Saved global model checkpoint to {save_dir}")


def server_fn(context: Context):

    strategy_kwargs = {}
    if RESUME and COMPLETED_ROUNDS > 0:
        strategy_kwargs["initial_parameters"] = load_initial_parameters()

    strategy = SaveModelFedAvg(**strategy_kwargs)

    config = ServerConfig(
        num_rounds=max(ROUNDS_TO_RUN, 0)
    )

    return ServerAppComponents(
        strategy=strategy,
        config=config,
    )


server_app = ServerApp(
    server_fn=server_fn
)