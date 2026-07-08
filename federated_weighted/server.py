import os
import re

from flwr.app import Context
from flwr.server import (
    ServerAppComponents,
    ServerConfig,
)

from flwr.common import (
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    NDArrays,
)
from flwr.server.strategy import FedAvg
from flwr.serverapp import ServerApp

import numpy as np

from federated import model_utils

"""
Language-weighted FedAvg server using q-FFL (q-Fair Federated Learning).

Plain FedAvg weights each client's update by num_examples, which lets
larger clients (e.g. English/SQuAD) dominate the global model every round
regardless of how well the model is already doing on those clients.

q-FFL corrects this by multiplying each client's weight by (loss^q), so
clients where the model is struggling (high loss) pull the global average
harder toward themselves. The result is a fairer model that doesn't just
optimize for already-easy languages.

Weight formula per client i:
    w_i = (loss_i ^ q) * num_examples_i
    then normalized: w_i / sum(w_j)

q controls aggressiveness:
    q=0  -> identical to plain FedAvg (loss ignored)
    q=1  -> mild correction
    q=2  -> moderate (default here, good starting point)
    q=5  -> strong correction, high-loss clients dominate heavily

One-round lag: aggregate_fit runs BEFORE aggregate_evaluate within each
round, so we use eval losses from the PREVIOUS round to weight the current
round's aggregation. Round 1 has no prior losses, so it falls back to
plain FedAvg weighting for that round only.

Resume support: same FL_RESUME=1 env var as federated/server.py.
"""

TARGET_TOTAL_ROUNDS = 3
Q = 2  # fairness exponent -- increase for stronger low-resource upweighting
OUTPUT_ROOT = f"outputs/fl_weighted/{model_utils.FL_INIT_FROM}/global"

RESUME = os.environ.get("FL_RESUME") == "1"


def find_latest_completed_round(output_root):
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
    print("[server] FL_RESUME=1 set but no existing checkpoint found -- starting fresh.")


def load_initial_parameters():
    checkpoint_dir = f"{OUTPUT_ROOT}/checkpoint-round-{COMPLETED_ROUNDS}"
    from transformers import XLMRobertaForQuestionAnswering
    model = XLMRobertaForQuestionAnswering.from_pretrained(checkpoint_dir)
    ndarrays = model_utils.get_parameters(model)
    return ndarrays_to_parameters(ndarrays)


class LanguageWeightedFedAvg(FedAvg):
    """
    q-FFL weighted aggregation. Stores per-client eval losses after each
    aggregate_evaluate call, then uses them as fairness multipliers in the
    next round's aggregate_fit. Everything else (server-side sampling,
    client selection, eval orchestration) is inherited from FedAvg unchanged.
    """

    def __init__(self, q=Q, **kwargs):
        super().__init__(**kwargs)
        self.q = q
        # Keyed by client id (cid string), value is the most recent eval loss
        # for that client. Populated by aggregate_evaluate, consumed by
        # aggregate_fit the following round.
        self._client_losses: dict[str, float] = {}
        print(f"[server] LanguageWeightedFedAvg initialized with q={self.q}")

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Build per-client weights using q-FFL formula.
        # results is a list of (ClientProxy, FitRes) tuples.
        weights_and_params = []

        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            num_examples = fit_res.num_examples
            ndarrays = parameters_to_ndarrays(fit_res.parameters)

            if self._client_losses and cid in self._client_losses:
                # q-FFL weight: loss^q * num_examples
                loss = self._client_losses[cid]
                weight = (loss ** self.q) * num_examples
            else:
                # Round 1 or unseen client: fall back to plain FedAvg weight
                weight = float(num_examples)

            weights_and_params.append((weight, ndarrays))

        # Log what weights were actually used this round
        total_weight = sum(w for w, _ in weights_and_params)
        print(f"\n[server] Round {COMPLETED_ROUNDS + server_round} aggregation weights:")
        for (client_proxy, fit_res), (weight, _) in zip(results, weights_and_params):
            cid = client_proxy.cid
            loss = self._client_losses.get(cid, None)
            loss_str = f"{loss:.4f}" if loss is not None else "N/A (round 1)"
            print(
                f"  client {cid} | "
                f"examples: {fit_res.num_examples} | "
                f"prev loss: {loss_str} | "
                f"weight: {weight/total_weight:.4f}"
            )

        # Weighted average across all parameter arrays
        aggregated_ndarrays = self._weighted_average(weights_and_params)
        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)

        # Save checkpoint
        absolute_round = COMPLETED_ROUNDS + server_round
        self._save_checkpoint(aggregated_parameters, absolute_round)

        return aggregated_parameters, {}

    def aggregate_evaluate(self, server_round, results, failures):
        """Store each client's eval loss so aggregate_fit can use it next round."""
        if results:
            for client_proxy, evaluate_res in results:
                if evaluate_res.loss is not None:
                    self._client_losses[client_proxy.cid] = evaluate_res.loss

        # Still call super() so Flower's own loss aggregation/logging runs normally
        return super().aggregate_evaluate(server_round, results, failures)

    def _weighted_average(self, weights_and_params: list) -> NDArrays:
        """Compute a normalized weighted average of parameter arrays."""
        total_weight = sum(w for w, _ in weights_and_params)
        # Initialize accumulator with zeros matching the shape of the first client's params
        accumulated = [np.zeros_like(arr) for arr in weights_and_params[0][1]]

        for weight, ndarrays in weights_and_params:
            normalized = weight / total_weight
            for i, arr in enumerate(ndarrays):
                accumulated[i] += normalized * arr.astype(np.float64)

        return [arr.astype(np.float32) for arr in accumulated]

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

        print(f"[server] Saved weighted checkpoint to {save_dir}")


def server_fn(context: Context):
    strategy_kwargs = {}
    if RESUME and COMPLETED_ROUNDS > 0:
        strategy_kwargs["initial_parameters"] = load_initial_parameters()

    strategy = LanguageWeightedFedAvg(
        q=Q,
        **strategy_kwargs,
    )

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