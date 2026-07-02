import flwr as fl

from federated.client import client_app
from federated_weighted.server import server_app

"""
Entry point for the language-weighted FedAvg experiment. Identical
structure to training/train_federated.py -- only the imported server_app
differs (weighted strategy instead of plain FedAvg).
"""

def main():

    fl.simulation.run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=5,
        backend_config={
            "client_resources": {
                "num_cpus": 2,
                "num_gpus": 1.0,
            }
        },
    )


if __name__ == "__main__":
    main()