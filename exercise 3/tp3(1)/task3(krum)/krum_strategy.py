import flwr as fl
import numpy as np
from typing import List, Tuple, Optional, Dict
from flwr.common import Parameters, FitRes, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy


class KrumStrategy(Strategy):
    def __init__(self, f: int):
        self.f = f

    def initialize_parameters(self, client_manager):
        return None

    def configure_fit(self, server_round, parameters, client_manager):
        clients = client_manager.sample(num_clients=10)
        fit_ins = fl.common.FitIns(parameters, {})
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        print(f"🔍 Running Krum aggregation... f={self.f}")

        weights_results = [parameters_to_ndarrays(res.parameters) for _, res in results]
        flattened = [np.concatenate([w.flatten() for w in weight]) for weight in weights_results]

        n = len(flattened)
        scores = []
        for i in range(n):
            distances = [np.linalg.norm(flattened[i] - flattened[j]) ** 2 for j in range(n) if i != j]
            distances.sort()
            score = sum(distances[: n - self.f - 2])
            scores.append(score)

        krum_idx = int(np.argmin(scores))
        krum_weights = weights_results[krum_idx]

        print(f"✅ Selected client update #{krum_idx} as most representative.")
        aggregated_parameters = ndarrays_to_parameters(krum_weights)
        return aggregated_parameters, {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        # Select a subset of clients for evaluation
        clients = client_manager.sample(num_clients=2)
        eval_ins = fl.common.EvaluateIns(parameters, {})
        return [(client, eval_ins) for client in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}
        accuracies = [res.metrics["accuracy"] for _, res in results if "accuracy" in res.metrics]
        losses = [res.loss for _, res in results if res.loss is not None]

        if not accuracies:
            return None, {}

        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_loss = sum(losses) / len(losses) if losses else None

        return avg_loss, {"accuracy": avg_accuracy}

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None
