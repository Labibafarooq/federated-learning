from typing import List, Tuple, Optional, Dict
import numpy as np
from flwr.common import Parameters, FitIns, FitRes, EvaluateIns, EvaluateRes, Scalar
from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common.parameter import parameters_to_ndarrays, ndarrays_to_parameters

class FedMedianStrategy(Strategy):
    def __init__(self, alpha: str = "alpha10", initial_parameters: Optional[Parameters] = None):
        self.alpha = alpha
        self.initial_parameters = initial_parameters

        self.evaluate_metrics_aggregation_fn = lambda metrics: {
            "accuracy": float(np.mean([m.get("accuracy", 0.0) for m in metrics])),
            "loss": float(np.mean([m.get("loss", 0.0) for m in metrics])),
        }

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return self.initial_parameters

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        print(f"[FedMedianStrategy] Configuring fit for round {server_round}")
        num_clients = min(5, len(client_manager.clients))
        clients = client_manager.sample(num_clients)
        fit_ins = FitIns(parameters, {"round": server_round})
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        print(f"[FedMedianStrategy] Aggregating fit results for round {server_round}")
        if not results:
            return None, {}

        weights = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        median_weights = [np.median(np.array(layer_stack), axis=0) for layer_stack in zip(*weights)]
        aggregated_parameters = ndarrays_to_parameters(median_weights)

        losses = [fit_res.metrics.get("loss", 0.0) for _, fit_res in results]
        avg_loss = float(np.mean(losses)) if losses else 0.0

        return aggregated_parameters, {"loss": avg_loss}

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        num_clients = min(5, len(client_manager.clients))
        clients = client_manager.sample(num_clients)
        evaluate_ins = EvaluateIns(parameters, {"round": server_round})
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        losses = [res.loss for _, res in results if res.loss is not None]
        accuracies = [res.metrics.get("accuracy", 0.0) for _, res in results]
        precisions = [res.metrics.get("precision", 0.0) for _, res in results]
        recalls = [res.metrics.get("recall", 0.0) for _, res in results]
        f1_scores = [res.metrics.get("f1_score", 0.0) for _, res in results]

        avg_loss = float(np.mean(losses)) if losses else None
        avg_acc = float(np.mean(accuracies)) if accuracies else 0.0
        avg_prec = float(np.mean(precisions)) if precisions else 0.0
        avg_rec = float(np.mean(recalls)) if recalls else 0.0
        avg_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

        return avg_loss, {
            "accuracy": avg_acc,
            "precision": avg_prec,
            "recall": avg_rec,
            "f1_score": avg_f1,
        }

    def evaluate(
        self,
        rnd: int,
        parameters: Parameters,
        config: Optional[Dict[str, Scalar]] = None
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None
