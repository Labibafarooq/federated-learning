# scaffold_strategy.py (correct server strategy)
from typing import List, Tuple, Optional, Dict
import numpy as np
import flwr as fl
from flwr.common import Parameters, Scalar
from flwr.common.parameter import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

class ScaffoldStrategy(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cc: Optional[List[np.ndarray]] = None

    def initialize_control_variate(self, initial_cc: List[np.ndarray]) -> None:
        self.cc = initial_cc

    def update_global_control_variate(self, cks: List[List[np.ndarray]]) -> None:
        if self.cc is None:
            self.cc = [np.zeros_like(layer) for layer in cks[0]]

        ck_avg = [np.mean(np.stack(layer_vals, axis=0), axis=0) for layer_vals in zip(*cks)]

        self.cc = [
            c_layer + (1 / len(cks)) * (ck_layer - c_layer)
            for c_layer, ck_layer in zip(self.cc, ck_avg)
        ]

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        fit_ins = super().configure_fit(server_round, parameters, client_manager)

        config_with_cv = []
        for client, ins in fit_ins:
            config = ins.config.copy()
            if self.cc is not None:
                control_variate_str = str([arr.tolist() for arr in self.cc])
                config["control_variate"] = control_variate_str
            config_with_cv.append((client, fl.common.FitIns(ins.parameters, config)))

        return config_with_cv

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        print(f"[Scaffold] Aggregating fit results for round {server_round}")

        if not results:
            return None, {}

        weights = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        averaged_weights = [np.mean(np.stack(layers, axis=0), axis=0) for layers in zip(*weights)]
        aggregated_parameters = ndarrays_to_parameters(averaged_weights)

        cks = []
        for _, fit_res in results:
            if "ck" in fit_res.metrics:
                ck_list = [np.array(layer) for layer in fit_res.metrics["ck"]]
                cks.append(ck_list)

        if cks:
            self.update_global_control_variate(cks)

        return aggregated_parameters, {}
