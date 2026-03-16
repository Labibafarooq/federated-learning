import torch
import flwr as fl
import numpy as np
from typing import List
from config import LEARNING_RATE
from model import CustomFashionModel

from flwr.common import (
    GetPropertiesIns, GetPropertiesRes,
    GetParametersIns, GetParametersRes,
    Parameters, FitRes, FitIns,
    EvaluateIns, EvaluateRes, Status, Code,
    ndarrays_to_parameters, parameters_to_ndarrays
)
from flwr.client import Client


class CustomClient(Client):
    def __init__(self, model: CustomFashionModel, train_loader, test_loader, device: torch.device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE)

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        return GetPropertiesRes(
            status=Status(code=Code.OK, message="Properties received"),
            properties={"device": str(self.device), "framework": "PyTorch"}
        )

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        ndarrays = self.model.get_model_parameters()
        parameters = ndarrays_to_parameters(ndarrays)
        return GetParametersRes(
            status=Status(code=Code.OK, message="Parameters sent"),
            parameters=parameters
        )

    def fit(self, ins: FitIns) -> FitRes:
        parameters = parameters_to_ndarrays(ins.parameters)
        self.model.set_model_parameters(parameters)
        self.model.train()

        # Save global model parameters for FedProx
        global_params = [p.clone().detach() for p in self.model.parameters()]
        mu = 0.1  # FedProx hyperparameter

        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            # FedProx proximal term
            prox_reg = 0.0
            for param, global_param in zip(self.model.parameters(), global_params):
                prox_reg += ((param - global_param.to(self.device)) ** 2).sum()
            loss += (mu / 2) * prox_reg

            loss.backward()
            self.optimizer.step()

        # Return updated model parameters
        updated_ndarrays = self.model.get_model_parameters()
        updated_parameters = ndarrays_to_parameters(updated_ndarrays)

        return FitRes(
            status=Status(code=Code.OK, message="Training complete"),
            parameters=updated_parameters,
            num_examples=len(self.train_loader.dataset),
            metrics={}
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        parameters = parameters_to_ndarrays(ins.parameters)
        self.model.set_model_parameters(parameters)
        self.model.eval()

        loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += self.criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        total = len(self.test_loader.dataset)
        loss /= total
        accuracy = correct / total

        return EvaluateRes(
            status=Status(code=Code.OK, message="Evaluation complete"),
            loss=loss,
            num_examples=total,
            metrics={"accuracy": accuracy}
        )
