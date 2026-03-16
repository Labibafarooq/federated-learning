import torch
import flwr as fl
import numpy as np
from typing import List
from config import LEARNING_RATE, EPOCHS
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
    def __init__(self, model: CustomFashionModel, train_loader, test_loader, device: torch.device, attack_type: str = "none"):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.attack_type = attack_type
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE)

        self.ck = [np.zeros_like(p.cpu().detach().numpy()) for p in self.model.parameters()]
        self.c = [np.zeros_like(p.cpu().detach().numpy()) for p in self.model.parameters()]

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

        if "control_variate" in ins.config:
            self.c = [np.array(arr) for arr in eval(ins.config["control_variate"])]

        eta = LEARNING_RATE
        T = len(self.train_loader) * EPOCHS
        wt = [p.clone().detach() for p in self.model.parameters()]

        for epoch in range(EPOCHS):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)

                # ✨ Data Poisoning Attack: label flipping
                if self.attack_type == "data":
                    target = (target + 1) % 10

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()

                with torch.no_grad():
                    for param, ck_arr, c_arr in zip(self.model.parameters(), self.ck, self.c):
                        if param.grad is not None:
                            param.grad -= torch.from_numpy(ck_arr).to(self.device).type_as(param.grad)
                            param.grad += torch.from_numpy(c_arr).to(self.device).type_as(param.grad)

                self.optimizer.step()

        wt_plus_1 = [p.clone().detach() for p in self.model.parameters()]

        for i in range(len(self.ck)):
            self.ck[i] = self.ck[i] - self.c[i] + (1 / (eta * T)) * (wt[i].cpu().numpy() - wt_plus_1[i].cpu().numpy())

        # ✨ Model Poisoning Attack: scale model weights
        if self.attack_type == "model":
            for param in self.model.parameters():
                param.data = param.data * 5.0  # or add noise if preferred

        ck_serializable = [ck_layer.tolist() for ck_layer in self.ck]
        updated_parameters = ndarrays_to_parameters(self.model.get_model_parameters())

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

        print(f"[Client {self.attack_type}] Eval Accuracy: {accuracy*100:.2f}%, Loss: {loss:.4f}")

        return EvaluateRes(
            status=Status(code=Code.OK, message="Evaluation complete"),
            loss=loss,
            num_examples=total,
            metrics={"accuracy": accuracy}
        )
