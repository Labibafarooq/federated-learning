import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Tuple

class CustomFashionModel(nn.Module):
    def __init__(self) -> None:
        super(CustomFashionModel, self).__init__()
        # Example: simple CNN for FashionMNIST (1 channel, 28x28 images)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes in FashionMNIST

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))  # (batch, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # (batch, 64, 7, 7)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def train_epoch(self, train_loader: DataLoader,
                    criterion: nn.Module, optimizer: torch.optim.Optimizer,
                    device: torch.device) -> Tuple[float, float]:
        self.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = self(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def test_epoch(self, test_loader: DataLoader,
                   criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
        self.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def get_model_parameters(self) -> List[np.ndarray]:
        # Extract parameters and convert to numpy arrays (CPU)
        return [param.cpu().detach().numpy() for param in self.parameters()]

    def set_model_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from numpy arrays
        for param_tensor, param_numpy in zip(self.parameters(), parameters):
            param_tensor.data = torch.from_numpy(param_numpy).to(param_tensor.device)
