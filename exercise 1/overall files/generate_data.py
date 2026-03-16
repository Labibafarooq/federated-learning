# generate_data.py

import os
import numpy as np
import pickle
from torchvision import datasets, transforms
from config import NUM_CLIENTS, ALPHA_DIRICHLET



def generate_distributed_datasets(k: int, alpha: float, save_dir: str = "./client_data") -> None:
    os.makedirs(save_dir, exist_ok=True)

    # Load FashionMNIST
    transform = transforms.ToTensor()
    dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    data = dataset.data.numpy()
    targets = dataset.targets.numpy()

    num_classes = 10
    class_indices = [np.where(targets == i)[0] for i in range(num_classes)]
    client_indices = [[] for _ in range(k)]

    for c in range(num_classes):
        class_idx = class_indices[c]
        np.random.shuffle(class_idx)

        proportions = np.random.dirichlet([alpha] * k)
        proportions = (proportions * len(class_idx)).astype(int)

        # Fix rounding issues
        while proportions.sum() < len(class_idx):
            proportions[np.random.randint(0, k)] += 1
        while proportions.sum() > len(class_idx):
            proportions[np.random.randint(0, k)] -= 1

        start = 0
        for i in range(k):
            end = start + proportions[i]
            client_indices[i].extend(class_idx[start:end])
            start = end

    # Save data for each client
    for i in range(k):
        client_data = {
            "data": data[client_indices[i]],
            "targets": targets[client_indices[i]]
        }
        with open(os.path.join(save_dir, f"client_{i}.pkl"), "wb") as f:
            pickle.dump(client_data, f)

# Example usage
if __name__ == "__main__":
    generate_distributed_datasets(k=NUM_CLIENTS, alpha=ALPHA_DIRICHLET)
