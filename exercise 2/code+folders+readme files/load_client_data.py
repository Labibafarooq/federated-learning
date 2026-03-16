#load_client_data.py
import os
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

def load_client_data(cid: int, data_dir: str, batch_size: int) -> tuple[DataLoader, DataLoader]:
    file_path = os.path.join(data_dir, f"client_{cid}.pkl")
    
    with open(file_path, "rb") as f:
        client_data = pickle.load(f)

    data = client_data["data"]
    targets = client_data["targets"]

    # Normalize and reshape
    data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(1) / 255.0
    targets_tensor = torch.tensor(targets, dtype=torch.long)

    dataset = TensorDataset(data_tensor, targets_tensor)

    # Split into 80% train / 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
