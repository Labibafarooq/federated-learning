import argparse
import torch
import flwr as fl
from model import CustomFashionModel
from load_client_data import load_client_data
from custom_client import CustomClient
from config import BATCH_SIZE

def main():
    parser = argparse.ArgumentParser(description="Run a Flower client.")
    parser.add_argument("--cid", type=int, required=True, help="Client ID to load data for")
    parser.add_argument("--alpha", type=str, required=True, help="Alpha setting: alpha01, alpha1, alpha10")
    parser.add_argument("--attack_type", type=str, default="none", choices=["none", "data", "model"], help="Type of client: none, data, model")

    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = f"./client_data_{args.alpha}"
    train_loader, val_loader = load_client_data(cid=args.cid, data_dir=data_dir, batch_size=BATCH_SIZE)

    model = CustomFashionModel().to(DEVICE)
    client = CustomClient(model=model, train_loader=train_loader, test_loader=val_loader, device=DEVICE, attack_type=args.attack_type)

    fl.client.start_client(server_address="localhost:8080", client=client.to_client())

if __name__ == "__main__":
    main()
