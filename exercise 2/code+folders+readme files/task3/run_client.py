#run_client file
import argparse
import torch
import flwr as fl
from model import CustomFashionModel
from load_client_data import load_client_data
from custom_client import CustomClient
from config import BATCH_SIZE


def main():
    # Parse the client ID from command-line arguments
    parser = argparse.ArgumentParser(description="Run a Flower client.")
    parser.add_argument("--cid", type=int, required=True, help="Client ID to load data for")
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load client data
    train_loader, val_loader = load_client_data(cid=args.cid, data_dir="./client_data_alpha10~", batch_size=BATCH_SIZE)

    # Initialize the model
    model = CustomFashionModel().to(DEVICE)

    # Instantiate Flower client
    client = CustomClient(model=model, train_loader=train_loader, test_loader=val_loader, device=DEVICE)

    # Start the client and connect to the server
    fl.client.start_client(server_address="localhost:8080", client=client.to_client())

    

if __name__ == "__main__":
    main()
