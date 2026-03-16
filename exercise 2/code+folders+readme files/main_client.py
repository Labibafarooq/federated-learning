import torch
from model import CustomFashionModel as Net  # ✅ Your custom model
from load_client_data import load_client_data  # ✅ Correct function to load client data
from custom_client import CustomClient
import flwr as fl

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load client-specific data
client_id = 0  # Change this for different clients
train_loader, test_loader = load_client_data(cid=client_id, data_dir="./client_data", batch_size=32)

# Load model
model = Net().to(DEVICE)

# Create client
client = CustomClient(model, train_loader, test_loader, DEVICE)

# Start client
fl.client.start_client(server_address="localhost:8080", client=client.to_client())
