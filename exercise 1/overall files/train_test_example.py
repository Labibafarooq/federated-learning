#train_test_example.py
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import CustomFashionModel  # adjust import if your file is named differently
from config import BATCH_SIZE


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),  # converts PIL image to tensor and scales [0,255] to [0,1]
    ])

    # Load datasets
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Model, loss, optimizer
    model = CustomFashionModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    epochs = 5
    for epoch in range(epochs):
        train_loss, train_acc = model.train_epoch(train_loader, criterion, optimizer, device)
        test_loss, test_acc = model.test_epoch(test_loader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.4f}")

if __name__ == "__main__":
    main()
