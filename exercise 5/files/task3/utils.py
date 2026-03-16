import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_central_test_loader(batch_size=32):
    transform = transforms.Compose([transforms.ToTensor()])
    test_data = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    return DataLoader(test_data, batch_size=batch_size)

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters_to_tensors(parameters))
    state_dict = {k: v for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)

def parameters_to_tensors(parameters):
    import numpy as np
    import torch
    return [torch.tensor(np.array(p)) for p in parameters]

def test_model(model, testloader, device):
    model.eval()
    correct, total, loss_total = 0, 0, 0.0
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = loss_fn(outputs, y)
            loss_total += loss.item() * x.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return loss_total / total, correct / total
