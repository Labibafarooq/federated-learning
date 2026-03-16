import json
import matplotlib.pyplot as plt

def load_loss_history(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data

def plot_clients_loss(data):
    plt.figure(figsize=(8,5))

    for key, rounds_losses in data.items():
        rounds = [item[0] for item in rounds_losses]
        losses = [item[1] for item in rounds_losses]
        plt.plot(rounds, losses, marker='o', label=key)

    plt.title("Training Loss per Round for Different Number of Clients")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("clients_loss_comparison.png")
    plt.show()
    print("✅ Plot saved as clients_loss_comparison.png")

if __name__ == "__main__":
    data = load_loss_history("noofclients.json")
    plot_clients_loss(data)
