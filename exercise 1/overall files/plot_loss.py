import json
import matplotlib.pyplot as plt

def load_loss_history(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return list(range(1, len(data) + 1)), data  # Rounds, Losses

def plot_loss(rounds, losses):
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, losses, marker="o", color="blue", linestyle="-")
    plt.title("Training Loss per Round")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("loss_curve.png")  # Saves the plot as a PNG file
    print("✅ Plot saved as loss_curve.png")

if __name__ == "__main__":
    rounds, losses = load_loss_history("loss_history.json")
    plot_loss(rounds, losses)
print("Loss history:", losses)
