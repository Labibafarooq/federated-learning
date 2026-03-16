import json
import matplotlib.pyplot as plt

def plot_multiple_loss_curves(data):
    plt.figure(figsize=(10, 6))
    for label, loss_data in data.items():
        rounds = [r[0] for r in loss_data]
        losses = [r[1] for r in loss_data]
        plt.plot(rounds, losses, marker='o', label=f"{label}")
    
    plt.title("Effect of Epochs per Round on Model Convergence")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("epoch_comparison_loss_curve.png")
    plt.show()
    print("✅ Plot saved as epoch_comparison_loss_curve.png")

if __name__ == "__main__":
    with open("epoch.json", "r") as f:
        epoch_data = json.load(f)
    plot_multiple_loss_curves(epoch_data)
