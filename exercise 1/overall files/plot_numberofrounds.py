# plot_loss1.py (updated for server_history.json format)
import json
import matplotlib.pyplot as plt

def load_loss_history(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    loss_data = data.get("loss", [])
    rounds = [r[0] for r in loss_data]
    losses = [r[1] for r in loss_data]
    return rounds, losses

def plot_loss(rounds, losses):
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, losses, marker="o", color="blue", linestyle="-")
    plt.title("Training Loss per Round")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("loss_curve1.png")  # Saves the plot as a PNG file
    plt.show()
    print("✅ Plot saved as loss_curve1.png")

if __name__ == "__main__":
    rounds, losses = load_loss_history("server_history.json")
    print("Loss history:", losses)
    
    if not losses:
        print("⚠️ No loss data found! Please check the JSON file.")
    else:
        plot_loss(rounds, losses)
        
def plot_metric(metric_name, metric_data):
    rounds = [r[0] for r in metric_data]
    values = [r[1] for r in metric_data]
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, values, marker="o", linestyle="-", label=metric_name)
    plt.title(f"{metric_name.capitalize()} per Round")
    plt.xlabel("Round")
    plt.ylabel(metric_name.capitalize())
    plt.grid(True)
    plt.savefig(f"{metric_name}_curve.png")
    plt.show()
    print(f"✅ Plot saved as {metric_name}_curve.png")
