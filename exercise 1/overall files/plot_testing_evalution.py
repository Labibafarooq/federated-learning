import json
import matplotlib.pyplot as plt

def load_loss_history(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    rounds = [item[0] for item in data["loss"]]
    losses = [item[1] for item in data["loss"]]
    return rounds, losses

def plot_loss(rounds, losses):
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, losses, marker="o", color="blue", linestyle="-")
    plt.title("Training Loss per Round")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("evaluation.png")
    print("✅ Plot saved as evaluation.png")

if __name__ == "__main__":
    rounds, losses = load_loss_history("server_history.json")  # Your actual history file
    print("Loss history:", losses)
    
    if not losses:
        print("⚠️ No loss data found! Please check the JSON file.")
    else:
        plot_loss(rounds, losses)
