import json
import matplotlib.pyplot as plt

def load_experiments(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data

def plot_comparison(data):
    plt.figure(figsize=(10, 6))
    for key, values in data.items():
        rounds = [v[0] for v in values]
        losses = [v[1] for v in values]
        plt.plot(rounds, losses, marker='o', label=key.replace('_', ' ').title())
    
    plt.title("Effect of Batch Size and Learning Rate on Training Loss")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig("batch.png")
    plt.show()
    print("✅ Plot saved as batch.png")

if __name__ == "__main__":
    data = load_experiments("batch.json")
    plot_comparison(data)
