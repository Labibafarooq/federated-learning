import json
import matplotlib.pyplot as plt

def load_history(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data

def plot_loss(history):
    rounds = [entry[0] for entry in history["loss"]]
    losses = [entry[1] for entry in history["loss"]]

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, losses, marker="o", color="red", linestyle="-")
    plt.title("Training Loss per Round")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.close()
    print("✅ Plot saved as loss_curve.png")

def plot_metrics(history):
    metrics = history["metrics"]
    rounds = [entry[0] for entry in metrics["accuracy"]]

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, [v[1] for v in metrics["accuracy"]], marker="o", label="Accuracy")
    plt.plot(rounds, [v[1] for v in metrics["precision"]], marker="o", label="Precision")
    plt.plot(rounds, [v[1] for v in metrics["recall"]], marker="o", label="Recall")
    plt.plot(rounds, [v[1] for v in metrics["f1_score"]], marker="o", label="F1 Score")

    plt.title("Evaluation Metrics per Round")
    plt.xlabel("Round")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.savefig("metrics_curve.png")
    plt.close()
    print("✅ Plot saved as metrics_curve.png")

if __name__ == "__main__":
    history = load_history("server_history.json")

    if not history or not history.get("loss"):
        print("⚠️ No loss data found! Please check the JSON file.")
    else:
        plot_loss(history)

    if not history or not history.get("metrics"):
        print("⚠️ No metrics data found! Please check the JSON file.")
    else:
        plot_metrics(history)
