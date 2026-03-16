import json
import matplotlib.pyplot as plt

def load_dirichlet_data(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    rounds_alpha1 = [r[0] for r in data["alpha_1"]]
    loss_alpha1 = [r[1] for r in data["alpha_1"]]
    rounds_alpha12 = [r[0] for r in data["alpha_12"]]
    loss_alpha12 = [r[1] for r in data["alpha_12"]]
    return rounds_alpha1, loss_alpha1, rounds_alpha12, loss_alpha12

def plot_dirichlet_comparison(rounds1, loss1, rounds2, loss2):
    plt.figure(figsize=(8, 5))
    plt.plot(rounds1, loss1, marker='o', linestyle='-', color='blue', label='α = 1 (High heterogeneity)')
    plt.plot(rounds2, loss2, marker='o', linestyle='-', color='green', label='α = 12 (Low heterogeneity)')
    plt.title("Training Loss Comparison: Dirichlet α Parameter")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig("dirichlet_alpha_comparison.png")
    plt.show()
    print("✅ Plot saved as dirichlet_alpha_comparison.png")

if __name__ == "__main__":
    rounds1, loss1, rounds2, loss2 = load_dirichlet_data("dirichlet_results.json")
    plot_dirichlet_comparison(rounds1, loss1, rounds2, loss2)
