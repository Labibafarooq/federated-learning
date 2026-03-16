#results_visualizer fiel:
import json
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import os

class ResultsVisualizer:
    def __init__(self) -> None:
        self.rounds = []
        self.losses = []
        self.metrics = {}

    def load_simulation_results(self, file_name: str) -> None:
        with open(file_name, "r") as f:
            data = json.load(f)

        # Parse loss: list of [round, loss]
        loss_data = data.get("loss", [])
        self.rounds = [item[0] for item in loss_data]
        self.losses = [item[1] for item in loss_data]

        # Parse metrics dictionary of lists: {"accuracy": [[round, val], ...], ...}
        self.metrics = {}
        metrics_data = data.get("metrics", {})

        if not metrics_data:
            print("⚠️ No evaluation metrics found in the JSON file.")
        else:
            for metric_name, values in metrics_data.items():
                # Extract only values (assumed to be list of [round, value])
                self.metrics[metric_name] = [v[1] for v in values]

            print(f"✅ Loaded metrics: {list(self.metrics.keys())}")

    def plot_results(self, fig_directory: str) -> None:
        os.makedirs(fig_directory, exist_ok=True)

        # Plot loss
        if self.losses:
            plt.figure()
            plt.plot(self.rounds, self.losses, marker="o", color="blue")
            plt.xlabel("Round")
            plt.ylabel("Loss")
            plt.title("Training Loss per Round")
            plt.grid(True)
            plt.savefig(os.path.join(fig_directory, "loss.png"))
            plt.close()

        # Plot each metric
        for metric_name, values in self.metrics.items():
            plt.figure()
            plt.plot(self.rounds, values, marker="o", label=metric_name.capitalize())
            plt.xlabel("Round")
            plt.ylabel(metric_name.capitalize())
            plt.title(f"{metric_name.capitalize()} per Round")
            plt.grid(True)
            plt.savefig(os.path.join(fig_directory, f"{metric_name}.png"))
            plt.close()

    def print_results_table(self) -> None:
        if not self.rounds:
            print("No data loaded.")
            return

        table = PrettyTable()
        columns = ["Round"]
        if self.losses:
            columns.append("Loss")
        columns.extend([name.capitalize() for name in self.metrics.keys()])
        table.field_names = columns

        for i, round_num in enumerate(self.rounds):
            row = [round_num]
            if self.losses:
                row.append(self.losses[i])
            for metric_name in self.metrics.keys():
                row.append(self.metrics[metric_name][i])
            table.add_row(row)

        print(table)


def main():
    visualizer = ResultsVisualizer()
    visualizer.load_simulation_results("server_history.json")  # Your JSON results file
    visualizer.print_results_table()
    visualizer.plot_results("figures5")  # Folder to save plots

if __name__ == "__main__":
    main()
