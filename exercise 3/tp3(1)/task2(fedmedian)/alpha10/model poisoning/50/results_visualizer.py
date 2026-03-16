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

        # Load losses
        loss_data = data.get("loss", [])
        self.rounds = [item[0] for item in loss_data]
        self.losses = [item[1] for item in loss_data]

        # Load other metrics
        metrics_data = data.get("metrics", {})
        self.metrics = {}
        if not metrics_data:
            print("⚠️ No evaluation metrics found in the JSON file.")
        else:
            for metric_name, value_list in metrics_data.items():
                self.metrics[metric_name] = [item[1] for item in value_list]
            print(f"✅ Loaded metrics: {list(self.metrics.keys())}")

    def plot_results(self, fig_directory: str) -> None:
        os.makedirs(fig_directory, exist_ok=True)

        # Plot loss
        if self.losses:
            plt.figure()
            plt.plot(self.rounds, self.losses, marker="o", label="Loss")
            plt.xlabel("Round")
            plt.ylabel("Loss")
            plt.title("Loss per Round")
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
        columns.extend([metric.capitalize() for metric in self.metrics])
        table.field_names = columns

        for i, round_num in enumerate(self.rounds):
            row = [round_num]
            if self.losses:
                row.append(self.losses[i])
            for metric in self.metrics:
                row.append(self.metrics[metric][i])
            table.add_row(row)

        print(table)


def main():
    # 📂 JSON file and figure output directory
    json_file = "/home/hproc03/Documents/federationLearning/server_history_fedmedian_model_50_alpha10.json"   # <--- Update this if your file is different
    fig_output = "figures_median_alpha10_model50%"              # <--- Folder to save plots

    visualizer = ResultsVisualizer()
    visualizer.load_simulation_results(json_file)
    visualizer.print_results_table()
    visualizer.plot_results(fig_output)

if __name__ == "__main__":
    main()
