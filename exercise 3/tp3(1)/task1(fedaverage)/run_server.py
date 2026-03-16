#run_server.py

import argparse
import flwr as fl
import json
from FedAvgStrategy import FedAvgStrategy as CustomFedAvgStrategy

def main():
    parser = argparse.ArgumentParser(description="Run Flower server.")
    parser.add_argument("--alpha", type=str, required=True, help="Data partitioning alpha")
    parser.add_argument("--tag", type=str, required=True, help="Unique tag for experiment (e.g. fedavg_data_25)")

    args = parser.parse_args()

    print(f"🚀 Starting server for {args.tag} on localhost:8080...")

    strategy = CustomFedAvgStrategy(alpha=args.alpha)

    # Run Flower server
    history = fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=10),  # You can increase to 50 if needed
        strategy=strategy,
    )

    # Save results
    # Save results (choose the correct history source)
    results_file = f"server_history_{args.tag}.json"
    metrics = history.metrics_centralized or history.metrics_distributed or {}

    if not metrics:
        print("⚠️ No metrics found in history.")
    else:
        with open(results_file, "w") as f:
            json.dump(metrics, f)
            print(f"✅ Results saved to {results_file}")


    print(f"📁 Results saved to {results_file}")

if __name__ == "__main__":
    main()
