# ✅ run_server.py (Updated for Krum with automatic `f`)

import argparse
import flwr as fl
import json
from krum_strategy import KrumStrategy  # Make sure this file exists

def main():
    parser = argparse.ArgumentParser(description="Run Flower server.")
    parser.add_argument("--alpha", type=str, required=True, help="Data partitioning alpha")
    parser.add_argument("--tag", type=str, required=True, help="Unique tag for experiment")
    parser.add_argument("--malicious_ratio", type=float, required=True, help="Ratio of malicious clients")

    args = parser.parse_args()

    num_clients = 10
    f = int(num_clients * args.malicious_ratio)

    print(f"\U0001F680 Starting server for {args.tag} on localhost:8080... f={f}")

    strategy = KrumStrategy(f=f)

    history = fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )

    results_file = f"server_history_{args.tag}.json"
    metrics = history.metrics_centralized or history.metrics_distributed or {}

    if not metrics:
        print("\u26A0\uFE0F No metrics found in history.")
    else:
        with open(results_file, "w") as f:
            json.dump(metrics, f)
            print(f"\u2705 Results saved to {results_file}")

if __name__ == "__main__":
    main()
