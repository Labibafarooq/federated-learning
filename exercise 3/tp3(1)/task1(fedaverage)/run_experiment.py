#run_experiment.py
import subprocess
import argparse
import time
import random

def run_experiment(alpha: str, attack_type: str, malicious_ratio: float, tag: str):
    NUM_CLIENTS = 10
    num_malicious = int(NUM_CLIENTS * malicious_ratio)
    malicious_clients = random.sample(range(NUM_CLIENTS), num_malicious)

    print(f"🔁 Starting experiment: {tag}")
    print(f"🔢 Malicious clients: {malicious_clients} ({attack_type})")

    # Launch server
    server_cmd = [
        "python3", "run_server.py",
        "--alpha", alpha,
        "--tag", tag
    ]
    server_proc = subprocess.Popen(server_cmd)
    time.sleep(3)  # wait for server to boot up

    # Launch clients
    client_procs = []
    for cid in range(NUM_CLIENTS):
        assigned_attack = attack_type if cid in malicious_clients else "none"
        client_cmd = [
            "python3", "run_client.py",
            "--cid", str(cid),
            "--alpha", alpha,
            "--attack_type", assigned_attack
        ]
        proc = subprocess.Popen(client_cmd)
        client_procs.append(proc)
        time.sleep(1)  # slight delay to avoid overloading

    # Wait for all clients to finish
    for proc in client_procs:
        proc.wait()

    # Kill server after all clients finish
    server_proc.terminate()
    server_proc.wait()
    print(f"✅ Experiment {tag} completed.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Experiment Runner")
    parser.add_argument("--alpha", type=str, required=True)
    parser.add_argument("--attack_type", type=str, choices=["data", "model"], required=True)
    parser.add_argument("--malicious_ratio", type=float, choices=[0.0, 0.25, 0.5], required=True)
    parser.add_argument("--tag", type=str, required=True, help="Unique name for saving results")

    args = parser.parse_args()
    run_experiment(
        alpha=args.alpha,
        attack_type=args.attack_type,
        malicious_ratio=args.malicious_ratio,
        tag=args.tag
    )
