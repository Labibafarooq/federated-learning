#!/bin/bash
set -euo pipefail

# Directory with Docker Compose files
PROJECT_DIR="."
CUSTOM_NETWORKS=("ml_comm_network" "dps_network")

# Show usage guide
print_help() {
  echo "Usage: $0 <docker-compose-yml-file>"
  echo "Example: $0 docker-compose-task2-modified.yml"
  exit 1
}

# Argument check
if [[ $# -ne 1 ]]; then
  echo "❌ Error: Docker Compose file not provided."
  print_help
fi

CONFIG_FILE="$1"

# File existence validation
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "❌ File '$CONFIG_FILE' not found."
  exit 1
fi

# Remove existing Docker networks (if any)
clean_network() {
  local net_name="$1"
  if docker network ls --format '{{.Name}}' | grep -q "^${net_name}$"; then
    echo "🧯 Removing old network: $net_name"
    docker network rm "$net_name"
  else
    echo "ℹ️ Network $net_name not present, skipping."
  fi
}

# Stop running containers gracefully
echo "🔻 Shutting down running containers..."
docker compose -f "$CONFIG_FILE" down --remove-orphans

# Remove exited containers
docker container prune -f

# Prune unused docker networks
echo "🧼 Cleaning unused Docker networks..."
docker network prune -f

# Remove specified networks
for network in "${CUSTOM_NETWORKS[@]}"; do
  clean_network "$network"
done

# Start the full stack
echo "🚀 Building and launching services..."
docker compose -f "$CONFIG_FILE" up --build
