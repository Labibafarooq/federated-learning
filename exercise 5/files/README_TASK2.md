# 🧠 Federated Learning DPS – Task 2

This setup simulates a distributed data processing system using Kafka, Docker, and Federated Learning.

## 🚀 What’s New in Task 2?

- 6 Data Sources (`data_gen_1` to `data_gen_6`)
- Two Kafka Brokers:
  - `broker_in`: receives raw data
  - `broker_out`: receives preprocessed data
- Preprocessing node (`dps_main`) now uses `processor_task2.py`

## 🧱 Architecture

```
data_gen_X → broker_in → dps_main → broker_out → ml_clients → ml_server
```

- **Data Sources** publish to `broker_in`
- **`dps_main`** normalizes/splits data using `processor_task2.py`
- **Preprocessed data** is sent to `broker_out`, then consumed by ML clients

## 🛠️ Run the System

```bash
docker compose -f docker-compose-task2.yml up --build
```

To stop:

```bash
docker compose -f docker-compose-task2.yml down
```

## ⚙️ Configuration

All settings (IPs, ports, training rounds) are in `.env`. Ensure IPs in `.env` match those in `docker-compose-task2.yml`.

✅ Already fixed: `DATASOURCE_DX_IP` entries now match the Docker IPs (10.60.0.31–36).

## 📂 Main Files

- `docker-compose-task2.yml`: container definitions
- `.env`: environment variables
- `processor_task2.py`: custom preprocessing logic
- `run_broker.sh`, `run_ndppf.sh`, `run_datasource.sh`: component runners

---

Let me know if you want to add logs, diagrams, or examples later!