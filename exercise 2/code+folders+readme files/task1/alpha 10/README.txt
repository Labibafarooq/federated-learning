📘 Federated Learning - FedAvg with Alpha = 10

## 🧪 Experiment Summary

This project implements the **FedAvg** algorithm in a Federated Learning setting, investigating the impact of **data heterogeneity** controlled by the **Dirichlet parameter α**. This README presents the detailed results for **α = 10**, representing **low data heterogeneity** across clients.

---

## ⚙️ Configuration

| Parameter                     | Value                         |
| ----------------------------- | ----------------------------- |
| Algorithm                     | FedAvg                        |
| Dataset                       | Fashion-MNIST                 |
| Number of Clients             | 10                            |
| Fraction of Clients per Round | 1.0 (all clients participate) |
| Number of Rounds              | 50                            |
| Local Epochs per Client       | 3                             |
| Batch Size                    | 64                            |
| Learning Rate                 | 0.01                          |
| Random Seed                   | 42                            |
| Batch Normalization           | Disabled                      |
| Dirichlet Alpha (α)           | **10** (low heterogeneity)    |

---

## 📈 Results Summary (α = 10)

### 🔹 Accuracy Over Rounds

| Round | Accuracy (%) |
| ----- | ------------ |
| 1     | 21.20        |
| 5     | 68.19        |
| 10    | 76.32        |
| 15    | 76.81        |
| 20    | 82.20        |
| 25    | 84.83        |
| 30    | 85.94        |
| 35    | 87.52        |
| 40    | 88.80        |
| 45    | 89.00        |
| 50    | **89.37**    |

→ **Observation**: Accuracy improves significantly and steadily, reaching nearly **89.4%** by round 50. Convergence is smooth with little fluctuation, which is expected for low heterogeneity (high α).

---

### 🔹 Loss Over Rounds

| Round | Loss      |
| ----- | --------- |
| 1     | 2.217     |
| 5     | 0.820     |
| 10    | 0.658     |
| 20    | 0.578     |
| 30    | 0.557     |
| 40    | 0.493     |
| 45    | 0.483     |
| 50    | **0.454** |

→ **Observation**: Loss steadily decreases over time, supporting consistent learning. Final loss is below **0.46**, indicating good model generalization.

---

## 🧠 Analysis

### ✅ Accuracy Trends

* Initial accuracy: \~21% (round 1)
* Reaches 68% by round 5, 76% by round 10
* Final accuracy: **89.37% at round 50**
* Model exhibits smooth convergence with minor fluctuations.

### ⚠️ Effect of Low Heterogeneity (α = 10)

* Clients receive nearly IID data → minimal client drift.
* Aggregated model benefits from stable local updates.
* Fast and stable convergence compared to runs with α = 1 or 0.1.

---

## 🔄 Validity of Custom Simulation Script

Although the professor recommends using a predefined `run_simulation` script, this experiment was conducted using a **custom script** that adheres to:

* Same FedAvg update strategy
* Same training configuration
* Per-round logging of metrics
* Proper Dirichlet-based data partitioning

## 📌 How to Reproduce

1. Set `alpha = 10` in your Dirichlet data partitioning function.
2. Ensure 10 clients and 50 rounds of training.
3. Run FedAvg with:

   * Learning rate = 0.01
   * Batch size = 64
   * Local epochs = 3
   * All clients participate every round
4. Log per-round accuracy and loss.

---

## 📎 Files Included

* `alpha10_results.json` — Raw results (accuracy and loss per round)
* `README.md` — This file
* `figure` — figures files
