📘 Federated Learning - FedAvg with Alpha = 1

## 🧪 Experiment Summary

This experiment investigates the impact of **data heterogeneity** controlled by the **Dirichlet parameter α**.
This README presents the detailed results for **α = 1**, representing **moderate-to-high data heterogeneity** across clients.

---

## ⚙️ Configuration

| Parameter                     | Value                          |
| ----------------------------- | ------------------------------ |
| Algorithm                     | FedAvg                         |
| Dataset                       | Fashion-MNIST                  |
| Number of Clients             | 10                             |
| Fraction of Clients per Round | 1.0 (all clients participate)  |
| Number of Rounds              | 50                             |
| Local Epochs per Client       | 3                              |
| Batch Size                    | 64                             |
| Learning Rate                 | 0.01                           |
| Random Seed                   | 42                             |
| Dirichlet Alpha (α)           | **1** (moderate heterogeneity) |

---

## 📈 Results Summary (α = 1)

### 🔹 Accuracy Over Rounds

| Round | Accuracy (%) |
| ----- | ------------ |
| 1     | 29.48        |
| 5     | 64.50        |
| 10    | 73.95        |
| 15    | 76.50        |
| 20    | 77.49        |
| 25    | 78.05        |
| 30    | 78.00        |
| 35    | 78.99        |
| 40    | 79.77        |
| 45    | 79.75        |
| 50    | **80.13**    |

→ **Observation**: Accuracy improves steadily over time, reaching **80.13%** by round 50.
However, the learning curve is **less smooth** compared to α = 10, showing minor fluctuations, indicating some level of **client drift** due to non-IID data.

---

### 🔹 Loss Over Rounds

| Round | Loss     |
| ----- | -------- |
| 1     | 2.23     |
| 5     | 0.95     |
| 10    | 0.71     |
| 20    | 0.59     |
| 30    | 0.58     |
| 40    | 0.53     |
| 45    | 0.52     |
| 50    | **0.52** |

→ **Observation**: Loss consistently decreases, showing successful training.
The final loss plateaus at \~**0.52**, indicating acceptable generalization, though slightly worse than α = 10 due to higher heterogeneity.

---

### ✅ Accuracy Trends

* Initial accuracy: \~29.5% (round 1)
* Reaches \~64.5% by round 5, \~74% by round 10
* Final accuracy: **80.13% at round 50**
* Moderate variance observed after round 20, likely due to **non-IID data** across clients.

⚠️ Effect of Moderate Heterogeneity (α = 1)

* Clients receive **moderately non-IID** data → leads to **some client drift**.
* Model convergence is **slightly slower and less stable** than in α = 10.
* Accuracy improvements are incremental after round 25, with small oscillations in both accuracy and loss.

---

## 📎 Files Included

* `alpha1_results.json` — Raw results (accuracy and loss per round)
* `README.md` — This file
* `figure1` — figures files
