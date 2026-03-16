📘 Results Visualizer Tool – Alpha = 1

This README corresponds to **α = 1**, representing **moderate data heterogeneity** — clients have somewhat diverse data distributions, but not extreme non-IID.

---

## 🛠 Configuration

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
| Dirichlet Alpha (α)           | **1** (moderate heterogeneity) |

---

## 📈 Results Summary (α = 1)

### ✅ Accuracy Over Rounds

| Round | Accuracy (%) |
| ----- | ------------ |
| 1     | 18.60        |
| 5     | 67.81        |
| 10    | 71.64        |
| 15    | 75.38        |
| 20    | 74.46        |
| 25    | 80.30        |
| 30    | 78.81        |
| 35    | 83.20        |
| 40    | 84.28        |
| 45    | 84.53        |
| 50    | 84.91        |

🔍 **Interpretation:**

* Training starts from a lower baseline (\~18.6%) due to heterogeneous initial data.
* Despite fluctuations, the model climbs steadily to reach **\~84.9% by round 50**.
* Learning is **less smooth** than α = 10, but convergence is still strong with some oscillation due to increased client drift.

---

### 🔻 Loss Over Rounds

| Round | Loss |
| ----- | ---- |
| 1     | 2.21 |
| 5     | 0.84 |
| 10    | 0.77 |
| 15    | 0.65 |
| 20    | 0.65 |
| 25    | 0.53 |
| 30    | 0.54 |
| 35    | 0.51 |
| 40    | 0.47 |
| 45    | 0.48 |
| 50    | 0.45 |

🔍 **Interpretation:**

* Initial high loss (\~2.21) reflects random weights and disjoint client data.
* Rapid drop till round 5, followed by oscillations.
* Final loss stabilizes at **\~0.45**, similar to α = 10, but with more fluctuations.

---

## 📌 Key Takeaways

* **Moderate Heterogeneity (α = 1):**

  * Clients have **somewhat different data distributions**.
  * Local updates diverge more than in near-IID settings.
  * Aggregation remains effective but slightly noisier, leading to learning variability.

* **Model Behavior:**

  * **Slower convergence** compared to α = 10.
  * **Performance plateaus around 84.9%** accuracy, showing that FedAvg still works well despite non-IID challenges.
  * **Loss trajectory includes minor bumps**, expected in more heterogeneous federated environments.

---

