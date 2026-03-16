 **Federated Learning - FedAvg with Alpha = 0.1**

🧪 **Experiment Summary**

This experiment explores the effects of **high data heterogeneity** using the **Dirichlet distribution** with **α = 0.1**. This setting represents **highly non-IID data** across clients, which simulates a challenging real-world federated learning scenario.

---

🔧 **Configuration**

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
| Dirichlet Alpha (α)           | **0.1** (high heterogeneity)  |

---

📈 **Results Summary (α = 0.1)**

### Accuracy Over Rounds

| Round | Accuracy (%) |
| ----- | ------------ |
| 1     | 26.52        |
| 5     | 70.62        |
| 10    | 76.08        |
| 15    | 77.40        |
| 20    | 77.69        |
| 25    | 79.68        |
| 30    | 76.98        |
| 35    | 80.77        |
| 40    | 84.10        |
| 45    | 85.07        |
| 50    | **85.83**    |

→ **Observation**: Despite high heterogeneity, the model shows **steady and strong improvements**, reaching **85.83%** accuracy by round 50. However, **oscillations** are more pronounced compared to α = 1 or α = 10, showing the instability induced by non-IID data.

---

### Loss Over Rounds

| Round | Loss     |
| ----- | -------- |
| 1     | 2.20     |
| 5     | 0.83     |
| 10    | 0.68     |
| 15    | 0.61     |
| 20    | 0.59     |
| 25    | 0.54     |
| 30    | 0.58     |
| 35    | 0.49     |
| 40    | 0.47     |
| 45    | 0.49     |
| 50    | **0.44** |

→ **Observation**: Loss consistently decreases with occasional **plateaus and spikes** — another indicator of the **non-IID effects**. The final loss value settles at **0.44**, which is competitive given the data distribution.

---

✅ **Accuracy Trends**

* Starts at \~26.5% (round 1)
* Surpasses 70% by round 5
* Crosses 80% by round 27
* Final accuracy: **85.83%**
* Noticeable **oscillations** and variance in mid-to-late rounds due to **client drift**.

⚠️ **Effect of High Heterogeneity (α = 0.1)**

* Clients have **highly skewed and imbalanced data**, resulting in **sharp fluctuations** in training curves.
* Convergence is slower in the middle rounds but **recovers strongly** after round 30.
* Accuracy reaches a high value, but the **training path is less smooth** than in more homogeneous scenarios.

---

📎 **Files Included**

* `alpha0.1_results.json` — Raw results (accuracy and loss per round)
* `README.md` — This file
* `figure2` — figures files

