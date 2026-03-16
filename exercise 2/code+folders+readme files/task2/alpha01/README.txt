📘 Results Visualizer Tool – Alpha = 0.1

This README corresponds to **α = 0.1**, representing **very low data heterogeneity** — clients have nearly identical (IID-like) data distributions.

---

## 🛠 Configuration

| Parameter                     | Value                            |
| ----------------------------- | -------------------------------- |
| Algorithm                     | FedAvg                           |
| Dataset                       | Fashion-MNIST                    |
| Number of Clients             | 10                               |
| Fraction of Clients per Round | 1.0 (all clients participate)    |
| Number of Rounds              | 50                               |
| Local Epochs per Client       | 3                                |
| Batch Size                    | 64                               |
| Learning Rate                 | 0.01                             |
| Dirichlet Alpha (α)           | **0.1** (very low heterogeneity) |

---

## 📈 Results Summary (α = 0.1)

### ✅ Accuracy Over Rounds

| Round | Accuracy (%) |
| ----- | ------------ |
| 1     | 13.63        |
| 5     | 59.56        |
| 10    | 71.17        |
| 15    | 75.43        |
| 20    | 77.93        |
| 25    | 78.98        |
| 30    | 80.30        |
| 35    | 81.02        |
| 40    | 81.38        |
| 45    | 81.56        |
| 50    | 81.80        |

🔍 **Interpretation:**

* Training starts lower due to random initialization and diversity among a few clients.
* Accuracy improves rapidly and stabilizes smoothly, reaching **\~81.8%**.
* Overall, **less oscillation and smoother convergence** compared to more heterogeneous cases.

---

### 🔻 Loss Over Rounds

| Round | Loss |
| ----- | ---- |
| 1     | 2.34 |
| 5     | 0.95 |
| 10    | 0.71 |
| 15    | 0.63 |
| 20    | 0.57 |
| 25    | 0.53 |
| 30    | 0.50 |
| 35    | 0.47 |
| 40    | 0.46 |
| 45    | 0.45 |
| 50    | 0.44 |

🔍 **Interpretation:**

* The initial high loss (\~2.34) reflects random initialization.
* A steady decline occurs throughout training, with final loss stabilizing at **\~0.44**.
* The **smooth descent** indicates strong global consistency among client updates.

---

## 📌 Key Takeaways

* **Very Low Heterogeneity (α = 0.1):**

  * Clients' local data distributions are almost identical (IID-like).
  * Results in highly aligned local updates and **smooth global convergence**.

* **Model Behavior:**

  * **Fast, stable learning** with minimal variance across rounds.
  * Final accuracy of **81.8%** is slightly lower than α = 1 or α = 10, but convergence is **cleaner**.
  * Loss curve is **monotonically decreasing** with no major bumps.

---

🚀 **Convergence Speed – FedAvg (α = 0.1)**

**Definition**: *Convergence speed* refers to how quickly the model reaches near-peak performance (in accuracy or loss) across rounds.

---

### 📊 Observations from the Results

| Round | Accuracy (%) | Remarks                          |
| ----- | ------------ | -------------------------------- |
| 1     | 13.63        | Random init, no learning yet     |
| 5     | 59.56        | **\~+46% gain in just 4 rounds** |
| 10    | 71.17        | Steep rise continues             |
| 15    | 75.43        | Slows down after round 10        |
| 20    | 77.93        |                                  |
| 25    | 78.98        | Plateau beginning                |
| 30    | 80.30        | Near-final performance           |
| 50    | 81.80        | Peak performance reached         |

---

### ✅ **Analysis**

* FedAvg reaches **\~80% accuracy by round 30**, meaning it achieves **>97.8% of its final performance** in just 60% of the total rounds.
* The model achieves **significant learning in the first 10 rounds**, suggesting **fast convergence** in the IID-like setting.
* Final performance improves only **\~1.5% between rounds 30 and 50**, showing that most learning **happens early**.

---

## 📉 Convergence Speed Summary

| Metric                                | Value                            |
| ------------------------------------- | -------------------------------- |
| Rounds to reach 95% of final accuracy | \~Round 28                       |
| Rounds to reach 80% accuracy          | \~Round 30                       |
| Final Accuracy                        | 81.8%                            |
| Initial Big Jump                      | From 13.63% → 59.56% in 5 rounds |
| Behavior                              | Fast rise, smooth plateau        |

---

## 📌 Comparative Insights (vs. α = 1 and α = 10)

| Feature                | α = 0.1 (IID) | α = 1 (Moderate)     | α = 10 (Highly Non-IID) |
| ---------------------- | ------------- | -------------------- | ----------------------- |
| Convergence Speed      | ✅ Fast        | ⚠️ Moderate          | ❌ Slower                |
| Accuracy Plateau Round | \~30          | \~35–40              | \~45–50                 |
| Final Accuracy         | \~81.8%       | \~84.9%              | \~86.6%                 |
| Smoothness             | ✅ Very smooth | ⚠️ Some oscillations | ❌ Frequent oscillations |

* **Conclusion**: **FedAvg converges fastest when α = 0.1**, due to minimal data heterogeneity.
* As α increases, convergence slows and becomes bumpier due to client drift and inconsistent local updates.

---

