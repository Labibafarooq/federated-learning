# 📘 Results Visualizer Tool – Alpha = 10

This tool helps visualize training results from **federated learning experiments**, such as **FedAvg** under different data heterogeneity conditions.

This README corresponds to **α = 10**, representing **low data heterogeneity** — nearly IID data across clients.

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
| Dirichlet Alpha (α)           | **10** (low heterogeneity)    |

---

## 📈 Results Summary (α = 10)

### ✅ Accuracy Over Rounds

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
| 50    | 89.37        |

🔍 **Interpretation:**
- The model starts with **~21% accuracy** at round 1.
- Reaches **~68%** by round 5 — a major jump due to effective aggregation across clients.
- Smooth and consistent convergence toward **~89.4% by round 50**.
- Minimal fluctuations due to low client drift (α = 10 → near-IID).

---

### 🔻 Loss Over Rounds

| Round | Loss      |
| ----- | --------- |
| 1     | 2.217     |
| 5     | 0.820     |
| 10    | 0.658     |
| 20    | 0.578     |
| 30    | 0.557     |
| 40    | 0.493     |
| 45    | 0.483     |
| 50    | 0.454     |

🔍 **Interpretation:**
- Initial loss is high (~2.2) as the model begins random initialization.
- Loss rapidly drops and **smoothly converges to ~0.45**.
- Consistently declining trend shows stable learning and effective client aggregation.

---

## 📌 Key Takeaways

- **Low Heterogeneity (α = 10):**
  - Clients have **similar data distributions**.
  - Local models align closely, enabling the global model to converge quickly.
  - Aggregation is highly effective → **high accuracy & low loss**.

- **Model Behavior:**
  - **Fast early learning** (huge gain from round 1 to 5).
  - **Plateau after round 40**, indicating nearing convergence.
  - **Final Accuracy: ~89.4%**; **Final Loss: ~0.45**

---

## 📊 Visual Outputs (from `results_visualizer.py`)

The script generates:

- `loss.png`: Training loss per round
- `accuracy.png`: Accuracy per round



