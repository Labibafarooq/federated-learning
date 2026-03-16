 ✅ **1. Final Accuracy Comparison**

| Alpha (α) | Final Accuracy | Interpretation                                                                                          |
| --------- | -------------- | ------------------------------------------------------------------------------------------------------- |
| **10**    | **89.37%**     | Highest accuracy. Near-IID data leads to well-aligned updates.                                          |
| **1**     | **84.91%**     | Moderate accuracy. Some client drift due to data differences.                                           |
| **0.1**   | **81.80%**     | Lowest accuracy. Despite smooth learning, the lower diversity leads to slightly limited generalization. |

---

### ⚡ **2. Convergence Speed (based on accuracy progress per round)**

| Round        | α = 10    | α = 1 | α = 0.1 |
| ------------ | --------- | ----- | ------- |
| **Round 1**  | 21.2%     | 18.6% | 13.6%   |
| **Round 5**  | 68.2%     | 67.8% | 59.6%   |
| **Round 10** | 76.3%     | 71.6% | 71.2%   |
| **Round 20** | 82.2%     | 74.5% | 77.9%   |
| **Round 30** | 85.9%     | 78.8% | 80.3%   |
| **Round 50** | **89.4%** | 84.9% | 81.8%   |

🔍 **Conclusion on Speed:**

* **α = 10** has **fastest and smoothest convergence**, improving \~47% in just 5 rounds.
* **α = 1** converges slightly slower, with **oscillations** caused by moderate client drift.
* **α = 0.1** converges steadily and cleanly but starts lower and climbs slower in early rounds.

---

### 🔁 **Client Drift Analysis**

| Alpha (α) | Client Drift                          | Resulting Behavior                                                  |
| --------- | ------------------------------------- | ------------------------------------------------------------------- |
| **10**    | Minimal (near-IID)                    | Client models agree → fast, stable convergence                      |
| **1**     | Moderate                              | Client updates diverge more → mild oscillations                     |
| **0.1**   | Very Low Drift, but Very Similar Data | No major disagreement, but low data diversity limits generalization |

📌 **Note**: While α = 0.1 has very little client drift, the model's slower performance is not due to update conflicts but rather **overfitting to too similar data**, reducing its ability to generalize.

---

### 📉 **Loss Curve Behavior**

| Alpha (α) | Final Loss | Curve Behavior         |
| --------- | ---------- | ---------------------- |
| **10**    | 0.454      | Smooth, clean drop     |
| **1**     | 0.450      | Drop with oscillations |
| **0.1**   | 0.440      | Most stable descent    |

Despite α = 0.1 achieving the lowest loss, it does **not correspond to the best accuracy** due to the overfitting effect on similar client data.

---

### 🧠 Summary Table

| Metric                | α = 10                | α = 1                 | α = 0.1                     |
| --------------------- | --------------------- | --------------------- | --------------------------- |
| **Final Accuracy**    | ⭐ **89.4%** (highest) | ✅ 84.9%               | ❗ 81.8% (lowest)            |
| **Convergence Speed** | ⭐ Fastest             | Moderate (with bumps) | ❗ Slowest start             |
| **Client Drift**      | ⭐ Very Low            | Moderate              | ✅ Very Low                  |
| **Stability**         | ⭐ Very Stable         | Some oscillations     | ⭐ Most Stable               |
| **Generalization**    | ⭐ Best                | Good                  | ❗ Limited (IID overfitting) |

---

### 🧠 Final Insights

* **α = 10** is the sweet spot in this experiment — high accuracy, fast convergence, minimal drift.
* **α = 1** works well but introduces some instability due to moderate heterogeneity.
* **α = 0.1** leads to over-similar updates, resulting in slow learning and lower peak performance, even though the training loss looks great.

