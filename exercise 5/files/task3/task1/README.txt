Federated Learning Performance Comparison Report (FedAvg - α = 0.1, 1, 10)**

🎯 **Objective**

To analyze how **data heterogeneity** (controlled by the Dirichlet distribution parameter α) impacts the performance of the **FedAvg algorithm** using the **Fashion-MNIST** dataset across 10 clients.

---

### 🔍 **Experimental Setup Summary**

| Parameter          | Value                                                         |
| ------------------ | ------------------------------------------------------------- |
| Algorithm          | FedAvg                                                        |
| Dataset            | Fashion-MNIST                                                 |
| Clients            | 10                                                            |
| Rounds             | 50                                                            |
| Fraction per Round | 1.0                                                           |
| Local Epochs       | 3                                                             |
| Batch Size         | 64                                                            |
| Learning Rate      | 0.01                                                          |
| Seed               | 42                                                            |
| α Values Compared  | 0.1 (high het.), 1 (moderate het.), 10 (low het.)             |

---

## 📊 **Results Overview**

| Metric                   | α = 0.1 (High Het.) | α = 1 (Moderate Het.) | α = 10 (Low Het.) |
| ------------------------ | ------------------- | --------------------- | ----------------- |
| **Initial Accuracy**     | 26.52%              | 29.48%                | 21.20%            |
| **Accuracy @ Round 10**  | 76.08%              | 73.95%                | 76.32%            |
| **Final Accuracy**       | **85.83%**          | 80.13%                | **89.37%**        |
| **Final Loss**           | 0.44                | 0.52                  | **0.454**         |
| **Convergence Behavior** | Fluctuating         | Slightly unstable     | Smooth & steady   |
| **Training Stability**   | Least stable        | Moderate              | Most stable       |

---

## 📈 **Accuracy Trends**

* **α = 10 (Low Heterogeneity)**:

  * Achieved **89.37%** accuracy.
  * **Fast and smooth convergence**, minimal fluctuations.
  * Data was nearly IID, enabling consistent aggregation.

* **α = 1 (Moderate Heterogeneity)**:

  * Reached **80.13%** accuracy.
  * **Moderate fluctuations** and slower convergence after round 25.
  * Indicates some **client drift** and inconsistencies due to partially non-IID data.

* **α = 0.1 (High Heterogeneity)**:

  * Achieved **85.83%** accuracy.
  * Despite highly non-IID data, performance was surprisingly **strong** but had **frequent oscillations** and **delayed smooth convergence** (stabilized after round 30).
  * Indicates **greater client drift** and instability during training.

---

## 📉 **Loss Trends**

* All three settings show **declining loss**, confirming training progression.
* **α = 10** has the **lowest final loss (0.454)** and most **consistent trend**.
* **α = 0.1** has a **more erratic loss curve** but still ends with a decent **final loss (0.44)**.
* **α = 1** ends at **0.52**, slightly higher, indicating **less efficient generalization**.

---

## 🧠 **Interpretation & Insights**

### 1. **Impact of Data Heterogeneity**

* **Higher heterogeneity (lower α)** leads to:

  * **Greater variation in client data distributions**
  * **Increased client drift**
  * **Oscillatory training behavior**
  * **Delayed convergence**
* However, **final accuracy can still be high** with enough training rounds.

### 2. **FedAvg's Robustness**

* FedAvg handles **low heterogeneity (α = 10)** with **fast and stable convergence**.
* Under **high heterogeneity (α = 0.1)**, although the learning curve is unstable, the algorithm **still converges to high accuracy** — showing **resilience** with longer training.
* **Moderate α = 1** leads to a compromise between stability and performance.

---

## ✅ **Conclusion**

Data heterogeneity significantly influences **convergence speed**, **training stability**, and **final accuracy** in federated learning:

* **Low Heterogeneity (α = 10)**: Best overall performance, smooth training, minimal drift.
* **Moderate Heterogeneity (α = 1)**: Acceptable performance, minor instability.
* **High Heterogeneity (α = 0.1)**: High variance and slower convergence, but still achieves strong accuracy with extended rounds.


