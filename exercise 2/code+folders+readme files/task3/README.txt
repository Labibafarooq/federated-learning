Comparative Training Report: α = 0.1, 1, 10

This report outlines and compares training outcomes of three classification models under different 
data heterogeneity conditions controlled by the parameter α (alpha). Each model was trained over 
50 rounds with key metrics—loss and accuracy—recorded throughout.

🔍 Experiment Summary
Model	  Alpha Value	       Data Heterogeneity	                Training Rounds
alpha01	  0.1			Low					50
alpha1	  1			Moderate				50
alpha10	  10			High					50


📉 Loss Comparison
Round	alpha01 Loss	alpha1 Loss	alpha10 Loss
1	2.2412		2.2747		2.2390
10	1.6598		2.2087		1.5385
20	1.6588		2.1708		1.5371
30	1.6629		2.1348		1.5361
40	1.6618		2.0914		1.5326
50	1.6618		2.0468		1.5288

Observations:

    alpha01: Shows rapid loss reduction early, then plateaus from round 10 onward.

    alpha1: Loss steadily decreases but remains relatively high, suggesting slower convergence.

    alpha10: Achieves fastest and lowest loss convergence among the three.


✅ Accuracy Comparison

Round	alpha01 Accuracy	alpha1 Accuracy	        alpha10 Accuracy
1	0.1529			0.3710			0.2770
10	0.5647			0.4721			0.6041
20	0.5833			0.4147			0.5986
30	0.5255			0.4420			0.6033
40	0.5548			0.4659			0.5932
50	0.5294			0.4641			0.5713

Observations:

    alpha01: Peaks at ~58.7% (round 33), but ends with a decline, indicating possible overfitting.

    alpha1: Gradual improvements, ends near its peak (~46%), with low variance.

    alpha10: Fast early gains and a more stable peak near ~60%, slightly tapering off later.

🧠 Interpretation
🔹 Alpha = 0.1 (Low Heterogeneity)

    High initial gains but struggles to maintain peak accuracy.

    Likely overfit due to uniform data and limited generalization.

🔹 Alpha = 1 (Moderate Heterogeneity)

    Moderate learning with noisier but steady progress.

    No major overfitting, but limited ceiling due to data diversity.

🔹 Alpha = 10 (High Heterogeneity)

    Best early and peak performance.

    Quick convergence in both loss and accuracy.

    Slight drop later may be due to client inconsistency or aggregation limits.

📌 Key Takeaways

    Low α (0.1): Fast learner, but vulnerable to overfitting.

    Medium α (1): Balanced but slower learner, decent generalization.

    High α (10): Strong early generalization, fast convergence, best peak accuracy.
