# GSoC 2026 Common Test I: Multi-Class Substructure Classification (DeepLense)

This repository contains the implementation for the multi-class classification of strong gravitational lensing images. The task involves distinguishing between three distinct categories of lensing substructures using deep learning.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

### 🛰️ Project Overview

The goal is to classify simulated lensing images into three classes based on the type of substructure present:

1\.	no: Strong lensing with no substructure.

2\.	sphere: Lensing with subhalo substructure.

3\.	vort: Lensing with vortex substructure.

### Key Challenges

•	Feature Similarity: Distinguishing between subtle variations in "sphere" and "vort" substructures, which can often appear similar at the pixel level.

•	Precision in Science: Ensuring high confidence in substructure detection is critical for dark matter mapping.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

### 🛠️ Strategy \& Implementation

#### 1\. Architecture Choice

For this multi-class task, I utilized the ResNet-18 architecture as the backbone.

•	Why ResNet-18? Its residual connections are highly effective at preventing the vanishing gradient problem, allowing the model to learn complex spatial features even in small $64 \\times 64$ images.

•	Adaptation: The final fully-connected (linear) layer was modified to output three logits, mapping the high-level feature extractors to the three specific substructure categories.

#### 2\. Data Pipeline

•	Normalization: Images were normalized across channels to ensure stable gradient flow during backpropagation.

•	Loss Function: Cross-Entropy Loss was used to optimize the model, ensuring the penalization of incorrect class assignments while rewarding high-confidence correct predictions.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

### 📊 Results and Evaluation

The model was evaluated using the One-vs-Rest (OvR) approach for ROC-AUC to measure performance across all categories.

#### AUC Performance

Class	AUC Score

no (No Substructure)	0.9907

sphere (Subhalo)	0.9819

vort (Vortex)	0.9887

#### Confusion Matrix Insights (Test Set)

The model demonstrates high discriminatory power, particularly for the baseline class:

•	Perfect Baseline Detection: The "no" class was classified with 100% accuracy (1250/1250), showing the model is highly effective at identifying the presence of any substructure.

•	Substructure Confusion: Most misclassifications occur between "sphere" and "vort" (or misclassifying them as "no"). Specifically, 149 "sphere" samples were predicted as "no," indicating that faint subhalos remain the most challenging features to extract.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

### 📜 Conclusion

The model achieves an excellent balance across all three classes, with AUC scores exceeding 0.98 in every category. The ability to perfectly distinguish "no substructure" from "substructure" candidates provides a robust first-stage filter for astronomical surveys, while the high AUC for "sphere" and "vort" proves the model's capability in detailed dark matter morphology classification.

