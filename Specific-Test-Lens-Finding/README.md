# GSoC 2026 Specific Test: Gravitational Lens Finding (DeepLense)

This repository contains the implementation of a deep learning-based pipeline to automate the detection of Strong Gravitational Lenses from multi-band astronomical imaging data. This task is part of the specific evaluation for the ML4Sci (DeepLense) organization.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

### 🛰️ Project Overview

The goal is to classify images into two categories: Lenses and Non-Lenses. The dataset consists of simulated observational data with 3-filter bands, resulting in input tensors of shape (3, 64, 64).

#### Key Challenges

•	Class Imbalance: The number of non-lensed galaxies significantly outweighs the number of lensed objects.

•	Subtle Features: Distinguishing between faint arcs (lenses) and standard galactic morphologies requires high sensitivity to spatial features.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

### 🛠️ Strategy \& Implementation

#### 1\. Data Pipeline

To handle the (3, 64, 64) input and the class imbalance, the following strategy was adopted:

•	Preprocessing: Images were normalized across channels to ensure stable gradient flow during backpropagation.

•	Data Augmentation: To improve generalization and compensate for the limited number of "Lens" samples, random horizontal/vertical flips and rotations were applied during training.

•	Handling Imbalance: To address the \~1:100 class skew, I implemented a dual-pronged strategy: using a WeightedRandomSampler for balanced batch distribution and a pos\_weight (calculated as n\_{neg}/n\_{pos}) within BCEWithLogitsLoss. This combination ensures the model avoids a trivial majority-class solution and remains highly sensitive to the rare 'Lens' samples, as evidenced by the high recall in the test confusion matrix.

#### 2\. Model Architectures

I evaluated two state-of-the-art architectures to compare their performance in extracting features from astronomical data:

•	ResNet-18: A residual network known for its ability to train deep architectures without vanishing gradients. Its skip connections are particularly effective at preserving spatial details in small 64 x 64 images.

•	EfficientNet-B0: A model optimized via compound scaling. It provides high accuracy with significantly fewer parameters, making it computationally efficient for large-scale astronomical surveys.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

### 📊 Results and Evaluation

The models were evaluated using ROC-AUC and targeted metric checks at FPR ≈ 0.01.

#### Comparative Performance (Test Set)

Model	Test AUC	TPR (at FPR ≈ 0.01)	Precision	False Positives

EfficientNet-B0	0.9816	75.90% (148/195)	49.50%	151

ResNet-18	0.9724	74.87% (146/195)	39.67%	222



#### Confusion Matrix Analysis (Test Set)

At the targeted threshold:

•	EfficientNet-B0 achieved superior precision-recall balance, flagging only 151 false positives while capturing 148 lenses.

•	ResNet-18 captured 146 lenses but generated 222 false positives, indicating a higher noise-to-signal ratio at low FPR levels.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

### 📜 Conclusion

EfficientNet-B0 is the preferred model for this pipeline. It demonstrates better generalization and significantly higher precision (49.5% vs 39.7%) in the low-FPR regime. This efficiency is critical for modern astronomical surveys (like LSST), where minimizing false positives is paramount to reducing the computational and human workload of downstream verification.

