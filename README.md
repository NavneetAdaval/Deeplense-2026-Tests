# GSoC 2026 | DeepLense Evaluation Tests (ML4Sci)

This repository contains my solutions for the Google Summer of Code 2026 evaluation tasks for the ML4Sci (DeepLense) organization. The project focuses on leveraging Deep Learning to automate the detection and classification of gravitational lenses and their underlying dark matter substructures.

### 📂 Repository Structure

The repository is organized into two primary modules, each containing its own specialized data pipeline, model weights, and performance reports.

| Directory | Task | Model(s) Used | Key Metrics |
| :--- | :--- | :--- | :--- |
| `Common_Test_I/` | Multi-Class Substructure Classification | ResNet-18 | Avg. AUC > 0.98 |
| `Specific_Test_V/` | Binary Lens Finding & Data Pipelines | EfficientNet-B0, ResNet-18 | Test AUC: 0.9816 |

### 🚀 Task Summaries

#### 1\. Common Test I: Multi-Class Classification

Objective: Classify lensing images into three categories: No Substructure (no), Subhalo (sphere), and Vortex (vort).

•	**Architecture**: ResNet-18.

•	**Result**: Achieved 100% accuracy on the "no substructure" class and high discriminatory power for complex morphologies (AUC ≈ 0.99).

•	**Focus**: Precision in distinguishing subtle differences between dark matter substructures.

#### 2\. Specific Test V: Gravitational Lens Finding

**Objective**: Identify the presence of a strong gravitational lens in a highly imbalanced dataset.

•	**Architecture**: Comparative analysis between EfficientNet-B0 and ResNet-18.

•	**Imbalance Strategy**: Dual-pronged approach using WeightedRandomSampler for batch distribution and pos\_weight within the loss function.

•	**Evaluation**: Optimized for the low-False Positive regime (FPR ≈ 0.01) to suit real-world astronomical survey requirements.

