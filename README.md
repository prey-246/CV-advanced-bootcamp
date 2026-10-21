# Multilingual/Multimodal Model Evaluation Capstone

This project evaluates and compares a traditional, custom-trained CNN architecture (U-Net) against a state-of-the-art Zero-Shot Foundation Model (Segment Anything Model - SAM) on the Oxford-IIIT Pet dataset.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Environment Setup & Dependencies](#environment-setup--dependencies)
3. [Dataset Selection](#dataset-selection)
4. [Model Architectures](#model-architectures)
   - [U-Net (Traditional)](#u-net-traditional)
   - [SAM (Foundation)](#sam-foundation)
5. [Performance & Quantitative Comparison](#performance--quantitative-comparison)
6. [Qualitative Analysis & Findings](#qualitative-analysis--findings)
7. [Conclusion & Use Cases](#conclusion--use-cases)

---

## Project Overview
This capstone project focuses on analyzing the trade-offs between **domain-specific deep learning models** and **large-scale zero-shot foundation models** for semantic segmentation tasks.

## Environment Setup & Dependencies
The following frameworks and libraries are required to run this evaluation:
- Python 3.x
- TensorFlow & TensorFlow Datasets (TFDS)
- PyTorch
- Segment Anything (SAM) by Meta AI
- OpenCV, Matplotlib, NumPy, Scikit-Learn, Seaborn

## Dataset Selection
- **Dataset**: Oxford-IIIT Pet Dataset (`oxford_iiit_pet:4.0.0` via TFDS)
- **Task**: Pixel-level binary segmentation (Background vs. Pet boundaries)
- **Input Dimensions**: Images resized and normalized to $128 \times 128$ resolution.

## Model Architectures

### U-Net (Traditional)
- **Type**: Custom CNN-based encoder-decoder model.
- **Training**: Trained directly on the target dataset for 5 epochs.
- **Parameters**: Structured with MaxPooling and UpSampling2D layers designed specifically for structured pixel classification.

### SAM (Foundation)
- **Type**: Segment Anything Model (ViT-H backbone).
- **Training**: None (Zero-shot inference using point grid prompts).

## Performance & Quantitative Comparison

Below is the comparative breakdown of both models evaluated on the test set:

| Metric | Traditional (U-Net) | Foundation (SAM) |
| :--- | :--- | :--- |
| **Training Required** | Yes (Custom) | No (Zero-shot) |
| **Inference Speed** | **High** (~0.0119s/img) | **Low** (~2.2592s/img) |
| **Accuracy (Avg IoU)** | **0.7513** | **0.0484** (with basic prompts) |
| **Generalization** | Specific to Pets | Open-domain |
| **Best Use Case** | Real-time / Production | Research / Labeling Assistance |

## Qualitative Analysis & Findings
1. **U-Net** yields highly localized segmentation masks matching the exact pet boundaries because it learned the distribution specifically during supervised training.
2. **SAM** acts as an open-domain generalist. Its zero-shot performance on localized prompts is lower out-of-the-box because it lacks fine-tuning on the target class structure, but generalizes immediately to complex environments without weights adjustment.

## Conclusion & Use Cases
- **Deploy U-Net** when working under latency constraints (e.g., edge or mobile devices) and when labeled training data is readily available.
- **Deploy SAM** when you need automated labeling assistance, have zero training data, or need to segment arbitrary objects in novel, unseen domains.
