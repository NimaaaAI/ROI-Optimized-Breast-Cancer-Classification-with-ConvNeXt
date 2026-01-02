# BreastVision: High-Sensitivity Cancer Detection Pipeline

![Domain](https://img.shields.io/badge/Domain-Medical%20Imaging-red) 
![Model](https://img.shields.io/badge/Architecture-ConvNeXt--Tiny-blue)
![Platform](https://img.shields.io/badge/Platform-Kaggle%20T4%20GPU-orange)
![License](https://img.shields.io/badge/License-Apache%202.0-green)

## 📌 Project Overview
This project addresses the challenge of automated breast cancer classification using ultrasound imagery. Unlike standard classification tasks, medical diagnostics require extreme **sensitivity (Recall)** to ensure no malignant cases are missed. This pipeline utilizes a modern **ConvNeXt** architecture combined with custom **Region of Interest (ROI)** extraction to focus on clinical features rather than background noise.

---

## ⚙️ The Diagnostic Pipeline

Our workflow is divided into four distinct phases, mimicking the process a radiologist follows to evaluate a scan:

### Phase 1: Data Preparation & Preprocessing
The raw dataset is organized into clinical splits (Train, Validation, Test). We normalize the imagery to ensure consistent pixel intensity across different ultrasound machine outputs.

### Phase 2: ROI Extraction (The "Cleaner")
Ultrasound scans often contain "noise" like pectoral muscle, text annotations, and rib shadows. 
* **Technique:** We use a U-Net inspired logic to find the tissue boundaries.
* **Impact:** By cropping the image to the specific area of interest, we prevent the model from learning "shortcuts" based on background artifacts.



### Phase 3: Augmentation Techniques & Balancing
Medical data is often imbalanced, with far more Benign cases than Malignant.
* **Strategic Balancing:** We apply advanced augmentations (rotations, flips, and color jitter) specifically to Malignant samples.
* **Impact:** This doubles the minority class samples, providing a 1:1 balance that prevents model bias and improves feature recognition.

### Phase 4: ConvNeXt-Tiny Architecture
We utilize the **ConvNeXt-Tiny** model, a state-of-the-art convolutional neural network that adopts the best features of Vision Transformers (like large kernels and AdamW optimization).
* **Speed:** Optimized with **Mixed Precision (FP16)** for fast training on T4 GPUs.
* **Metric:** We prioritize the **F1-Score** over accuracy to ensure a balanced performance between precision and recall.



---

## 🏥 Clinical Optimization (The "Safety" Step)
In a hospital setting, a **"False Negative"** (missing cancer) is much more dangerous than a **"False Positive"** (a benign case flagged for further review). 

We implemented **Custom Threshold Tuning**:
1. **Default Threshold:** 0.5 (Standard AI)
2. **Optimized Threshold:** **0.3** (Clinical Safety)
3. **Result:** By lowering the threshold to 0.3, we significantly boosted our **Malignant Recall**, ensuring the AI acts as a high-sensitivity screening layer.



---

## 📊 Evaluation & Visual Audit
The project concludes with a full visual audit of the unseen test set. We categorize results into three clear clinical buckets:
* ✅ **Correct:** Accurate diagnosis.
* ❌ **Missed Cancer:** High-risk errors (minimized by our 0.3 threshold).
* ⚠️ **False Alarm:** Benign cases flagged for review.

---

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Frameworks:** PyTorch, Hugging Face `transformers`, `datasets`
* **Architecture:** ConvNeXt-Tiny
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Kaggle T4 GPU

---

## 📜 License
This project is licensed under the **Apache License 2.0**.
