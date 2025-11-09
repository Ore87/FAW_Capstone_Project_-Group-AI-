[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ore87/FAW_Capstone_Project_-Group-AI-/blob/main/FAW_(Group_AI).ipynb)

# Early Detection of Fall Armyworm (FAW) with YOLOv11s

<img src="results/Project_Banner.png" alt="Project Banner" width="700"/>

This repository contains the capstone project for the AI Bootcamp, focusing on the detection of the Fall Armyworm (FAW) using a YOLOv11s object detection model. The model is trained to identify FAW in images, providing a crucial tool for early pest detection to protect crops and ensure food security.

---

## 🌾 The Problem: A Global Threat

The Fall Armyworm (*Spodoptera frugiperda*) is a highly invasive pest that has spread globally from the Americas. It threatens food security worldwide, causing devastating crop loss, particularly in maize.
* **Unstoppable Spread:** FAW cannot be eradicated; an adult moth can fly up to 100km per night.
* **Massive Crop Loss:** It is responsible for an estimated **17.7 million tonnes** of annual maize loss in Africa alone (a 31% average loss).
* **Critical Need:** Early and accurate detection is the only effective way to manage the pest and protect the livelihoods of smallholder farmers.

## 🎯 Project Aim

The goal of this project was to:
1.  **Design and train** a high-performance, supervised AI model for the early and accurate detection of FAW from visual data.
2.  **Produce a compact, deployable, and performant AI model** in the **ONNX format** to support sustainable pest management (IPM) in real-world applications.

---

## 🛠️ Project Pipeline

The success of this project was not just in training a model, but in a rigorous data engineering pipeline.

### 1. Data Acquisition
We began with a small, 183-image dataset. To build a robust model, we sourced additional public FAW datasets from Roboflow, expanding our dataset to a final, combined set of **13,111 images** (11,222 training, 1,889 validation).

### 2. The Data Problem: "The 23-Label Mess"
After gathering the data, we discovered a critical problem: the annotation files (`.xml`) were a mess, containing **23 different and confusing labels**.
* **Junk Labels:** Included incorrect pests like 'Cabbage-worm' or 'Cutworm' and gibberish like '-', '1', '2'.
* **Redundant Labels:** 18 different labels (e.g., 'arm Worm', 'pest-armyworm', 'Plaga') were used for the *exact same thing*: the Fall Armyworm.

### 3. The Solution: The 1-Class Strategy
We made the key decision to simplify the problem. Our custom Python script performed three jobs:
1.  **Filter & Merge:** It discarded all 5 "junk" labels.
2.  **Re-Label:** It merged all 18 "armyworm-like" labels into a single class: **`ID 0: FallArmyworm`**.
3.  **Format:** It converted the bounding box coordinates from pixel format to the percentage format required by YOLO.

### 4. Training the Model
* **Model:** `YOLOv11s` (the "small" model) was chosen for its optimal balance of high accuracy and fast performance.
* **Technique:** We used **Transfer Learning** on a model pre-trained on 80 classes, fine-tuning it to find only our `FallArmyworm` class.
* **Training:** The model was trained for **50 epochs** with a batch size of 16 in a Google Colab environment, using a Tesla T4 GPU.

---

## 📊 Final Performance & Results

Our 1-Class strategy and choice of the YOLOv11s model were highly successful.

| Metric | Score |
| :--- | :--- |
| **mAP50** (Lenient Grade) | **95.0%** |
| **mAP50-95** (Strict Grade) | **71.1%** |
| **Precision** (Low False Positives) | **91.0%** |
| **Recall** (Low Missed Detections) | **90.3%** |

### Training & Validation Curves
![Results Chart](results/results%20(2).png)
The learning curves show a perfect training run:
* **Loss (Confusion)** curves (top and bottom left) steadily decrease, showing the model learned.
* **Metrics (Scores)** curves (right) steadily increase and plateau, showing high performance.
* Validation and Training curves are closely aligned, proving the model **did not overfit**.

### Confusion Matrix
![Confusion Matrix](results/confusion_matrix%20(1).png)
The "Mistake Report" shows a well-balanced model:
* **2,160 True Positives:** Correct "bullseye" detections.
* **188 False Negatives:** "Missed" detections (explains 90.3% Recall).
* **244 False Positives:** "Ghost" detections (explains 91.0% Precision).

### Detailed Metric Curves
The training process also generates detailed breakdowns of key metrics at different confidence thresholds, which confirm the model's robustness.

* **F1-Confidence Curve (`BoxF1_curve.png`)**
    This graph shows the F1 score (the balance between precision and recall). Our model's F1 score peaks and stays high across a wide range of confidence thresholds, which is excellent.
    ![F1 Curve](results/BoxF1_curve.png)

* **Precision-Recall (PR) Curve (`BoxPR_curve.png`)**
    This plot shows the tradeoff between precision and recall. Our curve stays high in the top-right corner, which indicates high precision is maintained even as recall increases.
    ![PR Curve](results/BoxPR_curve.png)

* **Recall-Confidence Curve (`BoxR_curve.png`)**
    This shows how recall (the ability to find *all* true pests) changes as the confidence threshold is adjusted.
    ![Recall Curve](results/BoxR_curve.png)

---

## 🚀 How to Use

This project is fully contained in the Google Colab notebook.

1.  **Open in Colab:** Click the "Open in Colab" badge at the top of this page.
2.  **Run the Notebook:** The notebook is self-contained. It will:
    * Install all dependencies (`ultralytics`, etc.).
    * Clone the dataset (which is part of this repository).
    * Run the complete data-cleaning pipeline (Steps 1-3) to create the clean dataset.
    * Run the model training (Step 4).
    * Export the final `model.onnx` file.

### Key Files
* **`FAW_(Group_AI).ipynb`**: The main Google Colab notebook with all code, from data cleaning to training and export.
* **`final_model/model.onnx`**: The final, trained, and deployable model file (36.2 MB).
* **`train/` & `validation/`**: The raw dataset folders used by the notebook.
* **`results/`**: A folder containing all output graphs from the training run.

---

## 🚧 Challenges Faced

* **Model Incompatibility:** Our first attempt with an **SSD MobileNet** model failed as it was not computationally feasible in the free Colab environment, forcing a pivot to the more efficient YOLO framework.
* **Data Chaos:** The 23-label mess was the single biggest hurdle and required significant data engineering to solve.
* **GPU Constraints:** We exhausted the free GPU resources in Google Colab multiple times, requiring careful scheduling of training runs.

## 💡 Proposed Future Work

The `model.onnx` file is ready for deployment. The logical next steps are:
* **Telegram Chatbot:** Create a simple bot where farmers can send a photo and get an instant "FAW Detected" response.
* **Automated Scouting:** Integrate the model into a drone or mobile robot for automated, real-time pest scouting in large fields.
