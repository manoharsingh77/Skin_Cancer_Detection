# 🩺 Skin Cancer Detection using Deep Learning (EfficientNetB0)

This project implements a **Convolutional Neural Network (CNN)** using **EfficientNetB0** for **skin cancer classification** on the **HAM10000** dataset.  
It leverages **transfer learning**, **Focal Loss**, and **fine-tuning** to improve detection of rare skin cancer types.

---

## 🚀 Features

- ✅ Transfer Learning using **EfficientNetB0 (ImageNet pretrained)**
- ✅ Handles **class imbalance** using **Focal Loss** and **class weighting**
- ✅ **Fine-tuned last 80 layers** for better generalization
- ✅ **Data augmentation** for improved robustness
- ✅ Model evaluation using **Confusion Matrix**, **F1-score**, **Recall**, and **Precision**
- ✅ Supports **Google Colab** training with Drive checkpoint saving
- ✅ Optional **Grad-CAM visualization** for model interpretability

---

## 🧠 Dataset — HAM10000

The **HAM10000** dataset (“Human Against Machine with 10,000 training images”) contains 7 classes of dermoscopic images:

| Label | Meaning |
|--------|----------|
| akiec | Actinic keratoses |
| bcc | Basal cell carcinoma |
| bkl | Benign keratosis |
| df | Dermatofibroma |
| mel | Melanoma |
| nv | Melanocytic nevi |
| vasc | Vascular lesions |

**Total Images:** ~10,015  
**Classes:** 7  

📦 Download from [Kaggle - HAM10000 Dataset](https://www.kaggle.com/kmader/ham10000)

---

## 🧩 Model Architecture

- **Base Model:** EfficientNetB0 (pretrained on ImageNet)
- **Classifier Head:**
  - GlobalAveragePooling2D  
  - Dropout (0.4)  
  - Dense(7, activation='softmax')

**Loss Function:** Focal Loss (γ = 2.0, α = 0.25)  
**Optimizer:** Adam (lr = 1e-4 → 1e-5 during fine-tuning)  
**Metrics:** Accuracy, Precision, Recall, F1-score  

---

## ⚙️ Project Structure


📂 skin-cancer-detection/
- ├── data/
- │ ├── train/
- │ └── val/
- ├── models/
- │ └── efficientnetb0_skin_cancer.h5
- ├── notebooks/
- │ └── skin_cancer_training.ipynb
- ├── utils/
- │ └── gradcam.py
- ├── README.md
- └── requirements.txt


---

## 🧾 Requirements

- tensorflow>=2.9
- tensorflow-addons
- numpy
- pandas
- opencv-python
- matplotlib
- seaborn
- scikit-learn
- efficientnet

---

## 📈 Results


- | Metric                  | Score |
- | ----------------------- | ----- |
- | **Validation Accuracy** | 58%   |
- | **Weighted F1-score**   | 0.63  |
- | **Macro Recall**        | 0.62  |
- | **Macro Precision**     | 0.44  |

---

## 📊 Sample Confusion Matrix

```bash
              precision    recall  f1-score   support
       akiec       0.49      0.33      0.40
         bcc       0.59      0.52      0.56
         bkl       0.37      0.64      0.47
          df       0.20      0.78      0.32
         mel       0.26      0.49      0.34
          nv       0.96      0.59      0.73
        vasc       0.20      0.97      0.34

