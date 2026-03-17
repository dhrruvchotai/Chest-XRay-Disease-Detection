---
title: Chest X-Ray Disease Classifier
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: true
license: mit
tags:
  - medical
  - radiology
  - chest-xray
  - densenet
  - multi-label-classification
  - medicalgpt
---

<div align="center">

# 🫁 Chest X-Ray Disease Classifier
### Part of the **MedicalGPT** Family of Models

![MedicalGPT](https://img.shields.io/badge/MedicalGPT-Family-red?style=for-the-badge&logoColor=white)
![Model](https://img.shields.io/badge/Model-DenseNet161-blue?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-NIH%20Chest%20X--Ray-green?style=for-the-badge)
![Classes](https://img.shields.io/badge/Classes-14-orange?style=for-the-badge)
![Framework](https://img.shields.io/badge/Framework-PyTorch-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)

*AI-powered radiological analysis for multi-disease chest X-ray detection*

---

</div>

## 🏥 About MedicalGPT

This model is part of the **MedicalGPT** family — a suite of specialized AI models designed to assist in clinical and diagnostic workflows. MedicalGPT models are built with the goal of making medical-grade AI accessible, interpretable, and deployable in real-world healthcare applications.

> ⚠️ **Disclaimer:** This model is intended for **research and assistive purposes only**. It is not a substitute for professional medical diagnosis. Always consult a qualified radiologist or physician for clinical decisions.

---

## 🔬 Model Overview

| Property | Details |
|---|---|
| **Architecture** | DenseNet161 (fine-tuned) |
| **Task** | Multi-label Chest X-Ray Disease Classification |
| **Dataset** | NIH Chest X-Ray Dataset |
| **Classes** | 14 diseases |
| **Input Size** | 224 × 224 px |
| **Framework** | PyTorch 2.x |
| **Test Accuracy** | 0.9490 (94.90%) |
| **Model Family** | MedicalGPT |

---

## 🏷️ Detectable Diseases

| Index | Disease |
|---|---|
| 0 | Atelectasis |
| 1 | Cardiomegaly |
| 2 | Consolidation |
| 3 | Edema |
| 4 | Effusion |
| 5 | Emphysema |
| 6 | Fibrosis |
| 7 | Hernia |
| 8 | Infiltration |
| 9 | Mass |
| 10 | Nodule |
| 11 | Pleural Thickening |
| 12 | Pneumonia |
| 13 | Pneumothorax |

> This is a **multi-label** model — a single X-ray can be flagged for multiple diseases simultaneously.

---

## ⚙️ Model Architecture & Training

- **Base Model:** DenseNet161 pretrained on ImageNet
- **Fine-tuning Strategy:** Classifier layer replaced and trained — feature extractor frozen
- **Loss Function:** BCEWithLogitsLoss (binary cross-entropy for multi-label)
- **Optimizer:** Adam (lr: 1e-4)
- **Epochs:** 5
- **Batch Size:** 32
- **Training Samples:** 40,000 (80% of 50,000 sampled from NIH dataset)
- **Test Samples:** 10,000 (20%)

### Preprocessing Pipeline
```
Resize (224×224) → ToTensor
```
> No normalization applied — matches training pipeline exactly.

### Inference
Since BCEWithLogitsLoss was used during training, **sigmoid activation** is applied at inference time. Any disease with confidence > 0.5 is considered detected.

---

## 🚀 API Usage

### Endpoint
```
POST /predict
```

### Python
```python
import requests

url = "https://dhrruvchotai-densenet161-chest-xray-classifier.hf.space/predict"

with open("chest_xray.png", "rb") as f:
    response = requests.post(url, files={"file": f})

print(response.json())
```

### JavaScript / Frontend
```javascript
const formData = new FormData();
formData.append("file", imageFile);

const response = await fetch(
  "https://dhrruvchotai-densenet161-chest-xray-classifier.hf.space/predict",
  { method: "POST", body: formData }
);

const result = await response.json();
console.log(result);
```

### Sample Response
```json
{
  "detected_diseases": [
    { "label": "Effusion",      "confidence": 0.7821 },
    { "label": "Infiltration",  "confidence": 0.6134 }
  ],
  "all_classes": [
    { "label": "Effusion",           "confidence": 0.7821 },
    { "label": "Infiltration",       "confidence": 0.6134 },
    { "label": "Atelectasis",        "confidence": 0.3201 },
    { "label": "Consolidation",      "confidence": 0.2874 },
    { "label": "Pneumonia",          "confidence": 0.1923 },
    { "label": "Edema",              "confidence": 0.1102 },
    { "label": "Cardiomegaly",       "confidence": 0.0891 },
    { "label": "Nodule",             "confidence": 0.0654 },
    { "label": "Mass",               "confidence": 0.0521 },
    { "label": "Pneumothorax",       "confidence": 0.0413 },
    { "label": "Pleural_Thickening", "confidence": 0.0312 },
    { "label": "Emphysema",          "confidence": 0.0201 },
    { "label": "Fibrosis",           "confidence": 0.0134 },
    { "label": "Hernia",             "confidence": 0.0043 }
  ]
}
```

> `detected_diseases` contains only diseases with confidence above the **0.5 threshold**. `all_classes` contains scores for all 14 diseases sorted by confidence.

---

## 📊 Performance

| Metric | Score |
|---|---|
| **Test Accuracy (Macro, MultilabelAccuracy)** | **0.9490 (94.90%)** |
| **Test Loss** | **0.1670** |

---

## 🧪 Dataset

- **Source:** [NIH Chest X-Ray Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data)
- **Total Images in Dataset:** ~112,000 X-ray images
- **Sampled for Training:** 50,000 images
- **Split:** 80% training (40,000) / 20% test (10,000)
- **Labels:** Multi-label — each image can have one or more of 14 disease labels
- **Note:** "No Finding" label excluded from classification output

---

## 🛠️ Deployment

This model is deployed using **Docker** on Hugging Face Spaces with a **FastAPI** backend and **Gradio** UI.

```
Docker → FastAPI → DenseNet161 → Sigmoid → REST API
                       ↓
                   Gradio UI (/ui)
```

---

## 🧬 MedicalGPT Model Family

This is the **second model** in the growing **MedicalGPT** ecosystem. The vision is to build a fully integrated AI platform where all specialist models work together, unified by a central Medical Chatbot that patients and doctors can interact with directly.

| Model | Domain | Status |
|---|---|---|
| Skin Lesion Classifier | Dermatology | ✅ Live |
| Chest X-Ray Classifier | Radiology | ✅ Live |
| MedicalGPT Chatbot | General Medical Queries & Patient Support | 🔜 Coming Soon |
| Medical OCR | Medicine & Report Reader | 🔜 Coming Soon |
| Retinal Scan Classifier | Ophthalmology | 🔜 Coming Soon |
| Pathology Slide Analyzer | Pathology | 🔜 Coming Soon |

### MedicalGPT Chatbot (Coming Soon)
A conversational AI assistant serving as the **central hub** of the MedicalGPT platform. Patients can describe symptoms, ask medical questions, and get guidance — while the chatbot intelligently routes to the right specialist model when needed. All diagnostic results feed back into the chatbot for a unified patient experience.

### Medical OCR (Coming Soon)
An OCR-powered module built specifically for healthcare documents — capable of reading prescription slips, medicine labels, lab reports, and discharge summaries, making it easy for patients to understand their medical documents without needing a doctor present.

---

## 📄 License

This project is licensed under the **MIT License**.

---

<div align="center">

Built with ❤️ as part of the **MedicalGPT** initiative

*Advancing AI for Healthcare*

</div>
