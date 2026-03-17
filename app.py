from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import io
import gradio as gr

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]


model = models.densenet161(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, 14)
model.load_state_dict(torch.load("model_weights.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def run_inference(image: Image.Image):
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.sigmoid(outputs)[0]  # BCEWithLogitsLoss needs sigmoid

    results = [
        {"label": CLASS_LABELS[i], "confidence": round(float(probs[i]), 4)}
        for i in range(14)
    ]
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results

@app.get("/")
def root():
    return RedirectResponse(url="/ui/")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    results = run_inference(image)

   
    detected = [r for r in results if r["confidence"] > 0.5]

    return {
        "detected_diseases": detected,
        "all_classes": results
    }

def gradio_predict(image):
    image = Image.fromarray(image).convert("RGB")
    results = run_inference(image)
    return {r["label"]: r["confidence"] for r in results}

gradio_app = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=14),
    title="MedicalGPT — Chest X-Ray Disease Classifier",
    description="Upload a chest X-ray to detect up to 14 diseases. Part of the MedicalGPT family.",
)

gr.mount_gradio_app(app, gradio_app, path="/ui")
