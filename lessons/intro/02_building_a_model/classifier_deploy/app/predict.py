import io
import time

import torch
import torchvision.transforms as T
from PIL import Image
from fastapi import APIRouter, File, UploadFile, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Prediction

router = APIRouter()

# Model is set by main.py at startup
learn = None

_transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@router.post("/predict")
def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    start = time.perf_counter()

    image_bytes = file.file.read()
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _transform(pil_img).unsqueeze(0)

    with torch.no_grad():
        out = learn.model(tensor)
        probs = out.softmax(dim=-1)[0]

    idx = probs.argmax().item()
    pred = learn.dls.vocab[idx]

    processing_time_ms = int((time.perf_counter() - start) * 1000)

    probabilities = {
        learn.dls.vocab[i]: round(float(p), 4)
        for i, p in enumerate(probs)
    }

    # Log to database
    db_prediction = Prediction(
        prediction=str(pred),
        confidence=float(probs[idx]),
        probabilities=probabilities,
        processing_time_ms=processing_time_ms,
    )
    db.add(db_prediction)
    db.commit()

    return {
        "prediction": str(pred),
        "confidence": round(float(probs[idx]), 4),
        "probabilities": probabilities,
        "processing_time_ms": processing_time_ms,
    }
