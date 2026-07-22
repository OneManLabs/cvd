import io
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from PIL import Image
from ..services.model_manager import model_manager
from ..services.inference import run_inference
from ..services.device import get_best_device
from ..schemas import InferenceResult, InferenceSettings

router = APIRouter(prefix="/api/inference", tags=["inference"])
MAX_IMAGE_BYTES = 25 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 100_000_000


@router.post("/{model_id}", response_model=InferenceResult)
async def infer_image(
    model_id: str,
    file: UploadFile = File(...),
    confidence: float = Form(0.25),
    iou: float = Form(0.45),
    img_size: int = Form(640),
):
    model = model_manager.get_loaded_model(model_id)
    if not model:
        # Auto-load with best device
        try:
            model = model_manager.load_model(model_id, device=get_best_device())
        except ValueError:
            raise HTTPException(404, f"Model {model_id} not found")

    content = await file.read(MAX_IMAGE_BYTES + 1)
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "The image is larger than 25 MB")
    try:
        image = Image.open(io.BytesIO(content))
        image.load()
        image = image.convert("RGB")
    except Exception as error:
        raise HTTPException(400, "The file is not a supported image") from error

    settings = InferenceSettings(
        confidence=confidence,
        iou=iou,
        img_size=img_size,
    )

    result = run_inference(model, image, model_id, settings)
    return result
