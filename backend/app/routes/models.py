from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File
from ..services.model_manager import model_manager
from ..services.device import get_best_device
from ..schemas import ModelInfo, ModelListResponse

router = APIRouter(prefix="/api/models", tags=["models"])
MAX_MODEL_BYTES = 100 * 1024 * 1024


@router.get("", response_model=ModelListResponse)
async def list_models():
    return ModelListResponse(models=model_manager.list_models())


@router.post("/{model_id}/load", response_model=ModelInfo)
async def load_model(model_id: str, device: str | None = None):
    try:
        dev = device or get_best_device()
        model_manager.load_model(model_id, device=dev)
        models = model_manager.list_models()
        info = next((m for m in models if m.id == model_id), None)
        if not info:
            raise HTTPException(404, "Model not found")
        return info
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to load model: {e}")


@router.post("/{model_id}/unload")
async def unload_model(model_id: str):
    model_manager.unload_model(model_id)
    return {"status": "ok"}


@router.post("/upload", response_model=ModelInfo)
async def upload_model(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No filename provided")
    safe_name = Path(file.filename).name
    if Path(safe_name).suffix.lower() not in {".pt", ".onnx"}:
        raise HTTPException(400, "Only .pt and .onnx files are supported")

    content = await file.read(MAX_MODEL_BYTES + 1)
    if len(content) > MAX_MODEL_BYTES:
        raise HTTPException(413, "The model file is larger than 100 MB")
    info = model_manager.add_custom_model(safe_name, content)
    return info
