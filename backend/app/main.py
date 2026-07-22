import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import models, inference, ws, training
from .services.device import get_available_devices, get_best_device, get_device_display_name
from .services.model_manager import model_manager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        model_manager.load_model("custom-yolo11n", device=get_best_device())
    except Exception as error:
        logger.warning("The included ONNX model did not load: %s", error)
    yield


app = FastAPI(title="CVD - Computer Vision Dashboard", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models.router)
app.include_router(inference.router)
app.include_router(ws.router)
app.include_router(training.router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/devices")
async def devices():
    available = get_available_devices()
    best = get_best_device()
    return {
        "available": [
            {"id": d, "name": get_device_display_name(d)} for d in available
        ],
        "recommended": best,
    }
