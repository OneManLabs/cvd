import io
import json
import base64
import asyncio
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from PIL import Image
from ..services.model_manager import model_manager
from ..services.inference import run_inference
from ..services.device import get_best_device
from ..schemas import InferenceSettings

router = APIRouter()
MAX_FRAME_BYTES = 25 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 100_000_000


@router.websocket("/ws/inference")
async def websocket_inference(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            model_ids = msg.get("model_ids", [])
            frame_b64 = msg.get("frame", "")
            settings_data = msg.get("settings", {})

            if not model_ids or not frame_b64:
                await websocket.send_json({"error": "Missing model_ids or frame"})
                continue

            # Decode base64 frame
            if len(frame_b64) > ((MAX_FRAME_BYTES * 4) // 3) + 16:
                await websocket.send_json({"error": "The frame is larger than 25 MB"})
                continue
            try:
                frame_bytes = base64.b64decode(frame_b64, validate=True)
                image = Image.open(io.BytesIO(frame_bytes))
                image.load()
                image = image.convert("RGB")
            except Exception:
                await websocket.send_json({"error": "The frame is not a supported image"})
                continue

            settings = InferenceSettings(**settings_data)

            # Run inference on all requested models
            results = []
            for mid in model_ids:
                model = model_manager.get_loaded_model(mid)
                if not model:
                    try:
                        model = model_manager.load_model(mid, device=get_best_device())
                    except Exception:
                        continue

                result = run_inference(model, image, mid, settings)
                results.append(result.model_dump())

            await websocket.send_json({"results": results})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
