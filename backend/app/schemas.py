from pydantic import BaseModel, Field
from typing import Optional


class DeviceInfo(BaseModel):
    available_devices: list[str]
    current_device: str


class ModelInfo(BaseModel):
    id: str
    name: str
    task: str  # detect, segment, pose, classify, obb
    size: Optional[str] = None  # n, s, m, l, x
    path: str
    is_custom: bool = False
    loaded: bool = False


class ModelListResponse(BaseModel):
    models: list[ModelInfo]


class InferenceSettings(BaseModel):
    confidence: float = Field(default=0.25, ge=0, le=1)
    iou: float = Field(default=0.45, ge=0, le=1)
    device: Optional[str] = None
    max_det: int = Field(default=300, ge=1, le=10000)
    img_size: int = Field(default=640, ge=32, le=8192)


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]  # x1, y1, x2, y2 normalized
    mask: Optional[list[list[float]]] = None  # polygon points for segmentation
    keypoints: Optional[list[list[float]]] = None  # pose keypoints


class InferenceResult(BaseModel):
    model_id: str
    detections: list[Detection]
    inference_time_ms: float
    image_width: int
    image_height: int
