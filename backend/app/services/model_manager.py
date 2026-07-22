import os
import time
from pathlib import Path
from ultralytics import YOLO
from ..schemas import ModelInfo

MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"

# Built-in model registry: (id, name, task, sizes)
BUILTIN_MODELS = [
    ("yolo26", "YOLO26", "detect", ["n", "s", "m", "l", "x"]),
    ("yolo26-seg", "YOLO26 Segmentation", "segment", ["n", "s", "m", "l", "x"]),
    ("yolo26-pose", "YOLO26 Pose", "pose", ["n", "s", "m", "l", "x"]),
    ("yolo26-cls", "YOLO26 Classification", "classify", ["n", "s", "m", "l", "x"]),
    ("yolo26-obb", "YOLO26 Oriented BBox", "obb", ["n", "s", "m", "l", "x"]),
    ("yolo11", "YOLO11", "detect", ["n", "s", "m", "l", "x"]),
    ("yolo11-seg", "YOLO11 Segmentation", "segment", ["n", "s", "m", "l", "x"]),
    ("yolo11-pose", "YOLO11 Pose", "pose", ["n", "s", "m", "l", "x"]),
    ("yolo11-cls", "YOLO11 Classification", "classify", ["n", "s", "m", "l", "x"]),
    ("yolo11-obb", "YOLO11 Oriented BBox", "obb", ["n", "s", "m", "l", "x"]),
    ("yolov8", "YOLOv8", "detect", ["n", "s", "m", "l", "x"]),
    ("yolov8-seg", "YOLOv8 Segmentation", "segment", ["n", "s", "m", "l", "x"]),
    ("yolov8-pose", "YOLOv8 Pose", "pose", ["n", "s", "m", "l", "x"]),
    ("yolov8-cls", "YOLOv8 Classification", "classify", ["n", "s", "m", "l", "x"]),
    ("yolov8-obb", "YOLOv8 Oriented BBox", "obb", ["n", "s", "m", "l", "x"]),
    ("yolov10", "YOLOv10", "detect", ["n", "s", "m", "b", "l", "x"]),
    ("yolov9", "YOLOv9", "detect", ["c", "e"]),
    ("yolov5", "YOLOv5", "detect", ["n", "s", "m", "l", "x"]),
]


def _model_filename(model_id: str, size: str) -> str:
    """Convert model id + size to ultralytics model filename."""
    # yolo11n.pt, yolov8n-seg.pt, etc.
    base = model_id.replace("yolo11", "yolo11").replace("-", "")
    # Handle task suffixes
    task_suffix = ""
    for task_tag in ["seg", "pose", "cls", "obb"]:
        if task_tag in model_id:
            base = model_id.split("-" + task_tag)[0].replace("-", "")
            task_suffix = f"-{task_tag}"
            break
    return f"{base}{size}{task_suffix}.pt"


class ModelManager:
    def __init__(self):
        self._loaded_models: dict[str, YOLO] = {}
        self._model_registry: dict[str, ModelInfo] = {}
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self._build_registry()

    def _build_registry(self):
        # Register built-in models
        for model_id, name, task, sizes in BUILTIN_MODELS:
            for size in sizes:
                full_id = f"{model_id}-{size}"
                filename = _model_filename(model_id, size)
                self._model_registry[full_id] = ModelInfo(
                    id=full_id,
                    name=f"{name} ({size.upper()})",
                    task=task,
                    size=size,
                    path=filename,
                    is_custom=False,
                    loaded=False,
                )

        # Scan for custom models in models/ directory
        for f in MODELS_DIR.glob("*.pt"):
            custom_id = f"custom-{f.stem}"
            if custom_id not in self._model_registry:
                self._model_registry[custom_id] = ModelInfo(
                    id=custom_id,
                    name=f.stem,
                    task="detect",  # default, will update on load
                    path=str(f),
                    is_custom=True,
                    loaded=False,
                )
        for f in MODELS_DIR.glob("*.onnx"):
            custom_id = f"custom-{f.stem}"
            if custom_id not in self._model_registry:
                display_name = "YOLO11 Nano · ONNX" if f.stem == "yolo11n" else f"{f.stem} · ONNX"
                self._model_registry[custom_id] = ModelInfo(
                    id=custom_id,
                    name=display_name,
                    task="detect",
                    size="n" if f.stem == "yolo11n" else None,
                    path=str(f),
                    is_custom=True,
                    loaded=False,
                )

    def list_models(self) -> list[ModelInfo]:
        for model_id, info in self._model_registry.items():
            info.loaded = model_id in self._loaded_models
        return list(self._model_registry.values())

    def load_model(self, model_id: str, device: str = "cpu") -> YOLO:
        if model_id in self._loaded_models:
            return self._loaded_models[model_id]

        info = self._model_registry.get(model_id)
        if not info:
            raise ValueError(f"Unknown model: {model_id}")

        if info.is_custom:
            model_path = info.path
        else:
            model_path = info.path  # ultralytics auto-downloads

        model = YOLO(model_path)
        # Exported runtimes such as ONNX select their provider during predict().
        # The .to() method is only valid for native PyTorch checkpoints.
        if Path(model_path).suffix.lower() == ".pt":
            model.to(device)
        self._loaded_models[model_id] = model

        # Update task from loaded model
        if hasattr(model, "task"):
            info.task = model.task

        return model

    def unload_model(self, model_id: str):
        if model_id in self._loaded_models:
            del self._loaded_models[model_id]

    def get_loaded_model(self, model_id: str) -> YOLO | None:
        return self._loaded_models.get(model_id)

    def add_custom_model(self, filename: str, file_bytes: bytes) -> ModelInfo:
        safe_name = Path(filename).name
        if Path(safe_name).suffix.lower() not in {".pt", ".onnx"}:
            raise ValueError("Unsupported model file type")
        filepath = MODELS_DIR / safe_name
        filepath.write_bytes(file_bytes)

        custom_id = f"custom-{Path(safe_name).stem}"
        info = ModelInfo(
            id=custom_id,
            name=Path(safe_name).stem,
            task="detect",
            path=str(filepath),
            is_custom=True,
            loaded=False,
        )
        self._model_registry[custom_id] = info
        return info


# Singleton
model_manager = ModelManager()
