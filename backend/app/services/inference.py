import time
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from ..schemas import Detection, InferenceResult, InferenceSettings

# Pose models crash on MPS with large tensors — track which need CPU fallback
_MPS_INCOMPATIBLE_TASKS = {"pose"}


def run_inference(
    model: YOLO,
    image: np.ndarray | Image.Image,
    model_id: str,
    settings: InferenceSettings | None = None,
) -> InferenceResult:
    if settings is None:
        settings = InferenceSettings()

    if isinstance(image, Image.Image):
        img_width, img_height = image.size
    else:
        img_height, img_width = image.shape[:2]

    # Fall back to CPU for tasks that crash on MPS
    device_override = None
    task = getattr(model, "task", "")
    if task in _MPS_INCOMPATIBLE_TASKS and str(model.device).startswith("mps"):
        device_override = "cpu"
        model.to("cpu")

    start = time.perf_counter()
    results = model.predict(
        source=image,
        conf=settings.confidence,
        iou=settings.iou,
        max_det=settings.max_det,
        imgsz=settings.img_size,
        device=device_override or None,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    detections = []
    if results and len(results) > 0:
        result = results[0]
        names = result.names

        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                x1, y1, x2, y2 = box.xyxyn[0].tolist()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                det = Detection(
                    class_id=cls_id,
                    class_name=names.get(cls_id, str(cls_id)),
                    confidence=conf,
                    bbox=[x1, y1, x2, y2],
                )

                # Segmentation masks
                if result.masks is not None and i < len(result.masks):
                    mask_xy = result.masks[i].xyn[0].tolist()
                    det.mask = mask_xy

                # Pose keypoints
                if result.keypoints is not None and i < len(result.keypoints):
                    kpts = result.keypoints[i].xyn[0].tolist()
                    det.keypoints = kpts

                detections.append(det)

    return InferenceResult(
        model_id=model_id,
        detections=detections,
        inference_time_ms=round(elapsed_ms, 1),
        image_width=img_width,
        image_height=img_height,
    )
