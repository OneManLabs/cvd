from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from ..services.training import training_manager
from ..services.device import get_best_device

router = APIRouter(prefix="/api/training", tags=["training"])
MAX_DATASET_BYTES = 2 * 1024 * 1024 * 1024


class DatasetResponse(BaseModel):
    id: str
    name: str
    path: str
    num_classes: int
    class_names: list[str]
    train_images: int
    val_images: int
    task: str


class TrainingMetricsResponse(BaseModel):
    epoch: int
    total_epochs: int
    box_loss: float
    cls_loss: float
    dfl_loss: float
    precision: float
    recall: float
    mAP50: float
    mAP50_95: float
    lr: float


class TrainingJobResponse(BaseModel):
    id: str
    dataset_id: str
    base_model: str
    status: str
    progress: float
    metrics: TrainingMetricsResponse
    error: str | None
    output_model_path: str | None
    config: dict


class StartTrainingRequest(BaseModel):
    dataset_id: str
    base_model: str
    epochs: int = Field(default=100, ge=1, le=10000)
    batch_size: int = Field(default=16, ge=1, le=1024)
    img_size: int = Field(default=640, ge=32, le=8192)
    lr0: float = Field(default=0.01, gt=0, le=1)
    device: str | None = None
    project_name: str | None = None


# --- Datasets ---

@router.get("/datasets", response_model=list[DatasetResponse])
async def list_datasets():
    datasets = training_manager.list_datasets()
    return [
        DatasetResponse(
            id=d.id, name=d.name, path=d.path,
            num_classes=d.num_classes, class_names=d.class_names,
            train_images=d.train_images, val_images=d.val_images, task=d.task,
        )
        for d in datasets
    ]


@router.post("/datasets/upload", response_model=DatasetResponse)
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".zip"):
        raise HTTPException(400, "Please upload a .zip file")

    content = await file.read(MAX_DATASET_BYTES + 1)
    if len(content) > MAX_DATASET_BYTES:
        raise HTTPException(413, "The dataset archive is larger than 2 GB")
    try:
        info = training_manager.upload_dataset(file.filename, content)
    except ValueError as e:
        raise HTTPException(400, str(e))

    return DatasetResponse(
        id=info.id, name=info.name, path=info.path,
        num_classes=info.num_classes, class_names=info.class_names,
        train_images=info.train_images, val_images=info.val_images, task=info.task,
    )


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    training_manager.delete_dataset(dataset_id)
    return {"status": "ok"}


# --- Training Jobs ---

@router.post("/start", response_model=TrainingJobResponse)
async def start_training(req: StartTrainingRequest):
    device = req.device or get_best_device()
    try:
        job = training_manager.start_training(
            dataset_id=req.dataset_id,
            base_model=req.base_model,
            epochs=req.epochs,
            batch_size=req.batch_size,
            img_size=req.img_size,
            lr0=req.lr0,
            device=device,
            project_name=req.project_name,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    return _job_response(job)


@router.get("/jobs", response_model=list[TrainingJobResponse])
async def list_jobs():
    return [_job_response(j) for j in training_manager.list_jobs()]


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_job(job_id: str):
    job = training_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return _job_response(job)


@router.post("/jobs/{job_id}/stop")
async def stop_job(job_id: str):
    training_manager.stop_training(job_id)
    return {"status": "ok"}


def _job_response(job) -> TrainingJobResponse:
    m = job.metrics
    return TrainingJobResponse(
        id=job.id,
        dataset_id=job.dataset_id,
        base_model=job.base_model,
        status=job.status,
        progress=job.progress,
        metrics=TrainingMetricsResponse(
            epoch=m.epoch, total_epochs=m.total_epochs,
            box_loss=m.box_loss, cls_loss=m.cls_loss, dfl_loss=m.dfl_loss,
            precision=m.precision, recall=m.recall,
            mAP50=m.mAP50, mAP50_95=m.mAP50_95, lr=m.lr,
        ),
        error=job.error,
        output_model_path=job.output_model_path,
        config=job.config,
    )
