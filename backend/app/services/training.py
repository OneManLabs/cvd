import os
import time
import shutil
import zipfile
import threading
import re
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from ultralytics import YOLO

DATASETS_DIR = Path(__file__).parent.parent.parent.parent / "datasets"
RUNS_DIR = Path(__file__).parent.parent.parent.parent / "runs"
MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"
MAX_ARCHIVE_FILES = 100_000
MAX_EXTRACTED_BYTES = 20 * 1024 * 1024 * 1024


@dataclass
class TrainingMetrics:
    epoch: int = 0
    total_epochs: int = 0
    box_loss: float = 0.0
    cls_loss: float = 0.0
    dfl_loss: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    mAP50: float = 0.0
    mAP50_95: float = 0.0
    lr: float = 0.0


@dataclass
class TrainingJob:
    id: str
    dataset_id: str
    base_model: str
    status: str = "pending"  # pending, running, completed, failed, stopped
    progress: float = 0.0
    metrics: TrainingMetrics = field(default_factory=TrainingMetrics)
    error: str | None = None
    output_model_path: str | None = None
    config: dict = field(default_factory=dict)


@dataclass
class DatasetInfo:
    id: str
    name: str
    path: str
    num_classes: int = 0
    class_names: list[str] = field(default_factory=list)
    train_images: int = 0
    val_images: int = 0
    task: str = "detect"


class TrainingManager:
    def __init__(self):
        self._jobs: dict[str, TrainingJob] = {}
        self._datasets: dict[str, DatasetInfo] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._stop_flags: dict[str, bool] = {}
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        self._scan_datasets()

    def _scan_datasets(self):
        """Scan datasets directory for existing datasets."""
        for d in DATASETS_DIR.iterdir():
            if d.is_dir():
                yaml_file = self._find_yaml(d)
                if yaml_file:
                    info = self._parse_dataset(d.name, d, yaml_file)
                    if info:
                        self._datasets[info.id] = info

    def _find_yaml(self, dataset_dir: Path) -> Path | None:
        """Find the data.yaml or *.yaml file in a dataset directory."""
        # Check common names first
        for name in ["data.yaml", "dataset.yaml", "data.yml"]:
            p = dataset_dir / name
            if p.exists():
                return p
        # Fall back to any yaml
        yamls = list(dataset_dir.glob("*.yaml")) + list(dataset_dir.glob("*.yml"))
        return yamls[0] if yamls else None

    def _parse_dataset(self, dataset_id: str, dataset_dir: Path, yaml_file: Path) -> DatasetInfo | None:
        """Parse a YOLO-format dataset yaml."""
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)

            names = data.get("names", [])
            if isinstance(names, dict):
                class_names = list(names.values())
                num_classes = len(names)
            else:
                class_names = names
                num_classes = len(names)

            # Count images
            train_images = 0
            val_images = 0
            train_path = data.get("train", "")
            val_path = data.get("val", "")

            # Resolve paths relative to dataset dir
            for base in [dataset_dir, yaml_file.parent]:
                tp = base / train_path
                if tp.exists():
                    train_images = len(list(tp.glob("*.[jJpP]*[gG]")))
                    break

            for base in [dataset_dir, yaml_file.parent]:
                vp = base / val_path
                if vp.exists():
                    val_images = len(list(vp.glob("*.[jJpP]*[gG]")))
                    break

            return DatasetInfo(
                id=dataset_id,
                name=dataset_id.replace("_", " ").replace("-", " ").title(),
                path=str(dataset_dir),
                num_classes=num_classes,
                class_names=class_names,
                train_images=train_images,
                val_images=val_images,
            )
        except Exception as e:
            print(f"Failed to parse dataset {dataset_id}: {e}")
            return None

    def upload_dataset(self, name: str, zip_bytes: bytes) -> DatasetInfo:
        """Extract a zip file containing a YOLO-format dataset."""
        stem = Path(name).name
        if stem.lower().endswith(".zip"):
            stem = stem[:-4]
        dataset_id = re.sub(r"[^a-z0-9_-]+", "_", stem.lower()).strip("_")
        if not dataset_id:
            raise ValueError("The dataset name is not valid.")
        dataset_dir = DATASETS_DIR / dataset_id

        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        dataset_dir.mkdir(parents=True)

        # Extract zip
        zip_path = dataset_dir / "upload.zip"
        zip_path.write_bytes(zip_bytes)

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                members = zf.infolist()
                if len(members) > MAX_ARCHIVE_FILES:
                    raise ValueError("The dataset archive has too many files.")
                if sum(member.file_size for member in members) > MAX_EXTRACTED_BYTES:
                    raise ValueError("The extracted dataset is larger than 20 GB.")
                root = dataset_dir.resolve()
                for member in members:
                    target = (dataset_dir / member.filename).resolve()
                    if target != root and root not in target.parents:
                        raise ValueError("The dataset archive contains an unsafe path.")
                zf.extractall(dataset_dir)
        except zipfile.BadZipFile as error:
            raise ValueError("The dataset archive is not a valid ZIP file.") from error
        zip_path.unlink()

        # If zip contained a single root folder, flatten it
        contents = list(dataset_dir.iterdir())
        if len(contents) == 1 and contents[0].is_dir():
            inner = contents[0]
            for item in inner.iterdir():
                shutil.move(str(item), str(dataset_dir / item.name))
            inner.rmdir()

        # Find and parse yaml
        yaml_file = self._find_yaml(dataset_dir)
        if not yaml_file:
            shutil.rmtree(dataset_dir)
            raise ValueError("No data.yaml found in dataset. Expected YOLO format.")

        # Fix paths in yaml to be relative to dataset dir
        with open(yaml_file) as f:
            data = yaml.safe_load(f)

        # Update path to point to dataset directory
        data["path"] = str(dataset_dir)
        with open(yaml_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

        info = self._parse_dataset(dataset_id, dataset_dir, yaml_file)
        if not info:
            shutil.rmtree(dataset_dir)
            raise ValueError("Failed to parse dataset yaml.")

        self._datasets[dataset_id] = info
        return info

    def list_datasets(self) -> list[DatasetInfo]:
        return list(self._datasets.values())

    def get_dataset(self, dataset_id: str) -> DatasetInfo | None:
        return self._datasets.get(dataset_id)

    def delete_dataset(self, dataset_id: str):
        info = self._datasets.pop(dataset_id, None)
        if info:
            shutil.rmtree(info.path, ignore_errors=True)

    def start_training(
        self,
        dataset_id: str,
        base_model: str,
        epochs: int = 100,
        batch_size: int = 16,
        img_size: int = 640,
        lr0: float = 0.01,
        device: str = "cpu",
        project_name: str | None = None,
    ) -> TrainingJob:
        dataset = self._datasets.get(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")

        job_id = f"train_{int(time.time())}"
        project_name = project_name or job_id
        project_name = re.sub(r"[^A-Za-z0-9_-]+", "_", project_name).strip("_")
        if not project_name:
            project_name = job_id

        job = TrainingJob(
            id=job_id,
            dataset_id=dataset_id,
            base_model=base_model,
            status="pending",
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "img_size": img_size,
                "lr0": lr0,
                "device": device,
                "project_name": project_name,
            },
        )
        job.metrics.total_epochs = epochs
        self._jobs[job_id] = job
        self._stop_flags[job_id] = False

        thread = threading.Thread(target=self._run_training, args=(job,), daemon=True)
        self._threads[job_id] = thread
        thread.start()

        return job

    def _run_training(self, job: TrainingJob):
        try:
            job.status = "running"

            dataset = self._datasets[job.dataset_id]
            yaml_file = self._find_yaml(Path(dataset.path))
            if not yaml_file:
                raise ValueError("Dataset yaml not found")

            # Load base model
            model = YOLO(job.base_model)

            # Custom callback to update metrics
            def on_train_epoch_end(trainer):
                if self._stop_flags.get(job.id, False):
                    raise KeyboardInterrupt("Training stopped by user")

                epoch = trainer.epoch + 1
                job.metrics.epoch = epoch
                job.progress = epoch / job.metrics.total_epochs

                # Extract losses from trainer
                if hasattr(trainer, "loss_items") and trainer.loss_items is not None:
                    losses = trainer.loss_items.cpu().numpy()
                    if len(losses) >= 3:
                        job.metrics.box_loss = float(losses[0])
                        job.metrics.cls_loss = float(losses[1])
                        job.metrics.dfl_loss = float(losses[2])

                # Learning rate
                if hasattr(trainer, "optimizer") and trainer.optimizer:
                    for pg in trainer.optimizer.param_groups:
                        job.metrics.lr = pg.get("lr", 0)
                        break

            def on_fit_epoch_end(trainer):
                # Extract validation metrics
                metrics = trainer.metrics
                if hasattr(metrics, "box"):
                    job.metrics.mAP50 = getattr(metrics.box, "map50", 0.0)
                    job.metrics.mAP50_95 = getattr(metrics.box, "map", 0.0)
                elif isinstance(metrics, dict):
                    job.metrics.mAP50 = metrics.get("metrics/mAP50(B)", 0.0)
                    job.metrics.mAP50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)

            model.add_callback("on_train_epoch_end", on_train_epoch_end)
            model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

            # Run training
            results = model.train(
                data=str(yaml_file),
                epochs=job.config["epochs"],
                batch=job.config["batch_size"],
                imgsz=job.config["img_size"],
                lr0=job.config["lr0"],
                device=job.config["device"],
                project=str(RUNS_DIR),
                name=job.config["project_name"],
                exist_ok=True,
                verbose=True,
            )

            # Copy best model to models directory
            run_dir = RUNS_DIR / job.config["project_name"]
            best_pt = run_dir / "weights" / "best.pt"
            if best_pt.exists():
                output_name = f"{job.config['project_name']}_best.pt"
                output_path = MODELS_DIR / output_name
                shutil.copy2(best_pt, output_path)
                job.output_model_path = str(output_path)

            job.status = "completed"
            job.progress = 1.0

        except KeyboardInterrupt:
            job.status = "stopped"
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            print(f"Training failed: {e}")

    def stop_training(self, job_id: str):
        self._stop_flags[job_id] = True
        job = self._jobs.get(job_id)
        if job:
            job.status = "stopped"

    def get_job(self, job_id: str) -> TrainingJob | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[TrainingJob]:
        return list(self._jobs.values())


# Singleton
training_manager = TrainingManager()
