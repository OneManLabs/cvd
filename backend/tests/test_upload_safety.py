import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from app.services.model_manager import ModelManager
from app.services.training import TrainingManager


class UploadSafetyTests(unittest.TestCase):
    def test_dataset_upload_rejects_parent_path(self):
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as file:
            file.writestr("../outside.txt", "unsafe")

        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory) / "datasets"
            data_root.mkdir()
            with patch("app.services.training.DATASETS_DIR", data_root):
                manager = TrainingManager()
                with self.assertRaisesRegex(ValueError, "unsafe path"):
                    manager.upload_dataset("sample.zip", archive.getvalue())
                self.assertFalse((Path(directory) / "outside.txt").exists())

    def test_model_upload_removes_parent_path(self):
        with tempfile.TemporaryDirectory() as directory:
            model_root = Path(directory) / "models"
            model_root.mkdir()
            with patch("app.services.model_manager.MODELS_DIR", model_root):
                manager = ModelManager()
                manager.add_custom_model("../sample.onnx", b"model")
                self.assertTrue((model_root / "sample.onnx").exists())
                self.assertFalse((Path(directory) / "sample.onnx").exists())


if __name__ == "__main__":
    unittest.main()
