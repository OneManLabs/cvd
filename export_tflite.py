#!/usr/bin/env python3
"""Export all YOLO models to TFLite format for the ATAK-CV Android app.

Usage:
    python export_tflite.py [--output-dir /path/to/atak-cv/app/src/main/assets]
    python export_tflite.py --models yolo26n yolo11n

This will download and convert each model. Takes a while on first run.
"""

import argparse
import shutil
import os
from pathlib import Path

# Disable ultralytics auto-install that keeps failing on version constraints
os.environ["YOLO_AUTOINSTALL"] = "false"

from ultralytics import YOLO

MODELS = [
    # YOLO26
    "yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x",
    "yolo26n-seg", "yolo26s-seg", "yolo26m-seg", "yolo26l-seg", "yolo26x-seg",
    "yolo26n-pose", "yolo26s-pose", "yolo26m-pose", "yolo26l-pose", "yolo26x-pose",
    "yolo26n-cls", "yolo26s-cls", "yolo26m-cls", "yolo26l-cls", "yolo26x-cls",
    "yolo26n-obb", "yolo26s-obb", "yolo26m-obb", "yolo26l-obb", "yolo26x-obb",
    # YOLO11
    "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
    "yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo11x-seg",
    "yolo11n-pose", "yolo11s-pose", "yolo11m-pose", "yolo11l-pose", "yolo11x-pose",
    "yolo11n-cls", "yolo11s-cls", "yolo11m-cls", "yolo11l-cls", "yolo11x-cls",
    "yolo11n-obb", "yolo11s-obb", "yolo11m-obb", "yolo11l-obb", "yolo11x-obb",
    # YOLOv8
    "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
    "yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg", "yolov8x-seg",
    "yolov8n-pose", "yolov8s-pose", "yolov8m-pose", "yolov8l-pose", "yolov8x-pose",
    "yolov8n-cls", "yolov8s-cls", "yolov8m-cls", "yolov8l-cls", "yolov8x-cls",
    "yolov8n-obb", "yolov8s-obb", "yolov8m-obb", "yolov8l-obb", "yolov8x-obb",
    # YOLOv10
    "yolov10n", "yolov10s", "yolov10m", "yolov10l", "yolov10x",
    # YOLOv9
    "yolov9c", "yolov9e",
    # YOLOv5
    "yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x",
]


def find_tflite(model_name: str) -> Path | None:
    """Search for the exported .tflite file in common locations."""
    search_patterns = [
        f"{model_name}_saved_model/{model_name}_float32.tflite",
        f"{model_name}_saved_model/{model_name}_float16.tflite",
    ]

    for pattern in search_patterns:
        p = Path(pattern)
        if p.exists():
            return p

    # Glob for any matching tflite
    for tflite in Path(".").glob(f"**/{model_name}*.tflite"):
        return tflite

    return None


def export_model(model_name: str, output_dir: Path) -> bool:
    try:
        print(f"\n{'='*50}")
        print(f"Exporting {model_name}...")
        print(f"{'='*50}")

        model = YOLO(f"{model_name}.pt")

        # First try: export directly to tflite
        try:
            model.export(format="tflite")
            tflite = find_tflite(model_name)
            if tflite:
                dest = output_dir / f"{model_name}.tflite"
                shutil.copy2(str(tflite), str(dest))
                print(f"  -> Saved to {dest}")
                return True
        except Exception as e:
            print(f"  -> Direct TFLite export failed: {e}")
            print(f"  -> Trying ONNX export instead...")

        # Fallback: export to ONNX (always works)
        model.export(format="onnx")
        onnx_path = Path(f"{model_name}.onnx")
        if onnx_path.exists():
            # Try onnx2tf conversion
            try:
                import onnx2tf
                onnx2tf.convert(
                    input_onnx_file_path=str(onnx_path),
                    output_folder_path=f"{model_name}_saved_model",
                    non_verbose=True,
                    output_nms_with_dynamic_tensor=True,
                )
                tflite = find_tflite(model_name)
                if tflite:
                    dest = output_dir / f"{model_name}.tflite"
                    shutil.copy2(str(tflite), str(dest))
                    print(f"  -> Saved to {dest} (via onnx2tf)")
                    return True
            except Exception as e2:
                print(f"  -> onnx2tf failed: {e2}")

            # Last resort: save ONNX for manual conversion
            dest = output_dir / f"{model_name}.onnx"
            shutil.copy2(str(onnx_path), str(dest))
            print(f"  -> Saved ONNX to {dest} (TFLite conversion failed, ONNX saved as fallback)")
            return True

        print(f"  -> WARNING: All export methods failed")
        return False

    except Exception as e:
        print(f"  -> FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export YOLO models to TFLite")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent.parent / "atak-cv" / "app" / "src" / "main" / "assets"),
        help="Output directory for .tflite files",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific models to export (default: all). E.g., --models yolo26n yolo11n",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_to_export = args.models if args.models else MODELS

    print(f"Exporting {len(models_to_export)} models to {output_dir}")
    print(f"This may take a while...\n")

    success = 0
    failed = 0
    for model_name in models_to_export:
        if export_model(model_name, output_dir):
            success += 1
        else:
            failed += 1

    print(f"\n{'='*50}")
    print(f"Done! {success} exported, {failed} failed.")
    print(f"Output: {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
