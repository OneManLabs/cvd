#!/usr/bin/env python3
"""Export YOLO models to CoreML format for the CVD iOS app.

Usage:
    python export_coreml.py --models yolo26n yolo11n
    python export_coreml.py --models yolov8n-seg yolo11n-seg
    python export_coreml.py --seg
    python export_coreml.py --output-dir /path/to/cvd-ios/CVD
"""

import argparse
import hashlib
import re
import shutil
import os
from pathlib import Path

os.environ["YOLO_AUTOINSTALL"] = "false"

from ultralytics import YOLO

MODELS = [
    "yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x",
    "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
    "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
    "yolov10n", "yolov10s", "yolov10m",
    "yolov5n", "yolov5s", "yolov5m",
]

SEG_MODELS = [
    "yolo26n-seg", "yolo26s-seg", "yolo26m-seg",
    "yolo11n-seg", "yolo11s-seg", "yolo11m-seg",
    "yolov8n-seg", "yolov8s-seg", "yolov8m-seg",
]


def make_pbx_id(seed: str) -> str:
    """Generate a deterministic 24-char hex ID for pbxproj entries."""
    return hashlib.md5(seed.encode()).hexdigest()[:24].upper()


def add_to_xcode_project(model_name: str, project_dir: Path) -> bool:
    """Add an .mlpackage to the Xcode project.pbxproj if not already present."""
    pbxproj = project_dir / "CVD.xcodeproj" / "project.pbxproj"
    if not pbxproj.exists():
        print(f"  -> WARNING: Could not find {pbxproj}, skipping Xcode registration")
        return False

    filename = f"{model_name}.mlpackage"
    content = pbxproj.read_text()

    # Already registered?
    if filename in content:
        print(f"  -> Already in Xcode project")
        return True

    # Generate deterministic IDs
    file_ref_id = make_pbx_id(f"fileref-{filename}")
    build_file_id = make_pbx_id(f"buildfile-{filename}")

    # 1. Add PBXBuildFile entry
    build_file_line = f"\t\t{build_file_id} /* {filename} in Resources */ = {{isa = PBXBuildFile; fileRef = {file_ref_id} /* {filename} */; }};\n"
    content = content.replace(
        "/* End PBXBuildFile section */",
        f"{build_file_line}/* End PBXBuildFile section */",
    )

    # 2. Add PBXFileReference entry
    file_ref_line = f'\t\t{file_ref_id} /* {filename} */ = {{isa = PBXFileReference; lastKnownFileType = folder.mlpackage; path = "{filename}"; sourceTree = "<group>"; }};\n'
    content = content.replace(
        "/* End PBXFileReference section */",
        f"{file_ref_line}/* End PBXFileReference section */",
    )

    # 3. Add to CVD group children (before the closing paren of the CVD group)
    # Find the CVD group and add before the last entry
    group_entry = f"\t\t\t\t{file_ref_id} /* {filename} */,\n"
    # Insert before the closing ");" of the CVD group's children list
    # The CVD group has "path = CVD;" — find its children list
    cvd_group_pattern = r'(children = \(\n(?:\t\t\t\t[A-F0-9]+ /\*[^*]*\*/,\n)*?)(\t\t\t\);\n\t\t\tpath = CVD;)'
    match = re.search(cvd_group_pattern, content)
    if match:
        content = content[:match.end(1)] + group_entry + content[match.start(2):]

    # 4. Add to PBXResourcesBuildPhase
    resource_entry = f"\t\t\t\t{build_file_id} /* {filename} in Resources */,\n"
    # Insert before the closing ");" of the Resources files list
    resources_pattern = r'(CA98F64D4E8358D7E8DD0323 /\* Resources \*/ = \{\n\t\t\tisa = PBXResourcesBuildPhase;\n\t\t\tbuildActionMask = \d+;\n\t\t\tfiles = \(\n(?:\t\t\t\t[A-F0-9]+ /\*[^*]*\*/,\n)*?)(\t\t\t\);)'
    match = re.search(resources_pattern, content)
    if match:
        content = content[:match.end(1)] + resource_entry + content[match.start(2):]

    pbxproj.write_text(content)
    print(f"  -> Added to Xcode project")
    return True


def export_model(model_name: str, output_dir: Path, project_dir: Path, fmt: str = "auto") -> bool:
    try:
        is_seg = "-seg" in model_name
        model = YOLO(f"{model_name}.pt")

        # Determine format: YOLO11/26 need ONNX (coremltools bug), YOLOv8 can use CoreML
        if fmt == "auto":
            needs_onnx = model_name.startswith(("yolo11", "yolo26"))
            fmt = "onnx" if needs_onnx else "coreml"

        print(f"\n{'='*50}")
        print(f"Exporting {model_name} to {fmt.upper()}...")
        print(f"{'='*50}")

        if fmt == "onnx":
            model.export(format="onnx", opset=13, simplify=True)
            onnx_file = Path(f"{model_name}.onnx")
            if onnx_file.exists():
                dest = output_dir / f"{model_name}.onnx"
                shutil.copy2(str(onnx_file), str(dest))
                print(f"  -> Saved to {dest}")
                add_onnx_to_xcode_project(model_name, project_dir)
                return True
        else:
            model.export(format="coreml", nms=not is_seg)
            mlpackage = Path(f"{model_name}.mlpackage")
            if mlpackage.exists():
                dest = output_dir / f"{model_name}.mlpackage"
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(str(mlpackage), str(dest))
                print(f"  -> Saved to {dest}")
                add_to_xcode_project(model_name, project_dir)
                return True

        print(f"  -> WARNING: Could not find exported model file")
        return False

    except Exception as e:
        print(f"  -> FAILED: {e}")
        return False


def add_onnx_to_xcode_project(model_name: str, project_dir: Path) -> bool:
    """Add an .onnx file to the Xcode project.pbxproj Resources."""
    pbxproj = project_dir / "CVD.xcodeproj" / "project.pbxproj"
    if not pbxproj.exists():
        print(f"  -> WARNING: Could not find {pbxproj}")
        return False

    filename = f"{model_name}.onnx"
    content = pbxproj.read_text()

    if filename in content:
        print(f"  -> Already in Xcode project")
        return True

    file_ref_id = make_pbx_id(f"fileref-{filename}")
    build_file_id = make_pbx_id(f"buildfile-{filename}")

    # PBXBuildFile
    build_file_line = f'\t\t{build_file_id} /* {filename} in Resources */ = {{isa = PBXBuildFile; fileRef = {file_ref_id} /* {filename} */; }};\n'
    content = content.replace(
        "/* End PBXBuildFile section */",
        f"{build_file_line}/* End PBXBuildFile section */",
    )

    # PBXFileReference
    file_ref_line = f'\t\t{file_ref_id} /* {filename} */ = {{isa = PBXFileReference; lastKnownFileType = file; path = "{filename}"; sourceTree = "<group>"; }};\n'
    content = content.replace(
        "/* End PBXFileReference section */",
        f"{file_ref_line}/* End PBXFileReference section */",
    )

    # CVD group
    group_entry = f"\t\t\t\t{file_ref_id} /* {filename} */,\n"
    cvd_group_pattern = r'(children = \(\n(?:\t\t\t\t[A-F0-9]+ /\*[^*]*\*/,\n)*?)(\t\t\t\);\n\t\t\tpath = CVD;)'
    match = re.search(cvd_group_pattern, content)
    if match:
        content = content[:match.end(1)] + group_entry + content[match.start(2):]

    # Resources build phase
    resource_entry = f"\t\t\t\t{build_file_id} /* {filename} in Resources */,\n"
    resources_pattern = r'(CA98F64D4E8358D7E8DD0323 /\* Resources \*/ = \{\n\t\t\tisa = PBXResourcesBuildPhase;\n\t\t\tbuildActionMask = \d+;\n\t\t\tfiles = \(\n(?:\t\t\t\t[A-F0-9]+ /\*[^*]*\*/,\n)*?)(\t\t\t\);)'
    match = re.search(resources_pattern, content)
    if match:
        content = content[:match.end(1)] + resource_entry + content[match.start(2):]

    pbxproj.write_text(content)
    print(f"  -> Added to Xcode project")
    return True


def main():
    ios_project_dir = Path(__file__).parent.parent / "cvd-ios"
    default_output = str(ios_project_dir / "CVD")

    parser = argparse.ArgumentParser(description="Export YOLO models to CoreML")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_output,
        help="Output directory for .mlpackage files",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific models to export (default: all detect). E.g., --models yolo26n yolov8n-seg",
    )
    parser.add_argument(
        "--seg",
        action="store_true",
        help="Export segmentation models (default seg list)",
    )
    parser.add_argument(
        "--xcode-project",
        type=str,
        default=str(ios_project_dir),
        help="Path to the cvd-ios project directory (for auto-registering in Xcode)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    project_dir = Path(args.xcode_project)

    if args.models:
        models_to_export = args.models
    elif args.seg:
        models_to_export = SEG_MODELS
    else:
        models_to_export = MODELS

    print(f"Exporting {len(models_to_export)} models to CoreML")
    print(f"Output: {output_dir}")
    if project_dir.exists():
        print(f"Xcode project: {project_dir}")
    print()

    success = 0
    failed = 0
    for model_name in models_to_export:
        if export_model(model_name, output_dir, project_dir):
            success += 1
        else:
            failed += 1

    print(f"\n{'='*50}")
    print(f"Done! {success} exported, {failed} failed.")
    print(f"Models are registered in the Xcode project — just build and run.")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
