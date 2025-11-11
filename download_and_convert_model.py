#!/usr/bin/env python3
"""
Script to download and convert YOLOv8 models to OpenVINO format
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

def check_ultralytics():
    """Check if ultralytics is installed"""
    try:
        import ultralytics
        print(f"✓ ultralytics {ultralytics.__version__} is installed")
        return True
    except ImportError:
        print("✗ ultralytics is not installed")
        print("\nInstalling ultralytics...")
        os.system(f"{sys.executable} -m pip install ultralytics")
        return True

def download_and_convert(model_name='yolov8n', task='detect'):
    """
    Download and convert YOLOv8 model to OpenVINO format

    Args:
        model_name: Model size (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        task: Task type (detect, segment, classify, pose)
    """
    from ultralytics import YOLO

    # Model file name
    if task == 'detect':
        pt_file = f"{model_name}.pt"
    else:
        pt_file = f"{model_name}-{task}.pt"

    print(f"\n{'='*60}")
    print(f"Downloading and converting {pt_file}...")
    print(f"{'='*60}\n")

    # Download model
    print(f"Step 1: Loading model {pt_file}...")
    model = YOLO(pt_file)
    print(f"✓ Model loaded successfully")

    # Export to OpenVINO
    print(f"\nStep 2: Converting to OpenVINO format...")
    model.export(format='openvino')
    print(f"✓ Model converted successfully")

    # Find output directory
    openvino_dir = f"{model_name}_openvino_model" if task == 'detect' else f"{model_name}-{task}_openvino_model"

    if not os.path.exists(openvino_dir):
        print(f"✗ ERROR: Output directory not found: {openvino_dir}")
        return False

    # Copy to model directory
    print(f"\nStep 3: Copying files to ./sdcb-openvino-yolov8-cls/model/...")

    model_dir = Path("./sdcb-openvino-yolov8-cls/model")
    model_dir.mkdir(parents=True, exist_ok=True)

    xml_file = Path(openvino_dir) / f"{model_name}.xml"
    bin_file = Path(openvino_dir) / f"{model_name}.bin"

    if not xml_file.exists() or not bin_file.exists():
        print(f"✗ ERROR: Model files not found in {openvino_dir}")
        return False

    # Copy files
    shutil.copy(xml_file, model_dir / f"{model_name}.xml")
    shutil.copy(bin_file, model_dir / f"{model_name}.bin")

    print(f"✓ Copied {xml_file.name} ({xml_file.stat().st_size / 1024:.1f} KB)")
    print(f"✓ Copied {bin_file.name} ({bin_file.stat().st_size / 1024 / 1024:.1f} MB)")

    print(f"\n{'='*60}")
    print(f"SUCCESS! Model ready to use.")
    print(f"{'='*60}")
    print(f"\nModel files location:")
    print(f"  - {model_dir / f'{model_name}.xml'}")
    print(f"  - {model_dir / f'{model_name}.bin'}")

    if task == 'detect':
        print(f"\nYou can now run: cd sdcb-openvino-yolov8-cls && dotnet run")
        print(f"Then select option 2 for Object Detection with Webcam")

    return True

def main():
    parser = argparse.ArgumentParser(
        description='Download and convert YOLOv8 models to OpenVINO format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download YOLOv8n detection model (default, fastest)
  python download_and_convert_model.py

  # Download YOLOv8s detection model (better accuracy)
  python download_and_convert_model.py --model yolov8s

  # Download YOLOv8m detection model (even better accuracy)
  python download_and_convert_model.py --model yolov8m

  # Download YOLOv8n classification model
  python download_and_convert_model.py --model yolov8n --task classify

Available models:
  - yolov8n: Nano (smallest, fastest)
  - yolov8s: Small
  - yolov8m: Medium
  - yolov8l: Large
  - yolov8x: Extra Large (best accuracy, slowest)

Available tasks:
  - detect: Object detection (default)
  - segment: Instance segmentation
  - classify: Classification
  - pose: Pose estimation
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n',
        choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
        help='Model size (default: yolov8n)'
    )

    parser.add_argument(
        '--task',
        type=str,
        default='detect',
        choices=['detect', 'segment', 'classify', 'pose'],
        help='Task type (default: detect)'
    )

    args = parser.parse_args()

    print("YOLOv8 Model Downloader and Converter for OpenVINO")
    print("=" * 60)

    # Check ultralytics installation
    if not check_ultralytics():
        print("Failed to install ultralytics")
        return 1

    # Download and convert
    success = download_and_convert(args.model, args.task)

    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
