"""Convert YOLO format annotations to COCO format.

This script converts bowl detection dataset annotations from YOLO format (normalized coordinates)
to COCO format (absolute coordinates) with proper metadata and category definitions.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2


def convert_yolo_to_coco(dataset_dir: str, output_dir: str, set_name: str) -> Dict:
    """Convert a dataset from YOLO format to COCO format.

    Args:
        dataset_dir: Path to the directory containing 'train' and 'test' subdirectories.
        output_dir: Path to the directory where COCO JSON files will be saved.
        set_name: Either 'train' or 'test'.

    Returns:
        Dictionary containing the COCO format data structure.

    Raises:
        FileNotFoundError: If the dataset directory doesn't exist.
        ValueError: If set_name is not 'train' or 'test'.
    """
    images_dir = os.path.join(dataset_dir, set_name)
    labels_dir = os.path.join(
        dataset_dir, set_name
    )  # Assuming labels are in the same dir as images

    coco_output = {
        "info": {
            "year": datetime.now().year,
            "version": "1.0",
            "description": f"{set_name} set for bowl detection",
            "contributor": "",
            "url": "",
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "categories": [],
        "images": [],
        "annotations": [],
    }

    annotation_id = 1
    image_id_counter = 1

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Directory not found: {images_dir}")

    if set_name not in ["train", "test"]:
        raise ValueError(f"set_name must be 'train' or 'test', got: {set_name}")

    for image_filename in sorted(os.listdir(images_dir)):
        if not image_filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        image_path = os.path.join(images_dir, image_filename)
        label_filename = os.path.splitext(image_filename)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_filename)

        # Read image to get dimensions
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image {image_path}. Skipping.")
                continue
            height, width = image.shape[:2]
        except Exception as e:
            print(f"Error reading image {image_path}: {e}. Skipping.")
            continue

        image_info = {
            "id": image_id_counter,
            "width": width,
            "height": height,
            "file_name": os.path.join(set_name, image_filename),
            "license": 1,
            "date_captured": datetime.now().isoformat(),
        }
        coco_output["images"].append(image_info)

        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    try:
                        class_id, center_x, center_y, bbox_width, bbox_height = map(
                            float, line.split()
                        )
                    except ValueError:
                        print(f"Warning: Skipping malformed line in {label_path}: {line.strip()}")
                        continue

                    x_min = (center_x - bbox_width / 2.0) * width
                    y_min = (center_y - bbox_height / 2.0) * height
                    coco_bbox_width = bbox_width * width
                    coco_bbox_height = bbox_height * height

                    annotation = {
                        "id": annotation_id,
                        "image_id": image_id_counter,
                        "category_id": int(class_id),
                        "bbox": [x_min, y_min, coco_bbox_width, coco_bbox_height],
                        "area": coco_bbox_width * coco_bbox_height,
                        "iscrowd": 0,
                        "segmentation": [],
                    }
                    coco_output["annotations"].append(annotation)
                    annotation_id += 1
        else:
            print(
                f"Warning: Label file not found for {image_filename}. Image added without annotations."
            )

        image_id_counter += 1

    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, f"annotations_{set_name}.json")
    with open(output_filepath, "w") as f:
        json.dump(coco_output, f, indent=4)

    print(f"Successfully converted {set_name} set to COCO format at {output_filepath}")
    return coco_output


def main() -> None:
    """Main function with proper CLI argument parsing."""
    parser = argparse.ArgumentParser(description="Convert YOLO format annotations to COCO format")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/dataset",
        help="Path to dataset directory containing train/test subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for COCO JSON files (default: {dataset_dir}/coco_format)",
    )

    args = parser.parse_args()

    base_dataset_dir = Path(args.dataset_dir)
    coco_output_dir = (
        Path(args.output_dir) if args.output_dir else base_dataset_dir / "coco_format"
    )

    defined_categories = [
        {
            "id": 0,
            "name": "bowl",
            "supercategory": "kitchenware",
        },
    ]

    # Create COCO output directory if it doesn't exist
    coco_output_dir.mkdir(parents=True, exist_ok=True)

    # Process both training and test sets
    for set_name in ["train", "test"]:
        print(f"Processing {set_name} set...")
        try:
            coco_output = convert_yolo_to_coco(
                str(base_dataset_dir), str(coco_output_dir), set_name
            )
            if coco_output and defined_categories:
                coco_output["categories"] = defined_categories
                # Save the updated JSON
                output_filepath = coco_output_dir / f"annotations_{set_name}.json"
                with open(output_filepath, "w") as f:
                    json.dump(coco_output, f, indent=4)
                print(f"Updated categories in {output_filepath}")
        except (FileNotFoundError, ValueError) as e:
            print(f"Error processing {set_name} set: {e}")
            continue

    print("\nConversion complete.")
    print(f"COCO JSON files saved in: {coco_output_dir}")


if __name__ == "__main__":
    main()
