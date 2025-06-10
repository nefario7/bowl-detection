"""Bowl detection dataset implementation with COCO format support.

This module provides dataset classes for loading and processing bowl detection data in COCO format,
including proper transforms and data augmentation.
"""

import os
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CocoBowlDataset(Dataset):
    """Dataset for the Bowl detection task using COCO format annotations."""

    def __init__(
        self,
        img_root_dir: str,
        annotation_file: str,
        transform: Optional[Callable] = None,
    ) -> None:
        """Initialize a `CocoBowlDataset`.

        Args:
            img_root_dir: Root directory where images are stored (e.g., 'data/Bowl.v1i.coco/train/').
            annotation_file: Path to the COCO annotation JSON file (e.g., 'data/Bowl.v1i.coco/annotations/instances_train.json').
            transform: Optional transform to be applied on the image.
        """
        self.img_root_dir = img_root_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Image ID
        img_id = self.ids[idx]

        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_root_dir, img_info["file_name"])

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}. Image info: {img_info}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        if self.transform:
            image_for_transform = image.copy()
            image = self.transform(image_for_transform)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []

        for ann in anns:
            x_min, y_min, bbox_w, bbox_h = ann["bbox"]

            # Convert to [x_min, y_min, x_max, y_max] format (absolute pixel values)
            # This is what torchvision Faster R-CNN and FCOS models expect for targets.
            x_max = x_min + bbox_w
            y_max = y_min + bbox_h

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann["category_id"])

        if boxes:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros(0, dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "iscrowd": torch.as_tensor([ann.get("iscrowd", 0) for ann in anns], dtype=torch.int64),
        }

        return image, target
