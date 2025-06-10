"""Format adapters for object detection models."""

from typing import Any, Dict, List, Tuple

import torch


class FormatAdapterCollate:
    """Collate function for object detection models.

    Currently supports COCO format which is used by torchvision models like FasterRCNN and
    RetinaNet.
    """

    def __init__(self, model_type: str = "coco"):
        """Initialize collate function.

        Args:
            model_type: Format type, currently only 'coco' is supported
        """
        self.model_type = model_type
        if model_type != "coco":
            raise ValueError(f"Only 'coco' format is currently supported, got: {model_type}")

    def __call__(self, batch):
        """Custom collate function for COCO format.

        Args:
            batch: List of (image, target) tuples

        Returns:
            images: Stacked tensor [B, C, H, W]
            targets: List of target dictionaries
        """
        images = []
        targets = []

        for image, target in batch:
            images.append(image)
            targets.append(target)

        images = torch.stack(images, dim=0)
        return images, targets
