"""YOLOX implementation using simplified pl_YOLO integration."""

from typing import Any, Dict, Optional

import torch
from torch import nn

from .pl_yolo_base import create_yolo_model


class YOLOX(nn.Module):
    """YOLOX object detection model using pl_YOLO implementation."""

    def __init__(
        self, num_classes: int = 2, model_size: str = "s", pretrained: bool = False
    ) -> None:
        """Initialize YOLOX model.

        Args:
            num_classes: Number of classes (excluding background)
            model_size: Model size - 's', 'm', 'l', 'x'
            pretrained: Whether to use pretrained weights (not implemented yet)
        """
        super().__init__()

        self.num_classes = num_classes
        self.model_size = model_size

        # Create the YOLO model using factory function
        self.model = create_yolo_model("yolox", model_size, num_classes)

    def forward(self, images, targets=None):
        """Forward pass.

        Args:
            images: Input images tensor [batch_size, 3, H, W]
            targets: Ground truth targets (for training)

        Returns:
            During training: loss dictionary
            During inference: detection predictions
        """
        return self.model(images, targets)

    def get_num_params(self) -> int:
        """Get the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": f"YOLOX-{self.model_size.upper()}",
            "num_classes": self.num_classes,
            "num_parameters": self.get_num_params(),
            "model_size": self.model_size,
        }
