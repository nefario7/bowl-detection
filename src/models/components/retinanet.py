"""RetinaNet model implementation for bowl detection.

This module provides a Hydra-compatible wrapper around torchvision's RetinaNet implementation with
customizable number of classes and pretrained weights support.
"""

import torch
from torch import nn
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetClassificationHead


class RetinaNet(nn.Module):
    """Hydra-compatible wrapper for torchvision's RetinaNet."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True) -> None:
        """
        Args:
            num_classes: Number of classes (including background, so for 1 class: num_classes=2)
            pretrained: Whether to use ImageNet-pretrained backbone
        """
        super().__init__()

        if pretrained:
            # Load with pretrained COCO weights (91 classes)
            self.model = retinanet_resnet50_fpn(weights="DEFAULT")

            # Replace the classification head for our custom number of classes
            num_anchors = self.model.head.classification_head.num_anchors
            self.model.head.classification_head = RetinaNetClassificationHead(
                in_channels=256,
                num_anchors=num_anchors,
                num_classes=num_classes,
            )
        else:
            # Load without pretrained weights
            self.model = retinanet_resnet50_fpn(weights=None, num_classes=num_classes)

    def forward(self, images, targets=None):
        if self.training and targets is not None:
            return self.model(images, targets)
        return self.model(images)
