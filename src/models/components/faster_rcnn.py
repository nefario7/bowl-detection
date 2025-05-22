import torch
from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRCNN(nn.Module):
    """Hydra-compatible wrapper for torchvision's Faster R-CNN."""
    def __init__(self, num_classes: int = 2, pretrained: bool = True) -> None:
        """
        Args:
            num_classes: Number of classes (including background, so for 1 class: num_classes=2)
            pretrained: Whether to use ImageNet-pretrained backbone
        """
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        print("Loading Faster R-CNN model with weights: ", weights)
        self.model = fasterrcnn_resnet50_fpn(weights=weights)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, images, targets=None):
        if self.training and targets is not None:
            return self.model(images, targets)
        return self.model(images)
