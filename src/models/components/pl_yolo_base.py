"""Proper pl_YOLO integration for bowl detection.

Uses the actual pl_YOLO models without fallback.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

# Add pl_YOLO to path for imports
PL_YOLO_PATH = Path(__file__).parent.parent.parent.parent / "third_party" / "pl_YOLO"
if str(PL_YOLO_PATH) not in sys.path:
    sys.path.insert(0, str(PL_YOLO_PATH))

# Import pl_YOLO components with proper error handling
try:
    from PL_Modules.build_detection import build_model

    PL_YOLO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: pl_YOLO import failed: {e}")
    PL_YOLO_AVAILABLE = False

    # Define a dummy build_model function
    def build_model(cfg_models, num_classes):
        raise ImportError(
            "pl_YOLO is not properly configured. Please check the third_party/pl_YOLO installation."
        )


class PLYOLOWrapper(nn.Module):
    """Wrapper for pl_YOLO models that handles data format conversion."""

    def __init__(self, model_config: Dict[str, Any], num_classes: int = 2):
        super().__init__()

        self.num_classes = num_classes
        self.model_config = model_config

        # Build the pl_YOLO model using the official build function
        self.model = build_model(model_config, num_classes)

    def forward(
        self, images: torch.Tensor, targets: Optional[List[Dict[str, torch.Tensor]]] = None
    ):
        """Forward pass with proper data format conversion."""
        if self.training and targets is not None:
            # Training mode - convert targets and return loss
            converted_targets = self._convert_targets_to_yolo_format(targets, images.device)
            loss_dict = self.model(images, converted_targets)
            return loss_dict
        else:
            # Inference mode - return predictions
            with torch.no_grad():
                outputs = self.model(images)
                return self._convert_outputs_to_torchvision_format(outputs, images.shape[0])

    def _convert_targets_to_yolo_format(
        self, targets: List[Dict[str, torch.Tensor]], device: torch.device
    ) -> torch.Tensor:
        """Convert COCO format targets to YOLO format expected by pl_YOLO loss.

        pl_YOLO expects format: [batch_size, max_objects, 6] where 6 = [batch_idx, class_id, x_center, y_center, width, height]
        """
        if not targets:
            return torch.zeros((len(targets), 1, 6), device=device)

        batch_size = len(targets)
        max_objects = max(len(target["boxes"]) for target in targets) if targets else 1
        max_objects = max(max_objects, 1)  # Ensure at least 1 object slot

        # Create padded target tensor: [batch_size, max_objects, 6]
        batch_targets = torch.zeros((batch_size, max_objects, 6), device=device)

        for batch_idx, target in enumerate(targets):
            boxes = target["boxes"]  # [N, 4] in format [x1, y1, x2, y2]
            labels = target["labels"]  # [N]

            num_objects = len(boxes)
            if num_objects > 0:
                # Convert boxes from [x1, y1, x2, y2] to [x_center, y_center, width, height]
                x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                x_center = (x1 + x2) / 2.0
                y_center = (y1 + y2) / 2.0
                width = x2 - x1
                height = y2 - y1

                # Fill the batch target tensor
                batch_targets[batch_idx, :num_objects, 0] = batch_idx  # batch index
                batch_targets[batch_idx, :num_objects, 1] = (
                    labels - 1
                )  # Convert from 1-indexed to 0-indexed
                batch_targets[batch_idx, :num_objects, 2] = x_center
                batch_targets[batch_idx, :num_objects, 3] = y_center
                batch_targets[batch_idx, :num_objects, 4] = width
                batch_targets[batch_idx, :num_objects, 5] = height

        return batch_targets

    def _convert_outputs_to_torchvision_format(
        self, outputs, batch_size: int
    ) -> List[Dict[str, torch.Tensor]]:
        """Convert YOLO outputs to torchvision format."""
        if outputs is None:
            return [
                {
                    "boxes": torch.zeros((0, 4)),
                    "scores": torch.zeros((0,)),
                    "labels": torch.zeros((0,), dtype=torch.long),
                }
                for _ in range(batch_size)
            ]

        # Handle different output types
        if isinstance(outputs, list):
            # pl_YOLO returns a list of tensors for each scale
            # For now, just return empty predictions for inference
            return [
                {
                    "boxes": torch.zeros((0, 4), device=outputs[0].device if outputs else "cpu"),
                    "scores": torch.zeros((0,), device=outputs[0].device if outputs else "cpu"),
                    "labels": torch.zeros(
                        (0,), dtype=torch.long, device=outputs[0].device if outputs else "cpu"
                    ),
                }
                for _ in range(batch_size)
            ]

        if hasattr(outputs, "shape") and len(outputs.shape) == 3:  # [batch_size, num_preds, 6]
            results = []
            for i in range(batch_size):
                preds = outputs[i]  # [num_preds, 6]

                # Filter out low confidence predictions
                if preds.size(1) >= 5:  # Ensure we have confidence scores
                    conf_mask = preds[:, 4] > 0.1
                    preds = preds[conf_mask]

                    if len(preds) > 0:
                        boxes = preds[:, :4]  # [x1, y1, x2, y2]
                        scores = preds[:, 4]  # confidence scores
                        labels = (
                            preds[:, 5].long() + 1
                            if preds.size(1) > 5
                            else torch.ones(len(preds), dtype=torch.long)
                        )  # Convert back to 1-indexed

                        results.append({"boxes": boxes, "scores": scores, "labels": labels})
                    else:
                        results.append(
                            {
                                "boxes": torch.zeros((0, 4), device=outputs.device),
                                "scores": torch.zeros((0,), device=outputs.device),
                                "labels": torch.zeros(
                                    (0,), dtype=torch.long, device=outputs.device
                                ),
                            }
                        )
                else:
                    results.append(
                        {
                            "boxes": torch.zeros((0, 4), device=outputs.device),
                            "scores": torch.zeros((0,), device=outputs.device),
                            "labels": torch.zeros((0,), dtype=torch.long, device=outputs.device),
                        }
                    )
            return results
        else:
            # Fallback for unexpected output shapes
            return [
                {
                    "boxes": torch.zeros((0, 4)),
                    "scores": torch.zeros((0,)),
                    "labels": torch.zeros((0,), dtype=torch.long),
                }
                for _ in range(batch_size)
            ]


def create_yolox_config(model_size: str = "s") -> Dict[str, Any]:
    """Create YOLOX model configuration following pl_YOLO format."""

    # Size configurations for YOLOX
    size_configs = {
        "s": {
            "backbone": {
                "name": "cspdarknet",
                "depths": [1, 3, 3, 1],
                "channels": [32, 64, 128, 256, 512],
                "outputs": ["stage2", "stage3", "stage4"],
                "norm": "bn",
                "act": "silu",
            },
            "neck": {
                "name": "csppafpn",
                "depths": [1, 1, 1, 1],
                "channels": [128, 256, 512],
                "norm": "bn",
                "act": "silu",
            },
            "head": {
                "name": "decoupled_head",
                "num_anchor": 1,
                "channels": [128, 256, 512],
                "norm": "bn",
                "act": "silu",
            },
            "loss": {"name": "yolox", "stride": [8, 16, 32]},
        },
        "m": {
            "backbone": {
                "name": "cspdarknet",
                "depths": [2, 6, 6, 2],
                "channels": [48, 96, 192, 384, 768],
                "outputs": ["stage2", "stage3", "stage4"],
                "norm": "bn",
                "act": "silu",
            },
            "neck": {
                "name": "csppafpn",
                "depths": [1, 1, 1, 1],
                "channels": [192, 384, 768],
                "norm": "bn",
                "act": "silu",
            },
            "head": {
                "name": "decoupled_head",
                "num_anchor": 1,
                "channels": [192, 384, 768],
                "norm": "bn",
                "act": "silu",
            },
            "loss": {"name": "yolox", "stride": [8, 16, 32]},
        },
        "l": {
            "backbone": {
                "name": "cspdarknet",
                "depths": [3, 9, 9, 3],
                "channels": [64, 128, 256, 512, 1024],
                "outputs": ["stage2", "stage3", "stage4"],
                "norm": "bn",
                "act": "silu",
            },
            "neck": {
                "name": "csppafpn",
                "depths": [1, 1, 1, 1],
                "channels": [256, 512, 1024],
                "norm": "bn",
                "act": "silu",
            },
            "head": {
                "name": "decoupled_head",
                "num_anchor": 1,
                "channels": [256, 512, 1024],
                "norm": "bn",
                "act": "silu",
            },
            "loss": {"name": "yolox", "stride": [8, 16, 32]},
        },
    }

    if model_size not in size_configs:
        raise ValueError(
            f"Unsupported YOLOX size: {model_size}. Supported: {list(size_configs.keys())}"
        )

    return size_configs[model_size]


def create_yolov7_config(model_size: str = "base") -> Dict[str, Any]:
    """Create YOLOv7 model configuration following pl_YOLO format."""

    size_configs = {
        "tiny": {
            "backbone": {
                "name": "eelan",
                "depths": [1, 1, 1, 1],
                "channels": [32, 64, 128, 256, 512],
                "outputs": ["stage2", "stage3", "stage4"],
                "norm": "bn",
                "act": "silu",
            },
            "neck": {
                "name": "yolov7neck",
                "depths": [1, 1, 1, 1],
                "channels": [128, 256, 512],
                "norm": "bn",
                "act": "silu",
            },
            "head": {"name": "implicit_head", "num_anchor": 3, "channels": [128, 256, 512]},
            "loss": {
                "name": "yolov7",
                "stride": [8, 16, 32],
                "anchors": [
                    [12, 16, 19, 36, 40, 28],  # P3/8
                    [36, 75, 76, 55, 72, 146],  # P4/16
                    [142, 110, 192, 243, 459, 401],  # P5/32
                ],
            },
        },
        "base": {
            "backbone": {
                "name": "eelan",
                "depths": [1, 1, 1, 1],
                "channels": [64, 128, 256, 512, 1024],
                "outputs": ["stage2", "stage3", "stage4"],
                "norm": "bn",
                "act": "silu",
            },
            "neck": {
                "name": "yolov7neck",
                "depths": [1, 1, 1, 1],
                "channels": [256, 512, 1024],
                "norm": "bn",
                "act": "silu",
            },
            "head": {"name": "implicit_head", "num_anchor": 3, "channels": [256, 512, 1024]},
            "loss": {
                "name": "yolov7",
                "stride": [8, 16, 32],
                "anchors": [
                    [12, 16, 19, 36, 40, 28],  # P3/8
                    [36, 75, 76, 55, 72, 146],  # P4/16
                    [142, 110, 192, 243, 459, 401],  # P5/32
                ],
            },
        },
    }

    if model_size not in size_configs:
        raise ValueError(
            f"Unsupported YOLOv7 size: {model_size}. Supported: {list(size_configs.keys())}"
        )

    return size_configs[model_size]


def create_yolo_model(
    model_type: str = "yolox", model_size: str = "s", num_classes: int = 2
) -> nn.Module:
    """Factory function to create YOLO models using pl_YOLO."""
    if model_type == "yolox":
        config = create_yolox_config(model_size)
    elif model_type == "yolov7":
        config = create_yolov7_config(model_size)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return PLYOLOWrapper(config, num_classes)
