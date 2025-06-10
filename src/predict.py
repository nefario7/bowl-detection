import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import rootutils
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

# Prediction constants
DEFAULT_CONFIDENCE_THRESHOLD = 0.3
PREDICTION_FILENAME_FORMAT = "prediction_batch{:03d}_img{:03d}.jpg"

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def visualize_predictions(
    image: torch.Tensor,
    predictions: Optional[Dict],
    target: Optional[Dict] = None,
    output_path: Optional[str] = None,
    confidence_threshold: float = 0.3,
):
    """Visualize predictions on an image."""

    # Convert tensor to numpy array
    if isinstance(image, torch.Tensor):
        # Assuming image is in CHW format and normalized
        image_np = image.permute(1, 2, 0).cpu().numpy()

        # Denormalize if needed (assuming ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = image_np * std + mean
        image_np = np.clip(image_np, 0, 1)
    else:
        image_np = image

    # Create figure
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_np)

    # Draw ground truth boxes in red
    if target is not None and "boxes" in target:
        boxes = target["boxes"].cpu().numpy()
        labels = target["labels"].cpu().numpy() if "labels" in target else None

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            # Draw red rectangle for ground truth
            rect = patches.Rectangle(
                (x1, y1), width, height, linewidth=2, edgecolor="red", facecolor="none"
            )
            ax.add_patch(rect)

            if labels is not None:
                label_text = f"GT: {labels[i]}"
                ax.text(
                    x1,
                    y1 - 5,
                    label_text,
                    color="red",
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red"),
                )

    # Draw predictions in green
    if predictions is not None:
        if "boxes" in predictions:
            boxes = predictions["boxes"].cpu().numpy()
            scores = predictions["scores"].cpu().numpy() if "scores" in predictions else None
            labels = predictions["labels"].cpu().numpy() if "labels" in predictions else None

            for i, box in enumerate(boxes):
                if scores is not None and scores[i] < confidence_threshold:
                    continue

                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1

                # Draw green rectangle for predictions
                rect = patches.Rectangle(
                    (x1, y1),
                    width,
                    height,
                    linewidth=2,
                    edgecolor="green",
                    facecolor="none",
                )
                ax.add_patch(rect)

                # Add label with confidence score
                label_text = f"Bowl: {scores[i]:.2f}" if scores is not None else "Bowl"
                ax.text(
                    x1,
                    y1 - 5,
                    label_text,
                    color="green",
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="green"),
                )

    ax.set_title("Bowl Detection Results (GT: Red, Predictions: Green)")
    ax.axis("off")

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        log.info(f"Saved visualization to {output_path}")

    plt.close()


@task_wrapper
def predict(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run predictions on the test set and create visualizations.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path, "Checkpoint path must be provided"

    # Create output directory
    output_dir = cfg.get("output_dir", "predictions")
    os.makedirs(output_dir, exist_ok=True)
    log.info(f"Output directory: {output_dir}")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting predictions!")
    predictions = trainer.predict(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    if predictions is None:
        log.error("No predictions returned from trainer.predict()")
        return {}, object_dict

    log.info("Creating visualizations...")

    # Setup datamodule for accessing test data
    datamodule.setup("test")
    test_dataloader = datamodule.test_dataloader()

    confidence_threshold = cfg.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)

    # Process predictions and create visualizations
    for batch_idx, (batch, batch_predictions) in enumerate(zip(test_dataloader, predictions)):
        if len(batch) >= 2:
            images, targets = batch[0], batch[1]
        else:
            images = batch[0]
            targets = [None] * len(images)

        for img_idx, (image, target, prediction) in enumerate(
            zip(images, targets, batch_predictions)
        ):
            output_path = os.path.join(
                output_dir, PREDICTION_FILENAME_FORMAT.format(batch_idx, img_idx)
            )

            visualize_predictions(
                image=image,
                predictions=prediction,
                target=target,
                output_path=output_path,
                confidence_threshold=confidence_threshold,
            )

    log.info(f"Visualizations saved to: {output_dir}")

    # Also run evaluation to get metrics
    log.info("Running evaluation to get metrics...")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_visualize.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for prediction with visualization.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    extras(cfg)
    predict(cfg)


if __name__ == "__main__":
    main()
