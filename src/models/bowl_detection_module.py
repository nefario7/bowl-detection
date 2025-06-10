from typing import Any, Dict, List, Optional, Tuple

import torch
from lightning import LightningModule
from torchmetrics.detection.ciou import CompleteIntersectionOverUnion
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class BowlDetectionModule(LightningModule):
    """`LightningModule` for Bowl Object Detection."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        compile: bool,
    ) -> None:
        """Initialize a `BowlDetectionModule`.

        :param net: The detection model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param compile: Whether to compile the model with torch.compile.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = net

        # Detection metrics
        self.train_ciou = CompleteIntersectionOverUnion()
        self.val_ciou = CompleteIntersectionOverUnion()
        self.test_ciou = CompleteIntersectionOverUnion()

        self.train_map = MeanAveragePrecision()
        self.val_map = MeanAveragePrecision()
        self.test_map = MeanAveragePrecision()

    def forward(self, x: torch.Tensor) -> Any:
        """Forward pass through the detection model (inference mode)."""
        return self.net(x)

    def model_step(
        self,
        batch: Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]],
        is_train: bool = True,
    ) -> Tuple[
        torch.Tensor, List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]
    ]:
        """Perform a single model step on a batch of data.

        :param batch: (images, targets) where images is a tensor and targets is a list of dicts.
        :param is_train: Whether this is a training step.
        :return: (loss, predictions, targets)
        """
        images, targets = batch

        if is_train:
            self.net.train()
            # In training mode, the model returns loss dict
            loss_dict = self.net(images, targets)
            # Don't create a new tensor - preserve gradients by summing directly
            losses = list(loss_dict.values())
            if losses:
                # Ensure we start with a tensor for proper gradient tracking
                loss = losses[0]
                for i in range(1, len(losses)):
                    loss = loss + losses[i]
            else:
                loss = torch.tensor(0.0, device=images.device, requires_grad=True)

            # Get predictions for metrics (in eval mode)
            self.net.eval()
            with torch.no_grad():
                preds = self.net(images)
            self.net.train()
        else:
            # In eval mode
            self.net.eval()
            with torch.no_grad():
                preds = self.net(images)
            loss = torch.tensor(0.0, device=images.device)

        return loss, preds, targets

    def training_step(
        self, batch: Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]], batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        loss, preds, targets = self.model_step(batch, is_train=True)
        images, _ = batch
        batch_size = len(images)

        # Update metrics
        self.train_ciou.update(preds, targets)
        self.train_map.update(preds, targets)

        # Log loss with explicit batch size
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        ciou_result = self.train_ciou.compute()
        map_result = self.train_map.compute()

        self.log("train/ciou", ciou_result["ciou"], on_epoch=True, prog_bar=True)
        self.log("train/mAP", map_result["map"], on_epoch=True, prog_bar=True)

        self.train_ciou.reset()
        self.train_map.reset()

    def validation_step(
        self, batch: Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]], batch_idx: int
    ) -> None:
        """Validation step."""
        loss, preds, targets = self.model_step(batch, is_train=False)
        images, _ = batch
        batch_size = len(images)

        # Update metrics
        self.val_ciou.update(preds, targets)
        self.val_map.update(preds, targets)

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        ciou_result = self.val_ciou.compute()
        map_result = self.val_map.compute()

        self.log("val/ciou", ciou_result["ciou"], on_epoch=True, prog_bar=True)
        self.log("val/mAP", map_result["map"], on_epoch=True, prog_bar=True)

        self.val_ciou.reset()
        self.val_map.reset()

    def test_step(
        self, batch: Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]], batch_idx: int
    ) -> None:
        """Test step."""
        loss, preds, targets = self.model_step(batch, is_train=False)
        images, _ = batch
        batch_size = len(images)

        # Update metrics
        self.test_ciou.update(preds, targets)
        self.test_map.update(preds, targets)

    def on_test_epoch_end(self) -> None:
        """Called at the end of test epoch."""
        ciou_result = self.test_ciou.compute()
        map_result = self.test_map.compute()

        self.log("test/ciou", ciou_result["ciou"], on_epoch=True, prog_bar=True)
        self.log("test/mAP", map_result["map"], on_epoch=True, prog_bar=True)

        self.test_ciou.reset()
        self.test_map.reset()

    def predict_step(
        self, batch: Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]], batch_idx: int
    ) -> List[Dict[str, torch.Tensor]]:
        """Prediction step - extract images from batch and run inference."""
        images, _ = batch  # Extract images from the batch, ignore targets

        self.net.eval()
        with torch.no_grad():
            predictions = self.net(images)

        return predictions

    def setup(self, stage: str) -> None:
        """Lightning hook for setup."""
        if self.hparams.get("compile", False) and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers."""
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/mAP",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    import torch.nn as nn

    # Create dummy objects for testing
    dummy_net = nn.Linear(1, 1)  # Simple dummy network
    dummy_optimizer = torch.optim.Adam(dummy_net.parameters())
    _ = BowlDetectionModule(
        net=dummy_net, optimizer=dummy_optimizer, scheduler=None, compile=False
    )
