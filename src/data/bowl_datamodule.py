import os
from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data.components.bowl_dataset import CocoBowlDataset
from src.data.components.format_adapters import FormatAdapterCollate


class BowlDataModule(LightningDataModule):
    """`LightningDataModule` for the Bowl dataset in COCO format."""

    def __init__(
        self,
        data_dir: str = "data/Bowl.v5i.coco",
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `BowlDataModule`.

        Args:
            data_dir: The root directory of the COCO-formatted dataset.
            batch_size: The batch size.
            num_workers: The number of workers.
            pin_memory: Whether to pin memory.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        # Define paths for COCO structure
        self.train_img_dir = os.path.join(data_dir, "train")
        self.train_ann_file = os.path.join(data_dir, "train", "_annotations.coco.json")
        self.val_img_dir = os.path.join(data_dir, "valid")
        self.val_ann_file = os.path.join(data_dir, "valid", "_annotations.coco.json")
        self.test_img_dir = os.path.join(data_dir, "test")
        self.test_ann_file = os.path.join(data_dir, "test", "_annotations.coco.json")

        # Set up transforms for COCO format
        self.train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.RandomHorizontalFlip(),
            ]
        )
        self.test_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of object classes expected by the model.

        For COCO format (torchvision models like RetinaNet, FasterRCNN): Assumes dataset provides
        category IDs 1 and 2 for two foreground classes. These models expect 1-indexed labels (1 to
        N). num_classes parameter should be max_category_id + 1 (for background). If dataset has
        max category ID 2, then num_classes = 2 + 1 = 3.
        """
        return 3

    def prepare_data(self) -> None:
        """Verify that the dataset exists and is structured correctly (COCO format)."""
        if not os.path.exists(self.train_img_dir):
            raise FileNotFoundError(
                f"Training image directory {self.train_img_dir} does not exist"
            )
        if not os.path.exists(self.train_ann_file):
            raise FileNotFoundError(
                f"Training annotation file {self.train_ann_file} does not exist"
            )
        if not os.path.exists(self.val_img_dir):
            raise FileNotFoundError(
                f"Validation image directory {self.val_img_dir} does not exist"
            )
        if not os.path.exists(self.val_ann_file):
            raise FileNotFoundError(
                f"Validation annotation file {self.val_ann_file} does not exist"
            )
        if not os.path.exists(self.test_img_dir):
            raise FileNotFoundError(f"Test image directory {self.test_img_dir} does not exist")
        if not os.path.exists(self.test_ann_file):
            raise FileNotFoundError(f"Test annotation file {self.test_ann_file} does not exist")

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data using CocoBowlDataset.

        Set variables: `self.data_train`, `self.data_test`.
        """
        current_batch_size = self.hparams["batch_size"]
        if self.trainer is not None:
            if current_batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({current_batch_size}) is not divisible by the "
                    f"number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = current_batch_size // self.trainer.world_size
        else:
            self.batch_size_per_device = current_batch_size

        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = CocoBowlDataset(
                img_root_dir=self.train_img_dir,
                annotation_file=self.train_ann_file,
                transform=self.train_transforms,
            )
            self.data_val = CocoBowlDataset(
                img_root_dir=self.val_img_dir,
                annotation_file=self.val_ann_file,
                transform=self.test_transforms,
            )
            self.data_test = CocoBowlDataset(
                img_root_dir=self.test_img_dir,
                annotation_file=self.test_ann_file,
                transform=self.test_transforms,
            )

    def train_dataloader(self) -> DataLoader[Any]:
        assert (
            self.data_train is not None
        ), "self.data_train is not initialized. Call setup() first."
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=True,
            collate_fn=FormatAdapterCollate("coco"),
        )

    def val_dataloader(self) -> DataLoader[Any]:
        assert self.data_val is not None, "self.data_val is not initialized. Call setup() first."
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=False,
            collate_fn=FormatAdapterCollate("coco"),
        )

    def test_dataloader(self) -> DataLoader[Any]:
        assert self.data_test is not None, "self.data_test is not initialized. Call setup() first."
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=False,
            collate_fn=FormatAdapterCollate("coco"),
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        """Return the same as test_dataloader for predictions."""
        return self.test_dataloader()
