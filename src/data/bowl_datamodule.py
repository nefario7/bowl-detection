import os
from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data.components.bowl_dataset import CocoBowlDataset


class BowlDataModule(LightningDataModule):
    """`LightningDataModule` for the Bowl dataset using COCO format."""
    def __init__(
        self,
        data_dir: str = "data/Bowl.v1i.coco/",
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `BowlDataModule`.

        Args:
            data_dir: The root directory of the COCO-formatted dataset. 
                      Defaults to `"data/Bowl.v1i.coco/"`.
            batch_size: The batch size. Defaults to `4`.
            num_workers: The number of workers. Defaults to `0`.
            pin_memory: Whether to pin memory. Defaults to `False`.
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

        self.train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomHorizontalFlip(),

        ])
        
        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of object classes.
        This should correspond to the number of categories defined in your COCO annotations
        (excluding a potential background class if your model handles that separately)
        """
        return 2

    def prepare_data(self) -> None:
        """Verify that the dataset exists and is structured correctly (COCO format)."""
        if not os.path.exists(self.train_img_dir):
            raise FileNotFoundError(f"Training image directory {self.train_img_dir} does not exist")
        if not os.path.exists(self.train_ann_file):
            raise FileNotFoundError(f"Training annotation file {self.train_ann_file} does not exist")
        if not os.path.exists(self.val_img_dir):
            raise FileNotFoundError(f"Validation image directory {self.val_img_dir} does not exist")
        if not os.path.exists(self.val_ann_file):
            raise FileNotFoundError(f"Validation annotation file {self.val_ann_file} does not exist")
        if not os.path.exists(self.test_img_dir):
            raise FileNotFoundError(f"Test image directory {self.test_img_dir} does not exist")
        if not os.path.exists(self.test_ann_file):
            raise FileNotFoundError(f"Test annotation file {self.test_ann_file} does not exist")

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data using CocoBowlDataset.

        Set variables: `self.data_train`, `self.data_test`.
        """
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the "
                    f"number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

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
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self._collate_fn,
        )
    
    def _collate_fn(self, batch):
        """Custom collate function for object detection batches.
        Images are stacked. Targets (dictionaries) are collected in a list.
        This is a common way to batch object detection data.
        """
        images = []
        targets = []
        
        for image, target in batch:
            images.append(image)
            targets.append(target)
            
        images = torch.stack(images, dim=0)
        
        return images, targets

    def predict_dataloader(self) -> DataLoader[Any]:
        """Return the same as test_dataloader for predictions."""
        return self.test_dataloader()


if __name__ == "__main__":
    _ = BowlDataModule()