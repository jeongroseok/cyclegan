import os
from typing import Optional, Union

from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pl_examples import _DATASETS_PATH
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as transform_lib

from ..datasets import ImageDataset, RandomZippingDataset


class Apple2OrangeDataModule(VisionDataModule):
    name = "apple2orange"
    dims = (3, 256, 256)

    def __init__(
        self,
        data_dir: Optional[str] = _DATASETS_PATH,
        val_split: Union[int, float] = 0.2,
        num_workers: int = 4,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        persistent_workers: bool = True,
        max_samples: Optional[int] = None,
        *args: any,
        **kwargs: any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.val_split = val_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers
        self.max_samples = max_samples

    def prepare_data(self, *args: any, **kwargs: any) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        dataset_dir = os.path.join(self.data_dir, self.name)
        if stage == "fit" or stage is None:
            train_transforms = (
                self.default_transforms()
                if self.train_transforms is None
                else self.train_transforms
            )
            val_transforms = (
                self.default_transforms()
                if self.val_transforms is None
                else self.val_transforms
            )

            a_train = ImageDataset(
                os.path.join(dataset_dir, "trainA"),
                train_transforms,
                max_samples=self.max_samples,
            )
            b_train = ImageDataset(
                os.path.join(dataset_dir, "trainB"),
                train_transforms,
                max_samples=self.max_samples,
            )
            self.dataset_train = RandomZippingDataset([a_train, b_train], True)

            a_val = ImageDataset(os.path.join(dataset_dir, "trainA"), val_transforms)
            b_val = ImageDataset(os.path.join(dataset_dir, "trainB"), val_transforms)
            self.dataset_val = RandomZippingDataset([a_val, b_val], True)

            # Split
            self.dataset_train = self._split_dataset(self.dataset_train)
            self.dataset_val = self._split_dataset(self.dataset_val, train=False)

        if stage == "test" or stage is None:
            test_transforms = (
                self.default_transforms()
                if self.test_transforms is None
                else self.test_transforms
            )

            a_test = ImageDataset(os.path.join(dataset_dir, "testA"), test_transforms)
            b_test = ImageDataset(os.path.join(dataset_dir, "testB"), test_transforms)
            self.dataset_test = RandomZippingDataset([a_test, b_test], True)

    def default_transforms(self) -> callable:
        return transform_lib.Compose(
            [
                transform_lib.ToTensor(),
                transform_lib.Normalize(mean=(0.5,), std=(0.5,)),
            ]
        )

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
            if self.num_workers > 0
            else False,
        )
