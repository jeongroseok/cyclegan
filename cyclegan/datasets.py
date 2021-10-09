from genericpath import sameopenfile
import os
from typing import Any, Callable, List, Optional, Tuple, TypeVar, cast

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import (
    IMG_EXTENSIONS,
    default_loader,
    has_file_allowed_extension,
)

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")


class RandomZippingDataset(Dataset[Tuple[T_co]]):
    def __init__(
        self, datasets: List[Dataset[T_co]], random_sample: bool = True
    ) -> None:
        super().__init__()
        self.datasets = datasets
        self.random_sample = random_sample

        self.dataset_len = len(self.datasets[0])

    def __getitem__(self, index: int) -> Tuple[T_co]:
        sample = tuple(
            dataset[torch.randint(high=len(dataset), size=(1,)).item()]
            for dataset in self.datasets
        )
        return sample

    def __len__(self) -> int:
        return self.dataset_len


def make_dataset(
    directory: str,
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[str]:
    directory = os.path.expanduser(directory)

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError(
            "Both extensions and is_valid_file cannot be None or not None at the same time"
        )

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if is_valid_file(path):
                instances.append(path)

    return instances


class ImageDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        max_samples: Optional[int] = None
    ) -> None:
        super().__init__()
        self.root = root
        self.transform = transform
        self.loader = loader
        self.max_samples = max_samples

        self.samples = make_dataset(
            root,
            IMG_EXTENSIONS if is_valid_file is None else None,
            is_valid_file=is_valid_file,
        )

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self) -> int:
        if self.max_samples is not None:
            return self.max_samples
        return len(self.samples)
