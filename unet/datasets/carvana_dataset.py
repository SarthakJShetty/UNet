from glob import glob
from typing import List, Optional, Tuple

import torch
from torchvision.io import read_image
from torchvision.transforms.v2 import Compose, ToDtype, ToImage


class CarvanaDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_location: str = "dataset/", images: Optional[List[str]] = None, targets: Optional[List[str]] = None):
        self._dataset_location = dataset_location
        self._transforms = Compose([ToImage(), ToDtype(torch.float32, scale=True)])
        self._images = images if images is not None else glob(self._dataset_location + "images/*.jpg")
        self._targets = (
            targets if targets is not None else [image.replace("images", "labels").replace(".jpg", "_mask.jpg") for image in self._images]
        )

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._transforms(read_image(self._images[index])), self._transforms(read_image(self._targets[index]))[:1, :, :]

    def split_dataset(self, split_ratio: float) -> Tuple["CarvanaDataset", "CarvanaDataset"]:
        return CarvanaDataset(
            images=self._images[: int(len(self._images) * split_ratio)], targets=self._targets[: int(len(self._targets) * split_ratio)]
        ), CarvanaDataset(
            images=self._images[int(len(self._images) * split_ratio) :],
            targets=self._targets[int(len(self._targets) * split_ratio) :],
        )
