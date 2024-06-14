from glob import glob

import torch
from torchvision.io import read_image
from torchvision.transforms.v2 import Compose, ToDtype, ToImage


class CarvanaDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_location: str = "dataset/"):
        self._dataset_location = dataset_location
        self._images = glob(self._dataset_location + "images/*.jpg")
        self._targets = [image.replace("images", "labels").replace(".jpg", "_mask.jpg") for image in self._images]
        self._transforms = Compose([ToImage(), ToDtype(torch.float32, scale=True)])

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._transforms(read_image(self._images[index])), self._transforms(read_image(self._targets[index]))[:1, :, :]
