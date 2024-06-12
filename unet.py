from collections import OrderedDict
from glob import glob
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
from torchvision.io import read_image
from torchvision.transforms.v2 import Compose, ToDtype, ToImage
from tqdm import tqdm


class ContractingBranch(torch.nn.Module):
    def __init__(
        self,
        out_channels=[3, 64, 128, 256, 512],
    ):
        super().__init__()
        self._blocks_list = [
            torch.nn.Sequential(
                OrderedDict(
                    [
                        (
                            f"contracting_layers_{i}",
                            torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=out_channels[i], out_channels=out_channels[i + 1], kernel_size=(3, 3)),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=out_channels[i + 1], out_channels=out_channels[i + 1], kernel_size=(3, 3)),
                                torch.nn.ReLU(),
                            ),
                        ),
                        (
                            f"max_pool_{i}",
                            torch.nn.MaxPool2d((2, 2), 2),
                        ),
                    ]
                )
            )
            for i in range(0, len(out_channels) - 1)
        ]

        self._contracting_branch = torch.nn.Sequential(*self._blocks_list)

    def forward(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        _contracting_block_latents = []
        for _contracting_block in self._contracting_branch:

            _downsampled_features = _contracting_block[0](input_tensor)
            input_tensor = _contracting_block[1](_downsampled_features)

            _contracting_block_latents.append(_downsampled_features)
        return input_tensor, _contracting_block_latents


class BottleNeck(torch.nn.Module):
    def __init__(self, in_channels=[512, 1024, 1024]):
        super().__init__()
        self._blocks_list = [
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_channels[i], out_channels=in_channels[i + 1], kernel_size=(3, 3)), torch.nn.ReLU()
            )
            for i in range(0, len(in_channels) - 1)
        ]
        self._bottleneck = torch.nn.Sequential(*self._blocks_list)

    def forward(self, contracting_branch_latent: torch.Tensor) -> torch.Tensor:
        return self._bottleneck(contracting_branch_latent)


class ExpandingBranch(torch.nn.Module):
    def __init__(
        self,
        in_channels=[1024, 512, 256, 128, 64],
    ):
        super().__init__()
        self._blocks_list = [
            torch.nn.Sequential(
                OrderedDict(
                    [
                        (
                            f"upsampling_conv_transpose_{i}",
                            torch.nn.ConvTranspose2d(
                                in_channels=in_channels[i], out_channels=in_channels[i + 1], kernel_size=(2, 2), stride=2
                            ),
                        ),
                        (
                            f"conv_layers_{i}",
                            torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=in_channels[i], out_channels=in_channels[i + 1], kernel_size=(3, 3)),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=in_channels[i + 1], out_channels=in_channels[i + 1], kernel_size=(3, 3)),
                                torch.nn.ReLU(),
                            ),
                        ),
                    ]
                )
            )
            for i in range(len(in_channels) - 1)
        ]

        self._expanding_branch = torch.nn.Sequential(*self._blocks_list)

    def forward(self, latent: torch.Tensor, contracting_branch_latents: torch.Tensor) -> torch.Tensor:
        for idx in range(len(self._expanding_branch)):
            _transpose_output = self._expanding_branch[idx][0](latent)
            _contracting_branch_skip_connection = contracting_branch_latents[::-1][idx][
                :,
                :,
                (contracting_branch_latents[::-1][idx].shape[-2] - _transpose_output.shape[-2])
                // 2 : (contracting_branch_latents[::-1][idx].shape[-2] - _transpose_output.shape[-2])
                // 2
                + _transpose_output.shape[-2],
                (contracting_branch_latents[::-1][idx].shape[-1] - _transpose_output.shape[-1])
                // 2 : (contracting_branch_latents[::-1][idx].shape[-1] - _transpose_output.shape[-1])
                // 2
                + _transpose_output.shape[-1],
            ]
            latent = self._expanding_branch[idx][1](torch.hstack([_contracting_branch_skip_connection, _transpose_output]))
        return latent


class SegmentationHead(torch.nn.Module):
    def __init__(self, in_channels=64, out_channels=1):
        super().__init__()
        self._segmentation_head = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1))
        )

    def forward(self, expanding_block_latent: torch.Tensor) -> torch.Tensor:
        return self._segmentation_head(expanding_block_latent)


class UpsamplingLayer(torch.nn.Module):
    def __init__(self, up_sampling_size: Tuple[int, int] = (1280, 1918)):
        super().__init__()
        self._upsampling_layer = torch.nn.Upsample(size=up_sampling_size)

    def forward(self, expanding_block_latent: torch.Tensor):
        return self._upsampling_layer(expanding_block_latent)


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


class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._contracting_branch = ContractingBranch()
        self._bottleneck = BottleNeck()
        self._expanding_branch = ExpandingBranch()
        self._upsampling_layer = UpsamplingLayer()
        self._segmentation_head = SegmentationHead()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        _contracting_block_latent, _contracting_block_latents = self._contracting_branch(image)
        _bottleneck_latent = self._bottleneck(_contracting_block_latent)
        _expanding_block_latent = self._expanding_branch(_bottleneck_latent, _contracting_block_latents)
        _upsampled_expanding_block_latent = self._upsampling_layer(_expanding_block_latent)
        _segmentation_map = self._segmentation_head(_upsampled_expanding_block_latent)
        return _segmentation_map


if __name__ == "__main__":
    dataset = CarvanaDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    model = UNet()
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-4)

    model.train()

    epochs = 100
    batch_loss = 0

    for epoch in range(epochs):

        for image, label in tqdm(dataloader):
            optimizer.zero_grad()

            prediction = model(image)
            loss = loss_function(prediction, label)

            batch_loss += loss
            loss.backward()
            optimizer.step()
        _image_label_prediction = (
            torch.cat(
                [image[0], label[0].repeat(3, 1, 1), torch.nn.functional.sigmoid(prediction[0].detach().to("cpu")).repeat(3, 1, 1)],
                dim=-1,
            )
            .permute(1, 2, 0)
            .numpy()
        )
        plt.figure(figsize=(3, 1))
        plt.imshow(_image_label_prediction)
        plt.axis("off")
        plt.savefig(f"prediction_{epoch}.png")
        plt.close()
        print(f"Epoch loss: {batch_loss/len(dataloader)}")
