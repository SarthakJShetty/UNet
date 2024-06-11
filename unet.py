from collections import OrderedDict

import torch
from ipdb import set_trace


class ContractingBranch(torch.nn.Module):
    def __init__(
        self,
        out_channels=[3, 64, 128, 256, 512],
    ):
        super().__init__()
        self._blocks_list = [
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=out_channels[i], out_channels=out_channels[i + 1], kernel_size=(3, 3)),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=out_channels[i + 1], out_channels=out_channels[i + 1], kernel_size=(3, 3)),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2), 2),
            )
            for i in range(0, len(out_channels) - 1)
        ]
        self._contracting_branch = torch.nn.Sequential(*self._blocks_list)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self._contracting_branch(image)


class BottleNeck(ContractingBranch):
    def __init__(self, in_channels=[512, 1024, 1024]):
        super().__init__()
        self._blocks_list = [
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_channels[i], out_channels=in_channels[i + 1], kernel_size=(3, 3)), torch.nn.ReLU()
            )
            for i in range(0, len(in_channels) - 1)
        ]
        self._bottleneck = torch.nn.Sequential(*self._blocks_list)

    def forward(self, contracting_block_latent: torch.Tensor) -> torch.Tensor:
        return self._bottleneck(contracting_block_latent)


class ExpandingBranch(torch.nn.Module):
    def __init__(
        self,
        in_channels=[1024, 512, 256, 128, 64],
    ):
        super().__init__()
        self._blocks_list = [
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=in_channels[i], out_channels=in_channels[i + 1], kernel_size=(2, 2), stride=2),
                torch.nn.Conv2d(in_channels=in_channels[i + 1], out_channels=in_channels[i + 1], kernel_size=(3, 3)),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=in_channels[i + 1], out_channels=in_channels[i + 1], kernel_size=(3, 3)),
                torch.nn.ReLU(),
            )
            for i in range(len(in_channels) - 1)
        ]
        self._expanding_branch = torch.nn.Sequential(*self._blocks_list)

    def forward(self, bottleneck_latent: torch.Tensor) -> torch.Tensor:
        _output = self._expanding_branch(bottleneck_latent)
        set_trace()


class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._contracting_branch = ContractingBranch()
        self._bottleneck = BottleNeck()
        self._expanding_branch = ExpandingBranch()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        _contracting_block_latent = self._contracting_branch(image)
        _bottleneck_latent = self._bottleneck(_contracting_block_latent)
        _expanding_block_output = self._expanding_branch(_bottleneck_latent)


if __name__ == "__main__":
    x = torch.randn(1, 3, 572, 572)
    model = UNet()
    model(x)
