import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Tuple, Optional

class VitHead(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 deconv_out_channels: Optional[Sequence[int]] = (256, 256, 256),
                 deconv_kernel_sizes: Optional[Sequence[int]] = (4, 4, 4),
                 conv_out_channels: Optional[Sequence[int]] = None,
                 conv_kernel_sizes: Optional[Sequence[int]] = None,
                 ):
        super(VitHead, self).__init__()

        if deconv_out_channels:
            if deconv_kernel_sizes is None or len(deconv_out_channels) != len(deconv_kernel_sizes):
                raise ValueError(
                    '"deconv_out_channels" and "deconv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {deconv_out_channels} and '
                    f'{deconv_kernel_sizes}')

            self.deconv_layers = self._make_deconv_layers(
                in_channels=in_channels,
                layer_out_channels=deconv_out_channels,
                layer_kernel_sizes=deconv_kernel_sizes,
            )
            in_channels = deconv_out_channels[-1]
        else:
            self.deconv_layers = nn.Identity()

        if conv_out_channels:
            if conv_kernel_sizes is None or len(conv_out_channels) != len(conv_kernel_sizes):
                raise ValueError(
                    '"conv_out_channels" and "conv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {conv_out_channels} and '
                    f'{conv_kernel_sizes}')

            self.conv_layers = self._make_conv_layers(
                in_channels=in_channels,
                layer_out_channels=conv_out_channels,
                layer_kernel_sizes=conv_kernel_sizes)
            in_channels = conv_out_channels[-1]
        else:
            self.conv_layers = nn.Identity()

        self.cls_seg = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def _make_conv_layers(self, in_channels: int,
                          layer_out_channels: Sequence[int],
                          layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create convolutional layers by given parameters."""
        layers = []
        for out_channels, kernel_size in zip(layer_out_channels, layer_kernel_sizes):
            padding = (kernel_size - 1) // 2
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding))
            layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.SiLU(inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def _make_deconv_layers(self, in_channels: int,
                            layer_out_channels: Sequence[int],
                            layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create deconvolutional layers by given parameters."""
        layers = []
        for out_channels, kernel_size in zip(layer_out_channels, layer_kernel_sizes):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
            else:
                raise ValueError(f'Unsupported kernel size {kernel_size} for deconvolutional layers')

            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding, output_padding=output_padding, bias=False))
            layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.SiLU(inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, inputs):
        x = self.deconv_layers(inputs)
        x = self.conv_layers(x)
        out = self.cls_seg(x)
        return out

if __name__ == "__main__":

    # Example usage:
    model = VitHead(in_channels=1536, out_channels=21, deconv_out_channels=(768, 768, 512, 512),
                    deconv_kernel_sizes=(4, 4, 4, 4),
                    conv_out_channels=(512, 256, 128), conv_kernel_sizes=(1, 1, 1),
                    )
    inputs = torch.randn(1, 1536, 64, 64)
    outputs = model(inputs)
    print(outputs.shape)