
# Adapted from: https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/

import torch
from torchvision import transforms


class Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.double_conv2d = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv2d(x)


class Encoder(torch.nn.Module):
    def __init__(self, channels):
        super(Encoder, self).__init__()
        self.encoder_blocks = torch.nn.ModuleList([Block(channels[i], channels[i+1]) for i in range(len(channels)-1)])
        self.pool = torch.nn.MaxPool2d(2)

    def forward(self, x):
        block_outputs = []
        for block in self.encoder_blocks:
            x = block(x)
            block_outputs.append(x)
            x = self.pool(x)
        return block_outputs


class Decoder(torch.nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.up_convs = torch.nn.ModuleList([torch.nn.ConvTranspose2d(
            channels[i], channels[i+1], 2, 2) for i in range(len(channels)-1)])
        self.decoder_blocks = torch.nn.ModuleList(
            [Block(channels[i], channels[i+1]) for i in range(len(channels)-1)])

    def crop(self, encoder_features, x):
        (_, _, H, W) = x.shape
        return transforms.CenterCrop([H, W])(encoder_features)

    def forward(self, x, encoder_features):
        for i in range(len(self.up_convs)):
            x = self.up_convs[i](x)
            encoder_feature = self.crop(encoder_features[i], x)
            x = torch.cat([x, encoder_feature], dim=1)
            x = self.decoder_blocks[i](x)
        return x


class UNet(torch.nn.Module):
    def __str__(self) -> str:
        return 'UNet'

    def __init__(self, encoder_channels=(3, 16, 32, 64), decoder_channels=(64, 32, 16), classes=1, retain_dim=True, output_size=(512, 512)):
        super(UNet, self).__init__()
        self.encoder = Encoder(encoder_channels)
        self.decoder = Decoder(decoder_channels)
        self.head = torch.nn.Conv2d(decoder_channels[-1], classes, 1)
        self.retain_dim = retain_dim
        self.output_size = output_size

    def forward(self, x):
        encoder_features = self.encoder(x)
        decoder_features = self.decoder(encoder_features[::-1][0], encoder_features[::-1][1:])
        map = self.head(decoder_features)
        if self.retain_dim:
            map = torch.nn.functional.interpolate(map, self.output_size)
        return map
