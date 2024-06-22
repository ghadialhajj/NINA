import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)


class Encoder(nn.Module):
    def __init__(self, encoder, *, pretrained=False):
        super().__init__()
        self.encoder = encoder(pretrained=pretrained)
        self.encoder_layers = list(self.encoder.children())
        self.block1 = nn.Sequential(*self.encoder_layers[:3])
        self.block2 = nn.Sequential(*self.encoder_layers[3:5])
        self.block3 = self.encoder_layers[5]
        self.block4 = self.encoder_layers[6]
        self.block5 = self.encoder_layers[7]

        if not pretrained:
            self._weights_init()

    def _weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        return [block1, block2, block3, block4, block5]


class Decoder(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()
        self.up_conv6 = up_conv(512, 512)
        self.conv6 = double_conv(768, 512)
        self.up_conv7 = up_conv(512, 256)
        self.conv7 = double_conv(384, 256)
        self.up_conv8 = up_conv(256, 128)
        self.conv8 = double_conv(192, 128)
        self.up_conv9 = up_conv(128, 64)
        self.conv9 = double_conv(128, 64)
        self.up_conv10 = up_conv(64, 32)
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        self._weights_init()

    def _weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, encoder_blocks):
        block5 = encoder_blocks[-1]
        block4 = encoder_blocks[-2]
        block3 = encoder_blocks[-3]
        block2 = encoder_blocks[-4]
        block1 = encoder_blocks[0]

        up6 = self.up_conv6(block5)
        conv6 = self.conv6(torch.cat([up6, block4], dim=1))

        up7 = self.up_conv7(conv6)
        conv7 = self.conv7(torch.cat([up7, block3], dim=1))

        up8 = self.up_conv8(conv7)
        conv8 = self.conv8(torch.cat([up8, block2], dim=1))

        up9 = self.up_conv9(conv8)
        conv9 = self.conv9(torch.cat([up9, block1], dim=1))

        up10 = self.up_conv10(conv9)
        output = self.final_conv(up10)

        return output


class UNet(nn.Module):
    def __init__(self, encoder, pretrained=True, out_channels=1):
        super().__init__()
        self.encoder = Encoder(encoder, pretrained=pretrained)
        self.decoder = Decoder(out_channels=out_channels)

    def forward(self, x):
        encoder_blocks = self.encoder(x)
        output = self.decoder(encoder_blocks)
        return output
