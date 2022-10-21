# Isola, P., Zhu, J.Y., Zhou, T. and Efros, A.A., 2016. Image-to-image translation with conditional adversarial
# networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1125-1134).

import torch
import torch.nn as nn
import torchvision.transforms as transforms


class SinoUNet(nn.Module):
    """Architecture inspired by the image-to-image U-Net in (Isola et al. 2016).
        - Down-Convs have Leaky ReLus with slope = 0.2
        - Up-Convs have non-leaky (i.e. normal) ReLus
        - No pooling layers of any kind
        - Batch Norm on every layer apart from the first & last
        - Dropout in the first 3 layers of the decoder
        - Final activation function is Tanh

    Code inspired by the accompanying GitHub repos:
        - https://github.com/phillipi/pix2pix
        - https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix"""

    def __init__(self):
        # Input (1, 256, 256) -> Output (1, 402, 362)
        super(SinoUNet, self).__init__()

        filters = 64

        # Input (1, 402, 362) -> Output(32, 256, 256)
        self.down0 = transforms.CenterCrop(256)  # would rather use convolution but that didn't work idk why

        # Input (1, 256, 256) -> Output (64, 128, 128)
        self.down1 = nn.Conv2d(1, filters, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))

        # Input (64, 128, 128) -> Output (128, 64, 64)
        self.down2 = self.down(filters, filters*2)

        # Input (128, 64, 64) -> Output (256, 32, 32)
        self.down3 = self.down(filters*2, filters*4)

        # Input (256, 32, 32) -> Output (512, 16, 16)
        self.down4 = self.down(filters*4, filters*8)

        # Input (512, 16, 16) -> Output (512, 8, 8)
        self.down5 = self.down(filters*8, filters*8)

        # Input (512, 8, 8) -> Output (512, 4, 4)
        self.down6 = self.down(filters*8, filters*8)

        # Input (512, 4, 4) -> Output (512, 2, 2)
        self.down7 = self.down(filters*8, filters*8)

        # Input (512, 2, 2) -> Output (512, 1, 1)
        self.down8 = self.down(filters*8, filters*8, batchNorm=False)

        # Input (512, 1, 1) -> Output (512, 2, 2)
        self.up1 = self.up(filters*8, filters*8, dropout=True)

        # Input (1024, 2, 2) -> Output (512, 4, 4)
        # Input channels double due to skip connections
        self.up2 = self.up(filters*8 * 2, filters*8, dropout=True)

        # Input (1024, 4, 4) -> Output (512, 8, 8)
        self.up3 = self.up(filters*8 * 2, filters*8, dropout=True)

        # Input (1024, 8, 8) -> Output (512, 16, 16)
        self.up4 = self.up(filters*8 * 2, filters*8)

        # Input (1024, 16, 16) -> Output (256, 32, 32)
        self.up5 = self.up(filters*8 * 2, filters*4)

        # Input (512, 32, 32) -> Output (128, 64, 64)
        self.up6 = self.up(filters*4 * 2, filters*2)

        # Input (256, 64, 64) -> Output (64, 128, 128)
        self.up7 = self.up(filters*2 * 2, filters)

        # Input (128, 128, 128) -> Output (1, 256, 256)
        self.up8 = self.up(filters * 2, 1)

        # Input (1, 256, 256) -> (1, 402, 362)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride=(2, 2), padding=(56, 76)),  # stupid
            nn.Tanh()
        )

    @staticmethod
    def down(in_c, out_c, batchNorm=True):
        if batchNorm:
            batchNorm = nn.BatchNorm2d(out_c, eps=0.001)
        else:
            batchNorm = nn.Identity()
        return nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_c, out_c, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            batchNorm,
        )

    @staticmethod
    def up(in_c, out_c, dropout=False):
        if dropout:
            dropout = nn.Dropout(0.5)
        else:
            dropout = nn.Identity()
        return nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_c, out_c, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_c, eps=0.001),
            dropout
        )

    def forward(self, x):

        # Downsampling
        down0_out = self.down0(x)
        down1_out = self.down1(down0_out)
        down2_out = self.down2(down1_out)
        down3_out = self.down3(down2_out)
        down4_out = self.down4(down3_out)
        down5_out = self.down5(down4_out)
        down6_out = self.down6(down5_out)
        down7_out = self.down7(down6_out)
        down8_out = self.down8(down7_out)

        # Bottom layer
        up1_out = self.up1(down8_out)

        # Upsampling
        up2_in = torch.cat((up1_out, down7_out), dim=1)
        up2_out = self.up2(up2_in)

        up3_in = torch.cat((up2_out, down6_out), dim=1)
        up3_out = self.up3(up3_in)

        up4_in = torch.cat((up3_out, down5_out), dim=1)
        up4_out = self.up4(up4_in)

        up5_in = torch.cat((up4_out, down4_out), dim=1)
        up5_out = self.up5(up5_in)

        up6_in = torch.cat((up5_out, down3_out), dim=1)
        up6_out = self.up6(up6_in)

        up7_in = torch.cat((up6_out, down2_out), dim=1)
        up7_out = self.up7(up7_in)

        up8_in = torch.cat((up7_out, down1_out), dim=1)
        up8_out = self.up8(up8_in)

        final_out = self.final(up8_out)

        return final_out
