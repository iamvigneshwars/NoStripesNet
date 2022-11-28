# Isola, P., Zhu, J.Y., Zhou, T. and Efros, A.A., 2016. Image-to-image translation with conditional adversarial
# networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1125-1134).

import torch
import torch.nn as nn


class BaseUNet(nn.Module):
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
        super(BaseUNet, self).__init__()

        filters = 64

        # Input (1, 402, 362) -> Output (64, 200, 180)
        self.down1 = nn.Conv2d(1, filters, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0), bias=False)

        # Input (64, 200, 180) -> Output (128, 99, 89)
        self.down2 = self.down(filters, filters * 2)

        # Input (128, 99, 89) -> Output (256, 48, 43)
        self.down3 = self.down(filters * 2, filters * 4)

        # Input (256, 48, 43) -> Output (512, 23, 20)
        self.down4 = self.down(filters * 4, filters * 8)

        # Input (512, 23, 20) -> Output (512, 10, 9)
        self.down5 = self.down(filters * 8, filters * 8)

        # Input (512, 10, 9) -> Output (512, 4, 4)
        self.down6 = self.down(filters * 8, filters * 8, p=(0, 1))

        # Input (512, 4, 4) -> Output (512, 1, 1)
        self.down7 = self.down(filters * 8, filters * 8, batchNorm=False)

        # Input (512, 1, 1) -> Output (512, 4, 4)
        self.up1 = self.up(filters*8, filters*8, dropout=True)

        # Input (1024, 4, 4) -> Output (512, 10, 9)
        # Input channels double due to skip connections
        self.up2 = self.up(filters*8 * 2, filters*8, dropout=True, k=(4, 3))

        # Input (1024, 10, 9) -> Output (512, 23, 20)
        self.up3 = self.up(filters*8 * 2, filters*8, dropout=True, k=(5, 4))

        # Input (1024, 23, 20) -> Output (256, 48, 43)
        self.up4 = self.up(filters*8 * 2, filters*4, k=(4, 5))

        # Input (512, 48, 43) -> Output (128, 99, 89)
        self.up5 = self.up(filters*4 * 2, filters * 2, k=(5, 5))

        # Input (256, 99, 89) -> Output (64, 200, 180)
        self.up6 = self.up(filters*2 * 2, filters)

        # Input (128, 200, 180) -> Output (1, 402, 362)
        self.up7 = self.up(filters * 2, 1, batchNorm=False)

        # Input (1, 402, 362) -> (1, 402, 362)
        self.final = nn.Sequential(
            nn.Tanh()
        )

    @staticmethod
    def down(in_c, out_c, batchNorm=True, k=(4, 4), s=(2, 2), p=(0, 0)):
        if batchNorm:
            batchNorm = nn.BatchNorm2d(out_c, eps=0.001, track_running_stats=False)
        else:
            batchNorm = nn.Identity()
        return nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
            batchNorm
        )

    @staticmethod
    def up(in_c, out_c, batchNorm=True, dropout=False, k=(4, 4), s=(2, 2), p=(0, 0)):
        if batchNorm:
            batchNorm = nn.BatchNorm2d(out_c, eps=0.001, track_running_stats=False)
        else:
            batchNorm = nn.Identity()
        if dropout:
            dropout = nn.Dropout(0.5)
        else:
            dropout = nn.Identity()
        return nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
            batchNorm,
            dropout
        )

    def forward(self, x):
        # Downsampling
        down1_out = self.down1(x)
        down2_out = self.down2(down1_out)
        down3_out = self.down3(down2_out)
        down4_out = self.down4(down3_out)
        down5_out = self.down5(down4_out)
        down6_out = self.down6(down5_out)
        down7_out = self.down7(down6_out)

        # Bottom layer
        up1_out = self.up1(down7_out)

        # Upsampling
        up2_in = torch.cat((up1_out, down6_out), dim=1)
        up2_out = self.up2(up2_in)

        up3_in = torch.cat((up2_out, down5_out), dim=1)
        up3_out = self.up3(up3_in)

        up4_in = torch.cat((up3_out, down4_out), dim=1)
        up4_out = self.up4(up4_in)

        up5_in = torch.cat((up4_out, down3_out), dim=1)
        up5_out = self.up5(up5_in)

        up6_in = torch.cat((up5_out, down2_out), dim=1)
        up6_out = self.up6(up6_in)

        up7_in = torch.cat((up6_out, down1_out), dim=1)
        up7_out = self.up7(up7_in)

        final_out = self.final(up7_out)
        return final_out


class WindowUNet(nn.Module):
    def __init__(self):
        super(WindowUNet, self).__init__()

        filters = 32

        # Input (1, 402, 25) -> Output (32, 402, 22)
        self.down1 = nn.Conv2d(1, filters, kernel_size=(1, 4), stride=(1, 1), padding=(0, 0))

        # Input (32, 402, 22) -> Output (64, 200, 19)
        self.down2 = self.down(filters, filters*2)

        # Input (64, 200, 19) -> Output (128, 99, 16)
        self.down3 = self.down(filters*2, filters*4)

        # Input (128, 99, 16) -> Output (256, 48, 13)
        self.down4 = self.down(filters*4, filters*8)

        # Input (256, 48, 13) -> Output (512, 23, 10)
        self.down5 = self.down(filters*8, filters*16)

        # Input (512, 23, 10) -> Output (512, 10, 7)
        self.down6 = self.down(filters*16, filters*16)

        # Input (512, 10, 7) -> Output (512, 4, 4)
        self.down7 = self.down(filters*16, filters*16)

        # Input (512, 4, 4) -> Output (512, 1, 1)
        self.down8 = self.down(filters*16, filters*16, batchNorm=False)

        # Input (512, 1, 1) -> Output (512, 4, 4)
        self.up1 = self.up(filters*16, filters*16, dropout=True)

        # Input (1024, 4, 4) -> Output (512, 10, 7)
        # Input channels double due to skip connections
        self.up2 = self.up(filters*16 * 2, filters*16, dropout=True)

        # Input (1024, 10, 7) -> Output (512, 23, 10)
        self.up3 = self.up(filters*16 * 2, filters*16, dropout=True, k=(5, 4))

        # Input (1024, 23, 10) -> Output (256, 48, 13)
        self.up4 = self.up(filters*16 * 2, filters*8)

        # Input (512, 48, 13) -> Output (128, 99, 16)
        self.up5 = self.up(filters*8 * 2, filters * 4, k=(5, 4))

        # Input (256, 99, 16) -> Output (64, 200, 19)
        self.up6 = self.up(filters*4 * 2, filters * 2)

        # Input (128, 402, 19) -> Output (32, 402, 22)
        self.up7 = self.up(filters*2 * 2, filters)

        # Input (64, 402, 25) -> Output (1, 402, 25)
        self.up8 = self.up(filters * 2, 1, k=(1, 4), s=(1, 1), batchNorm=False)

        # Input (1, 402, 25) -> (1, 402, 25)
        self.final = nn.Sequential(
            nn.Tanh()
        )

    @staticmethod
    def down(in_c, out_c, batchNorm=True, k=(4, 4), s=(2, 1), p=(0, 0)):
        if batchNorm:
            batchNorm = nn.BatchNorm2d(out_c, eps=0.001)
        else:
            batchNorm = nn.Identity()
        return nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p),
            batchNorm,
        )

    @staticmethod
    def up(in_c, out_c, batchNorm=True, dropout=False, k=(4, 4), s=(2, 1), p=(0, 0)):
        if batchNorm:
            batchNorm = nn.BatchNorm2d(out_c, eps=0.001, track_running_stats=False)
        else:
            batchNorm = nn.Identity()
        if dropout:
            dropout = nn.Dropout(0.5)
        else:
            dropout = nn.Identity()
        return nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_c, out_c, kernel_size=k, stride=s, padding=p),
            batchNorm,
            dropout
        )

    def forward(self, x):
        if x.shape[-1] != 25:
            down0_out = nn.ReplicationPad2d((0, 25 - x.shape[-1], 0, 0))(x)
        else:
            down0_out = nn.Identity()(x)
        # Downsampling
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


