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


class MPNUNet(nn.Module):
    def __init__(self):
        super(MPNUNet, self).__init__()
        filters = 64

        self.down1 = self.down(1, filters)
        # Input (64, 200, 180) -> Output (128, 99, 89)
        self.down2 = self.down(filters+1, filters * 2)
        # Input (128, 99, 89) -> Output (256, 48, 43)
        self.down3 = self.down(filters * 2 + 1, filters * 4)
        # Input (256, 48, 43) -> Output (512, 23, 20)
        self.down4 = self.down(filters * 4 + 1, filters * 8)
        # Input (512, 23, 20) -> Output (512, 10, 9)
        self.down5 = self.down(filters * 8 + 1, filters * 8)
        # Input (512, 10, 9) -> Output (512, 4, 4)
        self.down6 = self.down(filters * 8 + 1, filters * 8, p=(0, 1))
        # Input (512, 4, 4) -> Output (512, 1, 1)
        self.down7 = self.down(filters * 8 + 1, filters * 8)

        self.mpn = [
            # Input (1, 402, 362) -> Output (64, 200, 180)
            self.mpn_layer(),
            # Input (64, 200, 180) -> Output (128, 99, 89)
            self.mpn_layer(),
            # Input (128, 99, 89) -> Output (256, 48, 43)
            self.mpn_layer(),
            # Input (256, 48, 43) -> Output (512, 23, 20)
            self.mpn_layer(),
            # Input (512, 23, 20) -> Output (512, 10, 9)
            self.mpn_layer(),
            # Input (512, 10, 9) -> Output (512, 4, 4)
            self.mpn_layer(p=(0, 1)),
            # Input (512, 4, 4) -> Output (512, 1, 1)
            self.mpn_layer()
        ]

        # Input (512, 1, 1) -> Output (512, 4, 4)
        self.up1 = self.up((4, 4), filters * 8 + 1, filters * 8)
        # Input (1024, 4, 4) -> Output (512, 10, 9)
        # Input channels double due to skip connections
        self.up2 = self.up((10, 9), filters * 8 * 2 + 1, filters * 8)
        # Input (1024, 10, 9) -> Output (512, 23, 20)
        self.up3 = self.up((23, 20), filters * 8 * 2 + 1, filters * 8)
        # Input (1024, 23, 20) -> Output (256, 48, 43)
        self.up4 = self.up((48, 43), filters * 8 * 2 + 1, filters * 4)
        # Input (512, 48, 43) -> Output (128, 99, 89)
        self.up5 = self.up((99, 89), filters * 4 * 2 + 1, filters * 2)
        # Input (256, 99, 89) -> Output (64, 200, 180)
        self.up6 = self.up((200, 180), filters * 2 * 2 + 1, filters)
        # Input (128, 200, 180) -> Output (1, 402, 362)
        self.up7 = self.up((402, 362), filters * 2 + 1, 1, batchNorm=False, final=True)

    @staticmethod
    def down(in_c, out_c, batchNorm=True, k=(4, 4), s=(2, 2), p=(0, 0)):
        if batchNorm:
            batchNorm = nn.BatchNorm2d(out_c, eps=0.001, track_running_stats=False)
        else:
            batchNorm = nn.Identity()
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
            batchNorm,
            nn.LeakyReLU(0.2, inplace=True),
        )

    @staticmethod
    def up(size, in_c, out_c, batchNorm=True, k=(3, 3), s=(1, 1), p=(1, 1), final=False):
        if batchNorm:
            batchNorm = nn.BatchNorm2d(out_c, eps=0.001, track_running_stats=False)
        else:
            batchNorm = nn.Identity()
        if final:
            activation = nn.Tanh()
        else:
            activation = nn.ReLU(inplace=True)
        return nn.Sequential(
            nn.Upsample(size=size, mode='bilinear'),
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
            batchNorm,
            activation,
        )

    @staticmethod
    def mpn_layer(k=(4, 4), s=(2, 2), p=(0, 0)):
        return nn.AvgPool2d(kernel_size=k, stride=s, padding=p)

    def forward(self, x):
        mask = x[:, 1].unsqueeze(dim=1)
        x = x[:, 0].unsqueeze(dim=1)

        # Downsampling
        down1_out = self.down1(x)
        mpn1_out = self.mpn[0](mask)

        down2_in = torch.cat((down1_out, mpn1_out), dim=1)
        mpn2_out = self.mpn[1](mpn1_out)
        down2_out = self.down2(down2_in)

        down3_in = torch.cat((down2_out, mpn2_out), dim=1)
        mpn3_out = self.mpn[2](mpn2_out)
        down3_out = self.down3(down3_in)

        down4_in = torch.cat((down3_out, mpn3_out), dim=1)
        mpn4_out = self.mpn[3](mpn3_out)
        down4_out = self.down4(down4_in)

        down5_in = torch.cat((down4_out, mpn4_out), dim=1)
        mpn5_out = self.mpn[4](mpn4_out)
        down5_out = self.down5(down5_in)

        down6_in = torch.cat((down5_out, mpn5_out), dim=1)
        mpn6_out = self.mpn[5](mpn5_out)
        down6_out = self.down6(down6_in)

        down7_in = torch.cat((down6_out, mpn6_out), dim=1)
        mpn7_out = self.mpn[6](mpn6_out)
        down7_out = self.down7(down7_in)

        # Bottom layer
        up1_in = torch.cat((down7_out, mpn7_out), dim=1)
        up1_out = self.up1(up1_in)

        # Upsampling
        up2_in = torch.cat((up1_out, down7_in), dim=1)
        up2_out = self.up2(up2_in)

        up3_in = torch.cat((up2_out, down6_in), dim=1)
        up3_out = self.up3(up3_in)

        up4_in = torch.cat((up3_out, down5_in), dim=1)
        up4_out = self.up4(up4_in)

        up5_in = torch.cat((up4_out, down4_in), dim=1)
        up5_out = self.up5(up5_in)

        up6_in = torch.cat((up5_out, down3_in), dim=1)
        up6_out = self.up6(up6_in)

        up7_in = torch.cat((up6_out, down2_in), dim=1)
        return self.up7(up7_in)


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


