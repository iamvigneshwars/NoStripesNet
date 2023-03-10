from torch import cat
import torch.nn as nn


class ClusterUNet(nn.Module):
    def __init__(self):
        super(ClusterUNet, self).__init__()
        filters = 64

        # Input (1, 1801, 256) -> Output (64, 900, 128)
        self.down1 = self.down(1, filters, batchNorm=False)

        # Input (64, 900, 128) -> Output (128, 450, 64)
        self.down2 = self.down(filters, filters*2)

        # Input (128, 450, 64) -> Output (256, 225, 32)
        self.down3 = self.down(filters*2, filters*4)

        # Input (256, 225, 32) -> Output (512, 112, 16)
        self.down4 = self.down(filters*4, filters*8)

        # Input (512, 112, 16) -> Output (512, 56, 8)
        self.down5 = self.down(filters*8, filters*8)

        # Input (512, 56, 8) -> Output (512, 28, 4)
        self.down6 = self.down(filters*8, filters*8)

        # Input (512, 28, 4) -> Output (512, 14, 2)
        self.down7 = self.down(filters*8, filters*8)

        # Input (512, 14, 2) -> Output (512, 7, 1)
        self.down8 = nn.Sequential(
            nn.Conv2d(filters*8, filters*8, kernel_size=(4, 4), stride=(2, 2),
                      padding=(1, 1), bias=False),
            nn.ReLU(inplace=True)
        )

        ####### UP #######

        # Input (512, 7, 1) -> Output (512, 14, 2)
        self.up1 = self.up(filters*8, filters*8, dropout=True)

        # Input (1024, 14, 2) -> Output (512, 28, 4)
        self.up2 = self.up(filters*8 * 2, filters*8, dropout=True)

        # Input (1024, 28, 4) -> Output (512, 56, 8)
        self.up3 = self.up(filters*8 * 2, filters*8, dropout=True)

        # Input (1024, 56, 8) -> Output (512, 112, 16)
        self.up4 = self.up(filters*8 * 2, filters*8)

        # Input (1024, 112, 16) -> Output (256, 225, 32)
        self.up5 = self.up(filters*8 * 2, filters*4, k=(5, 4))

        # Input (512, 225, 32) -> Output (128, 450, 64)
        self.up6 = self.up(filters*4 * 2, filters*2)

        # Input (256, 450, 64) -> Output (64, 900, 128)
        self.up7 = self.up(filters*2 * 2, filters)

        # Input (128, 900, 128) -> Output (1, 1801, 256)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(filters * 2, 1, (5, 4), (2, 2), (1, 1)),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        down1_out = self.down1(x)
        down2_out = self.down2(down1_out)
        down3_out = self.down3(down2_out)
        down4_out = self.down4(down3_out)
        down5_out = self.down5(down4_out)
        down6_out = self.down6(down5_out)
        down7_out = self.down7(down6_out)
        down8_out = self.down8(down7_out)

        # Decoder
        up1_out = self.up1(down8_out)

        # Skip connections are concatenated
        up2_in = cat((up1_out, down7_out), dim=1)
        up2_out = self.up2(up2_in)

        up3_in = cat((up2_out, down6_out), dim=1)
        up3_out = self.up3(up3_in)

        up4_in = cat((up3_out, down5_out), dim=1)
        up4_out = self.up4(up4_in)

        up5_in = cat((up4_out, down4_out), dim=1)
        up5_out = self.up5(up5_in)

        up6_in = cat((up5_out, down3_out), dim=1)
        up6_out = self.up6(up6_in)

        up7_in = cat((up6_out, down2_out), dim=1)
        up7_out = self.up7(up7_in)

        up8_in = cat((up7_out, down1_out), dim=1)
        up8_out = self.up8(up8_in)

        return up8_out

    @staticmethod
    def down(in_c, out_c, batchNorm=True, k=(4, 4), s=(2, 2), p=(1, 1)):
        if batchNorm:
            batchNorm = nn.BatchNorm2d(out_c, eps=0.001,
                                       track_running_stats=False)
        else:
            batchNorm = nn.Identity()
        return nn.Sequential(
            nn.Conv2d(in_c, out_c,
                      kernel_size=k, stride=s,
                      padding=p, bias=False),
            batchNorm,
            nn.LeakyReLU(0.2, inplace=True)
        )

    @staticmethod
    def up(in_c, out_c, batchNorm=True, dropout=False, k=(4, 4), s=(2, 2),
           p=(1, 1)):
        if batchNorm:
            batchNorm = nn.BatchNorm2d(out_c, eps=0.001,
                                       track_running_stats=False)
        else:
            batchNorm = nn.Identity()
        if dropout:
            dropout = nn.Dropout(0.5)
        else:
            dropout = nn.Identity()
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c,
                               kernel_size=k, stride=s,
                               padding=p, bias=False),
            batchNorm,
            dropout,
            nn.ReLU(inplace=True),
        )


class ClusterDiscriminator(nn.Module):
    def __init__(self):
        super(ClusterDiscriminator, self).__init__()
        filters = 64
        self.down = ClusterUNet.down

        self.model = nn.Sequential(
            self.down(2, filters, batchNorm=False),
            self.down(filters, filters*2),
            self.down(filters*2, filters*4),
            self.down(filters*4, filters*8),
            self.down(filters*8, filters*8),
            self.down(filters*8, filters*8),
            self.down(filters*8, filters*8),
            nn.Conv2d(filters*8, 1, kernel_size=(4, 4), stride=(2, 2),
                      padding=(1, 1), bias=False)
        )

    def forward(self, x):
        return self.model(x)
