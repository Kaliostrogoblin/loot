import torch 
import torch.nn as nn
import torch.nn.functional as F

from src.conv_coord import AddCoords


class UnetLoot(nn.Module):
    def __init__(self, n_stations, radius_channel=False):
        super(UnetLoot, self).__init__()

        if n_stations < 3:
            raise ValueError("Only detector configuration "
                             "with 3 or more stations is supported")

        self.n_stations = n_stations
        self.radius_channel = radius_channel
        self.add_coords = AddCoords(self.radius_channel)
        # 2 coords - X and Y
        in_channels = self.n_stations + 2
        # add radius channel
        if self.radius_channel:
            in_channels += 1
        
        self.inconv = InConv(in_channels, 32)
        self.down1 = DownScale(32, 64)
        self.down2 = DownScale(64, 128)
        self.down3 = DownScale(128, 256)
        self.down4 = DownScale(256, 512)
        self.down5 = DownScale(512, 512)
        self.up1 = UpScale(1024, 256)
        self.up2 = UpScale(512, 128)
        self.up3 = UpScale(256, 64)
        self.up4 = UpScale(128, 32)
        self.up5 = UpScale(64, 32)
        # output (all shifts for x and y and probabilities)
        self.out_channels = (self.n_stations-1)*2+1
        self.out = OutConv(32, self.out_channels)

    def forward(self, x):
        x_in = self.add_coords(x)
        # encoder
        x1 = self.inconv(x_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        # decoder with skip-connections
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        # output
        x = self.out(x)
        return x


class X2Conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(X2Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = X2Conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownScale(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownScale, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            X2Conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class UpScale(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpScale, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.conv = X2Conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        # split on probabilities and shifts
        probs = torch.sigmoid(x[:, :1])
        shifts = torch.tanh(x[:, 1:])
        return torch.cat([probs, shifts], 1)


if __name__ == "__main__":
    t_in = torch.rand(1, 6, 512, 256)
    unet_loot = UnetLoot(t_in.size(1))
    print(unet_loot)
    out = unet_loot(t_in)
    print(out.shape)