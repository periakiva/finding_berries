import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import morphology as morph
from scipy import ndimage
import matplotlib.pyplot as plt
from torchvision import transforms, datasets, models



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1,x2], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



from random import random
class UNetRefined(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetRefined, self).__init__()
        # self.base_model = models.resnet50(pretrained=True)
        # self.base_layers = list(self.base_model.children())  
        # print(self.base_layers)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.down5 = Down(512, 512)
        self.down6 = Down(512, 512)
        self.down7 = Down(512, 512)
        self.down8 = Down(512, 512)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(1024, 512, bilinear)
        self.up3 = Up(1024, 512, bilinear)
        self.up4 = Up(1024, 512, bilinear)
        self.up5 = Up(1024, 256, bilinear)
        self.up6 = Up(512, 128, bilinear)
        self.up7 = Up(256, 64, bilinear)
        self.up8 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)


        self.out_nonlin = nn.Sigmoid()

        self.count_branch = nn.Sequential(nn.Linear(1*2*512,64),
                                        nn.LeakyReLU(inplace=True),
                                        nn.Dropout(p=0.5))
        
        self.branch_2 = nn.Sequential(nn.Linear(456*608*n_classes,64),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(p=0.5))

        self.regressor = nn.Sequential(nn.Linear(64+64,1),
                                        nn.PReLU())
                                        
        

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)
        x9 = self.down8(x8)
        x = self.up1(x9, x8)
        x = self.up2(x, x7)
        x = self.up3(x, x6)
        x = self.up4(x, x5)
        x = self.up5(x, x4)
        x = self.up6(x, x3)
        x = self.up7(x, x2)
        x = self.up8(x, x1)
        logits = self.outc(x)

        x_nonlin = self.out_nonlin(logits)
        x_flat = x_nonlin.view(1,-1)
        # print(x9.shape)
        mid_layer_flat = x9.view(1,-1)
        lateral_flat = self.count_branch(mid_layer_flat)
        x_flat = self.branch_2(x_flat)
        regression_features = torch.cat((x_flat,lateral_flat),dim=1)
        regression = self.regressor(regression_features)
        # regression = self.regressor(lateral_flat)
        return logits, regression



        

    