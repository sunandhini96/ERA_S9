import torch
import torchvision
import torchvision.transforms as transforms
import albumentations as A
import torch.nn.functional as F
import torch.nn as nn

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, padding = "same",stride=1, dilation=1,bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, stride=stride,padding=padding,dilation=dilation, bias=bias,groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1,padding=0,bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


dropout_value = 0.1


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # CONVOLUTION BLOCK 1
        self.convblock1 = nn.Sequential(
            depthwise_separable_conv(nin=3, nout=32, kernel_size = 3, padding = "same",stride=1, dilation=1,bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1)
        ) # output_size = 32 , RF =  3, J_out=1

      # (here we are applying depthwise separable convolution to reduce computation)
        self.convblock2 = nn.Sequential(
            depthwise_separable_conv(nin=32, nout=64, kernel_size = 3, padding = "same",stride=1, dilation=1,bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.05)
        ) # output_size = 32 , RF = 5 , j_out= 1

        # Transition block 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=1,stride=2,dilation=2, bias=False)) # output_size = 16 , RF = 7 , j_out= 2


        # Convolution BLOCK 2 (applied dilation rate 2 to increase the field of view)
        self.convblock4 = nn.Sequential(
            depthwise_separable_conv(nin=32, nout=64, kernel_size = 3, padding = "same",stride=1, dilation=1,bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.05), # output size =14 , RF = 15 ,j_out=2
        )

        self.convblock5 = nn.Sequential(
            #nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3),padding=1,bias=False),
            depthwise_separable_conv(nin=64, nout=64, kernel_size = 3, padding = "same",stride=1, dilation=1,bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.05)
        ) # output_size = 14 , RF = 19 , j_out= 2

        # Transition block 2
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=1,stride=2, bias=False)) # output_size = 14 , RF = 23 , j_out= 4



        # CONVOLUTION BLOCK 3
        self.convblock7 = nn.Sequential(
            depthwise_separable_conv(nin=32, nout=64, kernel_size = 3, padding = "same",stride=1, dilation=1,bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.05), # output size =7 , RF = 31 ,j_out=4
        )

        self.convblock8 = nn.Sequential(
            depthwise_separable_conv(nin=64, nout=64, kernel_size = 3, padding = "same",stride=1, dilation=1,bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.05)
        ) # output_size = 7 , RF = 39 , j_out= 4

        # Transition block 3
        self.convblock9 = nn.Sequential(
            depthwise_separable_conv(nin=64, nout=64, kernel_size = 3, padding = "same",stride=1, dilation=1,bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.05)
        ) # output_size = 5 , RF = 47 , j_out= 4



        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)
        ) # output_size = 1

        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        )# output_size = 1 , RF = 47 , j_out= 4


        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.gap(x)
        x = self.convblock10(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
