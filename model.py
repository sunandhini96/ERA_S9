import torch
import torchvision
import torchvision.transforms as transforms
import albumentations as A

import torch.nn as nn

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, padding = "same",stride=1, dilation=1,bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, stride=stride,padding=padding,dilation=dilation, bias=bias,groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1,bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

import torch.nn.functional as F
dropout_value = 0.1
        

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding="same", bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1)
        ) # output_size = 30 , RF =  5

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            depthwise_separable_conv(nin=32, nout=64, kernel_size=(3, 3),dilation=2,stride=2, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.05), # output size= 13 , RF = 13
            depthwise_separable_conv(nin=64, nout=128, kernel_size=(3, 3),dilation=2,stride=1, padding="same", bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.05)
        ) # output_size = 13 , RF = 21

        # Convolution BLOCK 2
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1,1),dilation=1, padding=0, bias=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3),dilation=2, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.05), # output size =9 , RF = 29
        )
      
        # CONVOLUTION BLOCK 3
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),dilation=2, padding="same", bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(64),
            nn.Dropout(0.05), # output size = 9 , RF = 37
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),dilation=2, padding="same", bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(128),
            nn.Dropout(0.05), # output size = 9 , RF = 45
        ) 

        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=9)
        ) # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        ) 


        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        #x = self.pool1(x)
        x = self.convblock4(x)
        # x = self.convblock5(x)
        # x = self.convblock6(x)
        # x = self.convblock7(x)
        x = self.gap(x)        
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
