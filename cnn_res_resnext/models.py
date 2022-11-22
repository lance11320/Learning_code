from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch

'''
def model_A(num_classes):
    model_resnet = models.resnet18(pretrained=False)
    num_features = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_features, num_classes)
    return model_resnet
'''
def model_A(num_classes):
    ## your code here
    class ResidualBlock(nn.Module):
        def __init__(self, inchannel, outchannel, stride=1):
            super(ResidualBlock, self).__init__()
            self.left = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(inplace=True),
                nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(outchannel)
            )
            self.shortcut = nn.Sequential()
            if stride != 1 or inchannel != outchannel:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(outchannel)
                )

        def forward(self, x):
            out = self.left(x)
            out += self.shortcut(x)
            out = F.relu(out)
            return out

    class ResNet(nn.Module):
        def __init__(self, ResidualBlock, num_classes):
            super(ResNet, self).__init__()
            self.inchannel = 64
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
            self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
            self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
            self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
            self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
            self.fc = nn.Linear(512, num_classes)

        def make_layer(self, block, channels, num_blocks, stride):
            strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
            layers = []
            for stride in strides:
                layers.append(block(self.inchannel, channels, stride))
                self.inchannel = channels
            return nn.Sequential(*layers)

        def forward(self, x):
            out = self.conv1(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 14)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out

    def ResNet18(num_classes):
        return ResNet(ResidualBlock,num_classes)
    
    model = ResNet18(num_classes)
    return model

def model_B(num_classes):
    class Block(nn.Module):
        def __init__(self,in_channels, out_channels, stride=1, is_shortcut=False):
            super(Block,self).__init__()
            self.relu = nn.ReLU(inplace=True)
            self.is_shortcut = is_shortcut
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels // 2, kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels // 2),
                nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1, groups=32,
                                       bias=False),
                nn.BatchNorm2d(out_channels // 2),
                nn.ReLU()
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(out_channels // 2, out_channels, kernel_size=1,stride=1,bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if is_shortcut:
                self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=1),
                nn.BatchNorm2d(out_channels)
            )
        def forward(self, x):
            x_shortcut = x
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            if self.is_shortcut:
                x_shortcut = self.shortcut(x_shortcut)
            x = x + x_shortcut
            x = self.relu(x)
            return x
    
    class Resnext(nn.Module):
        def __init__(self,num_classes,layer=[3,4,6,3]):
            super(Resnext,self).__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            self.conv2 = self._make_layer(64,256,1,num=layer[0])
            self.conv3 = self._make_layer(256,512,2,num=layer[1])
            self.conv4 = self._make_layer(512,1024,2,num=layer[2])
            self.conv5 = self._make_layer(1024,2048,2,num=layer[3])
            self.global_average_pool = nn.AvgPool2d(kernel_size=7, stride=1)
            self.fc = nn.Linear(2048,num_classes)
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.global_average_pool(x)
            x = torch.flatten(x,1)
            x = self.fc(x)
            return x
        def _make_layer(self,in_channels,out_channels,stride,num):
            layers = []
            block_1=Block(in_channels, out_channels,stride=stride,is_shortcut=True)
            layers.append(block_1)
            for i in range(1, num):
                layers.append(Block(out_channels,out_channels,stride=1,is_shortcut=False))
            return nn.Sequential(*layers)

    model = Resnext(num_classes)
    return model
