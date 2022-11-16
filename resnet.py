import torch
import torch.nn as nn
import torchvision
import numpy as np

import torchaudio
import torch.nn.functional as F

import os



class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)
    
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

__all__ = ['ResNet50', 'ResNet101','ResNet152']

def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class SAP(nn.Module):

    def __init__(self, out_size, dim=128):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Conv1d(out_size, dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(dim),
            nn.Conv1d(dim, out_size, kernel_size=1),
            nn.Softmax(dim=2)
        )

    def forward(self, X):
        b, c, h, w = X.size()

        X = X.reshape(b, -1, w)
        W = self.attention(X)
        return torch.sum(W * X, dim=2)
    
class ResNet(nn.Module):
    def __init__(self,blocks, num_classes=192, expansion = 4):
        super(ResNet,self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes = 1, places= 64)

        self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)

        self.avgpool = nn.AdaptiveMaxPool2d((1,1))
        self.SAP = SAP(2048 * 3)
        self.fc = nn.Linear(2048 * 3,num_classes)
        self.mel_feature = torch.nn.Sequential(
            PreEmphasis(),        #预加重 no large influence     
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            ) ## (channel, n_mels, time)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward(self, x):

        x = x.squeeze(1)
        with torch.no_grad():
            #print("m",x.shape)
            x = self.mel_feature(x) + 1e-6   #根源是f16在这里产生了inf ，下面inf - inf 就变成nan了。gpu和cpu的运算有差别？？ 特征提取放到dataset并不会有inf的情况。 因为cpu用的是f32计算，gpu用的是f16，超过65535就溢出了
            #(B,bin, t)
            #normalize
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True) # (1, n_mel, t) 这个语句会导致loss变nan，在f16的情况下
            x = x.unsqueeze(1) 
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #print(x.shape)
        #x = self.avgpool(x)
        x = self.SAP(x)
        # print(x.shape)
        # exit()
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def ResNet50():
    return ResNet([3, 4, 6, 3])

def ResNet101():
    return ResNet([3, 4, 23, 3])

def ResNet152():
    return ResNet([3, 8, 36, 3])


if __name__=='__main__':
    #model = torchvision.models.resnet50()
    model = ResNet50()
    print(model)

    input = torch.randn(1, 1, 224, 224)
    out = model(input)
    print(out.shape)