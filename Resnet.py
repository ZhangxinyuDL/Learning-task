import torch.nn.functional as F
import torchvision
import torch.nn as nn

class ResnetbasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(self,ResnetbasicBlock).__init__()
        self.conv1 = nn.Conv2d(in_channels,   #卷积
                               out_channels,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)  #批标准化

        self.conv2 = nn.Conv2d(in_channels,  # 卷积
                               out_channels,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)  # 批标准化

    def forward(self,x):
        residual = x    #作为残差加到输出上
        out = self.conv1(x)
        out = F.relu(self.bn1(out),inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        return F.relu(out)
'''
model = torchvision.models.resnet34()
print(model)
'''