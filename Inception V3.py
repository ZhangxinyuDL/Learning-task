import torch
from torch.utils import data
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torchvision
import torch.nn as nn
from torchvision import transforms

'''
Inception模块包括四个分支：
(1*1卷积)分支
(1*1卷积 + 3*3卷积)分支
(1*1卷积 + 5*5卷积)分支
(3*3最大池化(3*3) + 1*1卷积)分支
'''

#定义基础卷积模型：卷积 + BN +激活

class Basicconv(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):  #参数省略,in_channels,out_channels为初始化模型时的输入
        super(Basicconv,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,bias=False,**kwargs) #BN层是做平移变换：*权重 + 偏置
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):   #x为调用时的输入
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x,inplace=True) #inplace=True:就地改变，在内存里直接将原数据改变

'''
pytorch默认的填充方式为'VALID',每次卷积都要下降k-1个像素，k为卷积核大小
'''
#定义Inception模块
class InceptionBlock(nn.Module):
    def __init__(self,in_channels,pool_features):
        super(InceptionBlock,self).__init__()
        # 1*1卷积分支
        self.b_1x1 = Basicconv(in_channels,64,kernel_size=1)
        # 3*3卷积分支的两部分
        self.b_3x3_1 = Basicconv(in_channels,64,kernel_size=1)
        self.b_3x3_2 = Basicconv(64,96, kernel_size=3,padding=1)  #经过填充维持卷积后图像大小不变
        # 5*5卷积分支的两部分
        self.b_5x5_1 = Basicconv(in_channels, 48, kernel_size=1)
        self.b_5x5_2 = Basicconv(48, 64, kernel_size=5, padding=2)  # 经过填充维持卷积后图像大小不变
        #最大池化分支
        self.b_pool = Basicconv(in_channels,pool_features,kernel_size=1)

    def forward(self,x):
        # 第一条分支
        b_1x1_out = self.b_1x1(x)
        # 第二条分支
        b_3x3 = self.b_3x3_1(x)
        b_3x3_out = self.b_3x3_2(b_3x3)
        # 第三条分支
        b_5x5 = self.b_5x5_1(x)
        b_5x5_out = self.b_5x5_2(b_5x5)
        # 第四条分支
        b_pool = F.max_pool2d(x,kernel_size=3,stride=1,padding=1) #保证池化后图像一样大
        b_pool_out = self.b_pool(b_pool)

        outputs = [b_1x1_out,b_3x3_out,b_5x5_out,b_pool_out]
        return torch.cat(outputs,dim=1) #沿着特征维度进行合并

my_inception_block = InceptionBlock(32,64)

#print(my_inception_block)

#Inception模块应用：GoogLeNet,Inception_v3

#Inception_v3
model = torchvision.models.inception_v3(pretrained=True)

print(model)