import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms



transform = transforms.Compose([
    transforms.RandomResizedCrop((28,28)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.RandomGrayscale(0.1),
    transforms.ToTensor()
])

class VGGbase(nn.Module):
    def __init__(self):
        super(VGGbase,self).__init__()
        #3 * 28 * 28: 32的图像经过裁剪会变成28
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.max_pool1 = nn.MaxPool2d(kernel_size=2,stride=2)

        #14 * 14
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        #7 * 7

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2,padding=1)
        #4 * 4

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        #2 * 2

        #512 * 2 *2 -> 512 * 4
        self.fc = nn.Linear(512*4,10)
    def forward(self,x):
        batchsize = x.size(0)
        out = self.conv1(x)
        out = self.max_pool1(out)
        out = self.conv2(out)
        out = self.max_pool2(out)
        out = self.conv3(out)
        out = self.max_pool3(out)
        out = self.conv4(out)
        out = self.max_pool4(out)

        out = out.view(batchsize,-1) #512 * 4
        out = self.fc(out)
        out = F.log_softmax(out,dim=1)

        return out