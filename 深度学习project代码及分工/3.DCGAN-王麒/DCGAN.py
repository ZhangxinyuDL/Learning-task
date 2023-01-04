import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()          #命令行选项、参数和子命令解析器
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")  #迭代次数
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")          #batch大小
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")            #学习率
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient") #动量梯度下降第一个参数
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient") #动量梯度下降第二个参数
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation") #CPU个数
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")  #噪声数据生成维度
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")  #输入数据的维度
parser.add_argument("--channels", type=int, default=1, help="number of image channels")      #输入数据的通道数
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")  #保存图像的迭代数
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False        #判断GPU可用，有GPU用GPU，没有用CPU


def weights_init_normal(m):            #自定义初始化参数
    classname = m.__class__.__name__   #获得类名
    if classname.find("Conv") != -1:   #在类classname中检索到了Conv
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2)) #l1函数进行Linear变换。线性变换的两个参数是变换前的维度，和变换之后的维度

        self.conv_blocks = nn.Sequential(           #nn.sequential{}是一个组成模型的壳子，用来容纳不同的操作
            nn.BatchNorm2d(128),                    # BatchNorm2d的目的是使我们的一批（batch）feature map 满足均值0方差1，就是改变数据的量纲
            nn.Upsample(scale_factor=2),            #上采样，将图片放大两倍（这就是为啥class最先开始将图片的长宽除了4，下面还有一次放大2倍）
            nn.Conv2d(128, 128, 3, stride=1, padding=1), #二维卷积函数，（输入数据channel，输出的channel，步长，卷积核大小，padding的大小）
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),        #relu激活函数
            nn.Upsample(scale_factor=2),            #上采样
            nn.Conv2d(128, 64, 3, stride=1, padding=1),#二维卷积
            nn.BatchNorm2d(64, 0.8),                #BN
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),                              #Tanh激活函数
        )

    def forward(self, z):
        out = self.l1(z)              #l1函数进行的是Linear变换 （第50行定义了）
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)#view是维度变换函数，可以看到out数据变成了四维数据，第一个是batch_size(通过整个的代码，可明白),第二个是channel，第三,四是单张图片的长宽
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]#Conv卷积，Relu激活，Dropout将部分神经元失活，进而防止过拟合
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))    #如果bn这个参数为True，那么就需要在block块里面添加上BatchNorm的归一化函数
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid()) #先进行线性变换，再进行激活函数激活
                          #上一句中 128是指model中最后一个判别模块的最后一个参数决定的，ds_size由model模块对单张图片的卷积效果决定的，而2次方是整个模型是选取的长宽一致的图片
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)    #将处理之后的数据维度变成batch * N的维度形式
        validity = self.adv_layer(out)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()         #定义了一个BCE损失函数

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:                                #初始化，将数据放在cuda上
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(     #显卡加速
    datasets.MNIST(
        "../../data/mnist",                  #进行训练集下载
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers                             定义神经网络的优化器  Adam就是一种优化器
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))     #将真实的图片转化为神经网络可以处理的变量

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()   #把梯度置零  每次训练都将上一次的梯度置零，避免上一次的干扰

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))#生成的噪音 随机构00维向量 均值0方差1维度(64，100)的噪音，随机初始化一个64大小batch的向量
                                          # 输入0到1之间，形状为imgs.shape[0], opt.latent_dim的随机高斯数据。np.random.normal()正态分布
        # Generate a batch of images
        gen_imgs = generator(z)           #得到一个批次的图片

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()         #反向传播和模型更新
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)     #判别器判别真实图片是真的的损失
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)  #判别器判别假图片是假的的损失
        d_loss = (real_loss + fake_loss) / 2     #判别器去判别真实图片是真的和生成图片是假的的损失之和，让这个和越大，说明判别器越准确

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
