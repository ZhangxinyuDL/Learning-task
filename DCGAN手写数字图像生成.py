import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms

#对数据做归一化（-1，1）
transform = transforms.Compose([
    transforms.ToTensor(),  #归一化0-1; channel,high,width
    transforms.Normalize(0.5,0.5) #均值方差都为0.5,范围变为(-1,1)
])
#MNIST数据集大小为1*28*28
train_ds = torchvision.datasets.MNIST('mnist',train=True,transform=transform,download=True)

dataloader = torch.utils.data.DataLoader(train_ds,batch_size=128,shuffle=True)


#定义生成器：输入是长度为100的噪声(正态分布随机数)，输出为（1，28，28）的图片

class Generator(nn.Module):
    def __init__(self,input_size,num_feature):
        super(Generator, self).__init__()
        self.fc=nn.Linear(input_size,num_feature)
        self.br=nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True),
        )
        self.gen = nn.Sequential(
            nn.Conv2d(1,64,3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64,32,3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32,1,3,stride=2,padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.fc(x)
        x=x.view(x.shape[0],1,56,56)
        x=self.br(x)
        x=self.gen(x)
        return x

#---------------------------------------------------------------------

#定义判别器：输入为(1,28,28)的图片，输出为二分类的概率值，输出使用sigmoid激活
#BCEloss计算交叉熵损失
#判别器推荐使用leakrelu()
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Conv2d(1,32,3,stride=1,padding=1),
            nn.LeakyReLU(0.2,True),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(32,64,3,stride=1,padding=1),
            nn.LeakyReLU(0.2,True),
            nn.MaxPool2d((2,2)),
        )
        self.fc = nn.Sequential(
            nn.Linear(7*7*64,1024),
            nn.LeakyReLU(0.2,True),
            nn.Linear(1024,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.dis(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x
#---------------------------------------------------------------------

#初始化模型，优化器，损失计算函数
device = 'cuda' if torch.cuda.is_available() else 'cpu'

gen = Generator(100,1*56*56).to(device)
dis = Discriminator().to(device)

g_optim = torch.optim.Adam(gen.parameters(),lr=0.0001)
d_optim = torch.optim.Adam(dis.parameters(),lr=0.0001)

loss_fn = torch.nn.BCELoss()


#绘图函数
def gen_img_plot(model,test_imput):
    prediction = np.squeeze(model(test_imput).detach().cpu().numpy()) #16
    fig = plt.figure(figsize=(4,4))
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow((prediction[i] + 1)/2)    #生成器生成图片取值范围在(-1,1),绘图需要把取值范围变为(0,1)
        plt.axis('off')
    plt.show()


test_imput = torch.randn(16,100,device=device) #16个长度为100的正态分布随机数



#GAN的训练
#记录每个epoch产生的损失值
D_loss = []
G_loss = []

#训练循环
for epoch in range(20):
    d_epoch_loss = 0
    g_epoch_loss = 0
    count = len(dataloader) #返回批次数
    for step,(img,_) in enumerate(dataloader): #img batch,label batch
        img = img.to(device)
        size = img.size(0)  #一批次的图片个数
        random_noise = torch.randn(size,100,device=device) #产生随机噪声

        #判别器
        d_optim.zero_grad()
        real_output = dis(img)    #对判别器输入真实的图片，real_output是对真实图片的预测结果
        d_real_loss = loss_fn(real_output,
                              torch.ones_like(real_output),#人为构造一个全1的数组和real_output进行比较
                              )      #判别器在真实图片上产生的损失
        d_real_loss.backward()

        gen_img = gen(random_noise) #生成器生成一张图片
        fake_output = dis(gen_img.detach()) #生成图片传给判别器，fake_output：对生成图片的预测
        d_fake_loss = loss_fn(fake_output,
                              torch.zeros_like(fake_output),#人为构造一个全0的数组和fake_output进行比较
                              )      #判别器在生成图片上产生的损失
        d_fake_loss.backward()

        d_loss = d_real_loss + d_fake_loss
        d_optim.step()

        #生成器:希望判别器判定为真
        g_optim.zero_grad()
        fake_output = dis(gen_img)
        g_loss = loss_fn(fake_output,
                         torch.ones_like(fake_output),
                         )      #生成器的损失
        g_loss.backward()
        g_optim.step()

        with torch.no_grad():  #把每一轮生成器和判别器的损失分别累加起来，不需要计算梯度
            d_epoch_loss +=d_loss
            g_epoch_loss +=g_loss

    with torch.no_grad(): #计算平均损失
        d_epoch_loss /= count
        g_epoch_loss /= count
        D_loss.append(d_epoch_loss)
        G_loss.append(g_epoch_loss)
        print('Epoch:',epoch)
        gen_img_plot(gen,test_imput)



