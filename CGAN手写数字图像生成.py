import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms

'''
条件GAN需要输入条件label，0-9，使用one-hot编码进行实现
'''
def one_hot(x,class_count=10):
    return torch.eye(class_count)[x,:]

#对数据做归一化（-1，1）
transform = transforms.Compose([
    transforms.ToTensor(),  #归一化0-1; channel,high,width
    transforms.Normalize(0.5,0.5) #均值方差都为0.5,范围变为(-1,1)
])
#MNIST数据集大小为1*28*28
train_ds = torchvision.datasets.MNIST('mnist',
                                      train=True,
                                      transform=transform,
                                      target_transform=one_hot,
                                      download=True)

dataloader = torch.utils.data.DataLoader(train_ds,batch_size=64,shuffle=True)


#定义生成器：输入是长度为100的噪声(正态分布随机数)，输出为（1，28，28）的图片
'''
linear1:100-256
linear2:256-512
linear3:512-28*28
reshape:28*28-1*28*28
'''
# 定义生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(10, 128 * 7 * 7)
        self.bn1 = nn.BatchNorm1d(128 * 7 * 7)
        self.linear2 = nn.Linear(100, 128 * 7 * 7)
        self.bn2 = nn.BatchNorm1d(128 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(256, 128,
                                          kernel_size=(3, 3),
                                          padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64,
                                          kernel_size=(4, 4),
                                          stride=2,
                                          padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 1,
                                          kernel_size=(4, 4),
                                          stride=2,
                                          padding=1)

    def forward(self, x1, x2):
        x1 = F.relu(self.linear1(x1))
        x1 = self.bn1(x1)
        x1 = x1.view(-1, 128, 7, 7)
        #print(x1.shape)
        x2 = F.relu(self.linear2(x2))
        x2 = self.bn2(x2)
        x2 = x2.view(-1, 128, 7, 7)
        #print(x2.shape)
        x = torch.cat([x1, x2], axis=1)
        x = F.relu(self.deconv1(x))
        x = self.bn3(x)
        x = F.relu(self.deconv2(x))
        x = self.bn4(x)
        x = torch.tanh(self.deconv3(x))
        #print(x.shape)
        return x

#---------------------------------------------------------------------

#定义判别器：输入为(1,28,28)的图片，输出为二分类的概率值，输出使用sigmoid激活
#BCEloss计算交叉熵损失
#判别器推荐使用leakrelu()
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear = nn.Linear(10, 1 * 28 * 28)
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.bn = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128 * 6 * 6, 1)  # 输出一个概率值

    def forward(self, x1, x2):
        x1 = F.leaky_relu(self.linear(x1))
        x1 = x1.view(-1, 1, 28, 28)
        #print(x1.shape)
        #print(x2.shape)
        x = torch.cat([x1, x2], axis=1)
        x = F.dropout2d(F.leaky_relu(self.conv1(x)))
        x = F.dropout2d(F.leaky_relu(self.conv2(x)))
        x = self.bn(x)
        x = x.view(-1, 128 * 6 * 6)
        x = torch.sigmoid(self.fc(x))
        #print(x.shape)
        return x

#---------------------------------------------------------------------

#初始化模型，优化器，损失计算函数
device = 'cuda' if torch.cuda.is_available() else 'cpu'

gen = Generator().to(device)
dis = Discriminator().to(device)

g_optim = torch.optim.Adam(gen.parameters(),lr=0.0001)
d_optim = torch.optim.Adam(dis.parameters(),lr=0.0001)

loss_fn = torch.nn.BCELoss()


#绘图函数
def gen_img_plot(model,label,test_imput):
    prediction = np.squeeze(model(label,test_imput,).detach().cpu().numpy()) #16
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

label_seed = torch.randint(0, 10, size=(16,))
label_seed_onehot = one_hot(label_seed).to(device)
print(label_seed_onehot.shape)
#训练循环
for epoch in range(20):
    d_epoch_loss = 0
    g_epoch_loss = 0
    count = len(dataloader) #返回批次数
    for step,(img,label) in enumerate(dataloader): #img batch,label batch
        img = img.to(device)
        label = label.to(device)
        size = img.size(0)  #一批次的图片个数
        random_noise = torch.randn(size,100,device=device) #产生随机噪声

        #判别器
        d_optim.zero_grad()
        real_output = dis(label,img)    #对判别器输入真实的图片，real_output是对真实图片的预测结果
        d_real_loss = loss_fn(real_output,
                              torch.ones_like(real_output),#人为构造一个全1的数组和real_output进行比较
                              )      #判别器在真实图片上产生的损失
        d_real_loss.backward()

        gen_img = gen(label,random_noise) #生成器生成一张图片
        fake_output = dis(label,gen_img.detach()) #生成图片传给判别器，fake_output：对生成图片的预测
        d_fake_loss = loss_fn(fake_output,
                              torch.zeros_like(fake_output),#人为构造一个全0的数组和fake_output进行比较
                              )      #判别器在生成图片上产生的损失
        d_fake_loss.backward()

        d_loss = d_real_loss + d_fake_loss
        d_optim.step()

        #生成器:希望判别器判定为真
        g_optim.zero_grad()
        fake_output = dis(label,gen_img)
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
        gen_img_plot(gen,  label_seed_onehot,test_imput,)



