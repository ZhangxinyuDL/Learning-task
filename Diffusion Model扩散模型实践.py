import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_checkerboard,make_circles,make_moons,make_s_curve,make_swiss_roll
import torch
import torch
import torch.nn as nn
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision import transforms

s_curve, _ = make_s_curve(10**4, noise=0.1)#10000个点
s_curve = s_curve[:,[0, 2]]/10.0#只取第0，2维

print('shape of moons',np.shape(s_curve))

data = s_curve.T

fig, ax = plt.subplots()

ax.scatter(*data, color='red', edgecolor='white')
ax.axis('off')

dataset = torch.Tensor(s_curve).float()



num_steps = 100 #对于步骤，开始由beta,分布的均值和标准差共同决定
#每一步的beta,逐步递增
betas = torch.linspace(-6,6,num_steps)
betas = torch.sigmoid(betas)*(0.5e-2 - 1e-5) + 1e-5
#计算alpha，alphas_prod等变量的值
alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, 0) #alphas连乘
alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)#从alphas连乘第一项开始，把第零项令成1
alphas_bar_sqrt = torch.sqrt(alphas_prod)#根号alphas_prod_p
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)#1-alphas连乘的log
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)#1-alphas连乘的开方

assert alphas.shape==alphas_prod.shape==alphas_prod_p.shape== \
alphas_bar_sqrt.shape==one_minus_alphas_bar_log.shape\
==one_minus_alphas_bar_sqrt.shape
print('all the same shape:',betas.shape)#所有参数尺寸都是一样的

#计算任意时刻x的采样值，基于x0和参数重整化技巧
def q_x(x_0, t):
    #可以基于x[0]得到任意t时刻的x[t]
    noise = torch.randn_like(x_0)#生成正态分布的随机噪音
    alphas_t = alphas_bar_sqrt[t] #均值
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t] #标准差
    return (alphas_t * x_0 + alphas_1_m_t * noise) #在x[0]的基础上添加噪声


num_shows = 20
fig, axs = plt.subplots(2, 10, figsize=(28, 3))
plt.rc('text', color='blue')
# 共有10000个点，每个点包括两个坐标

# 生成100步内每隔5步加噪声的图像
for i in range(num_shows):
    j = i // 10
    k = i % 10
    q_i = q_x(dataset, torch.tensor([i * num_steps // num_shows]))
    axs[j, k].scatter(q_i[:, 0], q_i[:, 1], color='red', edgecolor='white')

    axs[j, k].set_axis_off()
    axs[j, k].set_title('$q(\mathbf{x}_{' + str(i * num_steps // num_shows) + '})$')


class MLPDiffusion(nn.Module):
    def __init__(self,n_steps,num_units=128):
        super(MLPDiffusion,self).__init__()

        self.linears = nn.ModuleList(
            [
                nn.Linear(2, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, 2),
            ]
        )
        self.step_embedding = nn.ModuleList(
            [
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
            ]
        )

    def forward(self, x_0, t):
        x = x_0
        for idx, embedding_layer in enumerate(self.step_embedding):
            t_embedding = embedding_layer(t)
            x = self.linears[2*idx](x)
            x += t_embedding
            x = self.linears[2*idx+1](x)
        x = self.linears[-1](x)

        return x

#误差函数
def diffusion_loss_fn(model,x_0,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,n_steps):
    '''
    对于任意时刻t进行采样计算loss
    '''
    batch_size = x_0.shape[0]
    #print(x_0.shape)#(128,3)
    #随机采样一个时刻t
    t = torch.randint(0, n_steps, size=(batch_size//2,))
    #print(t.shape) (64)
    t = torch.cat([t, n_steps-1-t], dim=0)
    #print(t.shape) (128)
    t = t.unsqueeze(-1)
    #print(t.shape) #(128,1)

    #x0的系数
    a = alphas_bar_sqrt[t]

    #eps的系数
    am1 = one_minus_alphas_bar_sqrt[t]

    #生成随机噪音eps
    e = torch.randn_like(x_0)

    #构建模型的输入
    x = x_0 * a + e * am1
    #print(x.shape)(128,3)
    #送入模型得到t时刻的随机噪声预测值
    output = model(x, t.squeeze(-1))

    #与真实噪声一起计算误差，求平均值
    return (e - output).square().mean()


#逆扩散采样函数
def p_sample_loop(model,shape,n_steps,betas,one_minus_alphas_bar_sqrt):
    '''
    从x[t]恢复x[t-1]，x[t-2]...x[0]
    '''
    cur_x = torch.randn(shape)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model,cur_x,i,betas,one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq

def p_sample(model,x,t,betas,one_minus_alphas_bar_sqrt):
    '''
    从x[t]采样t时刻的重构值
    '''
    t = torch.tensor([t])
    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
    eps_theta = model(x, t)
    mean = (1 / (1-betas[t]).sqrt()) * (x - (coeff * eps_theta))

    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()

    sample = mean + sigma_t * z

    return (sample)




#开始训练
seed = 1234
'''
class EMA():
    
    #构建一个参数平滑器
    
    def __init__(self,mu=0.01):
        self.mu = mu
        self.shadow = {}

    def register(self,name,val):
        self.shadow[name] = val.clone()

    def __call__(self,name,x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average


ema = EMA(0.5)
for name,param in model.named_parameters():
    if param.requirea_grad:
        ema.register(name.param.data)
'''
print('Training model...')

batch_size = 128
dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)
num_epoch = 4000
plt.rc('text',color='blue')
model = MLPDiffusion(num_steps)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

for t in range(num_epoch):
    for idx,batch_x in enumerate(dataloader):
        #print(batch_x.shape)#(128,3)
        loss = diffusion_loss_fn(model,batch_x,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,num_steps)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(),1.)
        optimizer.step()

    if (t % 100 == 0):
        print(loss)
        x_seq = p_sample_loop(model,dataset.shape,num_steps,betas,one_minus_alphas_bar_sqrt)

        fig, axs = plt.subplots(1,10,figsize=(28,3))
        for i in range(1,11):
            cur_x = x_seq[i * 10].detach()
            axs[i-1].scatter(cur_x[:,0],cur_x[:,1],color='red',edgecolor='white')
            axs[i-1].set_axis_off()
            axs[i - 1].set_title('$q(\mathbf{x}_{'+str(i*10)+'})$')






'''

#生成扩散过程和逆扩散过程的动画演示
import io
from PIL import Image

imgs = []
for i in range(100):
    plt.clf()
    q_i = q_x(dataset,torch.tensor([i]))
    plt.scatter(q_i[:,0],q_i[:,1],color='white', edgecolor='gray',s=5)
    plt.axis('off')

    img_buf = io.BytesIO()
    plt.savefig(img_buf,format='png')
    img = Image.open(img_buf)
    imgs.append(img)


reverse = []
for i in range(100):
    plt.clf()
    cur_x = x_seq[i].detach()
    plt.scatter(cur_x[:,0],cur_x[:,1],color='white', edgecolor='gray',s=5)
    plt.axis('off')

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img = Image.open(img_buf)
    imgs.append(img)
'''