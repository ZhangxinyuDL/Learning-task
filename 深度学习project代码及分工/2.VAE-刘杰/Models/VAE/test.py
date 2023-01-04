from model import cVAE
import torchvision
import torch
from matplotlib import pyplot as plt
from matplotlib import cm
from model import VAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = VAE((1, 28, 28), nhid=4)
checkpoint = torch.load(
    "D:\研究生课程文件\研究生上学期\深度学习\project\VAE-pytorch-master\Models\VAE\VAE.pt", map_location=device)
net.load_state_dict(checkpoint["net"])
net.to(device)
net.eval()

with torch.no_grad():
    x = net.generate()
plt.imshow(x.squeeze(0).cpu().numpy(), cm.gray)

with torch.no_grad():
    x = net.generate(batch_size=15)

print("fake images")
for i in range(15):
    plt.subplot(3, 5, i+1)
    plt.axis("off")
    plt.imshow(x[i].squeeze(0).cpu().numpy(), cm.gray)

train_data = torchvision.datasets.MNIST(
    root='../../Datasets', train=True, download=True, transform=torchvision.transforms.ToTensor())
print("real images")
for i in range(15):
    plt.subplot(3, 5, i+1)
    plt.axis("off")
    plt.imshow(train_data[i][0].squeeze(0).cpu().numpy(), cm.gray)
