'''
Densenet比Resnet更进一步，它引入了每层与所有后续层的连接，
即每一层都接收所有前置层的特征平面作为输入，
网络每一层的输入都是前面所有层输出的并集(concat)

采用DenseBlock + Transition
DenseBlock:是包含很多层的模块，每个层的特征图大小相同，层与层之间采用密集连接的方式
Transition:是连接两个相邻的DenseBlock，并且通过Pooling使特征图大小降低
'''
import torch
from torch.utils import data
from PIL import Image
import numpy as np
import torchvision
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import glob

#使用birds数据集-200分类
#使用torchvision.models.densenet121()预训练模型

imgs_path = glob.glob('birds/*/*.jpg')
#print(imgs_path[:5])
#提取路径里的图片类别名称
img_p = imgs_path[100]
#print(img_p.split('/')[1].split('.')[1])
all_labels_name = [img_p.split('/')[1].split('.')[1] for img_p in imgs_path]
#print(all_labels_name)
unique_labels = np.unique(all_labels_name)
#print(len(unique_labels))

label_to_index = dict((v,k) for k,v in enumerate(unique_labels)) #映射成label:index 的字典格式

index_to_label = dict((v,k) for k,v in label_to_index.items()) #映射成index:label 的字典格式

all_labels = [label_to_index.get(name) for name in all_labels_name] #将all_labels_name映射成index格式
#print(all_labels[-5:])

#划分训练数据和测试数据
np.random.seed(2021)
random_index = np.random.permutation(len(imgs_path))
imgs_path = np.array(imgs_path)[random_index]
all_labels = np.array(all_labels)[random_index]

i = int(len(imgs_path)*0.8)

train_path = imgs_path[:i]
train_labels = all_labels[:i]
test_path = imgs_path[i:]
test_labels = all_labels[i:]

transform = transforms.Compose([
                   transforms.Resize((224,224)),
                   transforms.ToTensor()
])

class BirdsDataset(data.Dataset):
    def __init__(self,imgs_path,labels):
        self.imgs = imgs_path
        self.labels = labels
    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]

        pil_img = Image.open(img) #黑白照片读入没有channel：(H,W),需要人为转化为channel为3的图片
        np_img = np.asarray(pil_img,dtype=np.uint8)
        if len(np_img.shape)==2:  #如果是黑白照片
            img_data = np.repeat(np_img[:,:,np.newaxis],3,axis=2)  #添加一个维度并重复3次
            pil_img = Image.fromarray(img_data) #转化回pillowimage形式

        img_tensor = transform(pil_img)
        return img_tensor,label
    def __len__(self):
        return len(self.imgs)

train_ds = BirdsDataset(train_path,train_labels)
test_ds = BirdsDataset(test_path,test_labels)

BATCH_SIZE = 32

train_dl = data.DataLoader(
    train_ds,
    batch_size=BATCH_SIZE
)

test_dl = data.DataLoader(
    test_ds,
    batch_size=BATCH_SIZE
)

img_batch,label_batch = next(iter(train_dl))

#print(img_batch.shape)
'''
for im,la in test_dl:
    print(im.type())
'''

plt.figure(figsize=(12,8))
for i ,(img,label) in enumerate(zip(img_batch[:6],label_batch[:6])):
    img = img.permute(1,2,0).numpy()
    plt.subplot(2,3,i+1)
    plt.title(index_to_label.get(label.item()))
    plt.imshow(img)


#使用DenseNet提取特征
my_densenet = torchvision.models.densenet121(pretrained=True).features #获取模型的卷积部分
#print(my_densenet)


if torch.cuda.is_available():
    my_densenet = my_densenet.cuda()
for p in my_densenet.parameters(): #设置参数为不可训练
    p.requires_grad = False

train_features = [] #提取特征放到列表里
train_feat_labels = [] #提取标签放到列表里
for im,la in train_dl:
    out = my_densenet(im.type(torch.float32))
    out = out.view(out.size(0),-1)
    train_features.extend(out.cpu().data)
    train_feat_labels.extend(la)
    print(la)

test_features = []
test_feat_labels = []
for im,la in test_dl:
    out = my_densenet(im.type(torch.float32))
    out = out.view(out.size(0),-1)
    test_features.extend(out.cpu().data)
    test_feat_labels.extend(la)
    print(la)

print(len(train_features))
print(train_features[0].shape)


class FeatureDataset(data.Dataset):
    def __init__(self,feat_list,label_list):
        self.feat_list = feat_list
        self.label_list = label_list
    def __getitem__(self, index):
        return self.feat_list[index],self.label_list[index]
    def __len__(self):
        return len(self.feat_list)


train_feat_ds = FeatureDataset(train_features,train_feat_labels)
test_feat_ds = FeatureDataset(test_features,test_feat_labels)

train_feat_dl = data.DataLoader(train_feat_ds,batch_size=BATCH_SIZE,shuffle=True)
test_feat_dl = data.DataLoader(test_feat_ds,batch_size=BATCH_SIZE)

in_feat_size = train_features[0].shape.item()

class FCModel(nn.Module):
    def __init__(self,in_size,out_size):
        super().__init__()
        self.linear = nn.Linear(in_size,out_size)

    def forward(self,input):
        return self.linear(input)

net = FCModel(in_feat_size,200) #初始化分类模型

if torch.cuda.is_available():
    net.to('cuda')

loss_fn = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(net.parameters(),lr=0.00001)


def fit(epoch, model, trainloader, testloader):
    correct = 0
    total = 0
    running_loss = 0

    model.train()
    for x, y in trainloader:
        if torch.cuda.is_available():
            y = torch.tensor(y, dtype=torch.long)
            x, y = x.to('cuda'), y.to('cuda')
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()

    epoch_loss = running_loss / len(trainloader.dataset)
    epoch_acc = correct / total

    test_correct = 0
    test_total = 0
    test_running_loss = 0

    model.eval()
    with torch.no_grad():
        for x, y in testloader:
            y = torch.tensor(y, dtype=torch.long)
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()

    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / test_total

    print('epoch: ', epoch,
          'loss： ', round(epoch_loss, 3),
          'accuracy:', round(epoch_acc, 3),
          'test_loss： ', round(epoch_test_loss, 3),
          'test_accuracy:', round(epoch_test_acc, 3)
          )

    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc


epochs = 50

train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in range(epochs):
    epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc = fit(epoch,
                                                                 net,
                                                                 train_feat_dl,
                                                                 test_feat_dl)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)
