
#coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets,transforms,models
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    trainset = datasets.ImageFolder('faces', data_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=5, shuffle=True, num_workers=4)


    def imshow(inputs, picname):
        plt.ion()

        inputs = inputs / 2 + 0.5
        inputs = inputs.numpy().transpose((1, 2, 0))
        plt.imshow(inputs)
        plt.pause(0.01)

        plt.savefig(os.path.join('faces', '0', picname + ".jpg"))
        plt.close()


    inputs, __ = next(iter(trainloader))
    imshow(torchvision.utils.make_grid(inputs), "RealDataSample")


    class D(nn.Module):
        def __init__(self, nc, ndf):
            super(D, self).__init__()
            self.layer1 = nn.Sequential(nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1),
                                        nn.BatchNorm2d(ndf),
                                        nn.LeakyReLU(0.2, inplace=True))
            self.layer2 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
                                        nn.BatchNorm2d(ndf * 2),
                                        nn.LeakyReLU(0.2, inplace=True))
            self.layer3 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
                                        nn.BatchNorm2d(ndf * 4),
                                        nn.LeakyReLU(0.2, inplace=True))
            self.layer4 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
                                        nn.BatchNorm2d(ndf * 8),
                                        nn.LeakyReLU(0.2, inplace=True))
            self.fc = nn.Sequential(nn.Linear(256 * 6 * 6, 1), nn.Sigmoid())

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = out.view(-1, 256 * 6 * 6)
            out = self.fc(out)
            return out


    # d = D(3,32)

    # print(d(inputs))

    class G(nn.Module):
        def __init__(self, nc, ngf, nz, feature_size):
            super(G, self).__init__()
            self.prj = nn.Linear(feature_size, nz * 6 * 6)
            self.layer1 = nn.Sequential(nn.ConvTranspose2d(nz, ngf * 4, kernel_size=4, stride=2, padding=1),
                                        nn.BatchNorm2d(ngf * 4),
                                        nn.ReLU())
            self.layer2 = nn.Sequential(nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1),
                                        nn.BatchNorm2d(ngf * 2),
                                        nn.ReLU())
            self.layer3 = nn.Sequential(nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1),
                                        nn.BatchNorm2d(ngf),
                                        nn.ReLU())
            self.layer4 = nn.Sequential(nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1),
                                        nn.Tanh())

        def forward(self, x):
            out = self.prj(x)
            out = out.view(-1, 1024, 6, 6)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            return out


    d = D(3, 32)
    g = G(3, 128, 1024, 100)

    criterion = nn.BCELoss()

    d_optimizer = torch.optim.Adam(d.parameters(), lr=0.0003)
    g_optimizer = torch.optim.Adam(g.parameters(), lr=0.0003)


    def train(d, g, criterion, d_optimizer, g_optimizer, epochs=1, show_every=1000, print_every=10):
        iter_count = 0
        for epoch in range(epochs):

            for inputs, _ in trainloader:

                real_inputs = inputs

                fake_inputs = g(torch.randn(5, 100))

                real_labels = torch.ones(real_inputs.size(0))
                fake_labels = torch.zeros(5)

                real_outputs = d(real_inputs)
                d_loss_real = criterion(real_outputs, real_labels)
                real_scores = real_outputs

                fake_outputs = d(fake_inputs)
                d_loss_fake = criterion(fake_outputs, fake_labels)
                fake_scores = fake_outputs

                d_loss = d_loss_real + d_loss_fake
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

                fake_inputs = g(torch.randn(5, 100))
                outputs = d(fake_inputs)
                real_labels = torch.ones(outputs.size(0))
                g_loss = criterion(outputs, real_labels)

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

                if (iter_count % show_every == 0):
                    print(
                        'Epoch:{},Iter: {}, D: {:.4}, G:{:.4}'.format(epoch, iter_count, d_loss.item(), g_loss.item()))
                    picname = "Epoch_" + str(epoch) + "Iter_" + str(iter_count)
                    imshow(torchvision.utils.make_grid(fake_inputs.data), picname)
                    save_param(d, 'd_model.pkl')
                    save_param(g, 'g_model.pkl')

                if (iter_count % print_every == 0):
                    print(
                        'Epoch:{},Iter: {}, D: {:.4}, G:{:.4}'.format(epoch, iter_count, d_loss.item(), g_loss.item()))
                iter_count += 1

        print('Finished Training')


    def load_param(model, path):
        if os.path.exists(path):
            model.load_state_dict(torch.load(path))


    def save_param(model, path):
        torch.save(model.state_dict(), path)


    if os.path.exists("d_model.pkl"):
        load_param(d, 'd_model.pkl')
        load_param(g, 'g_model.pkl')

    train(d, g, criterion, d_optimizer, g_optimizer, epochs=300)

    save_param(d, 'd_model.pkl')
    save_param(g, 'g_model.pkl')


