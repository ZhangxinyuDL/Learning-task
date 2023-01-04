from model import UnetModel
import numpy as np
import time
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from diffusion import GaussianDiffusion

batch_size = 128
timesteps = 500

save_dir = 'diffusion_ep_models/'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# use MNIST dataset
dataset = datasets.MNIST(root='mnist', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# define model and diffusion
device = "npu" 
model = UnetModel(
    in_channels=1,
    model_channels=96,
    out_channels=1,
    channel_mult=(1, 2, 2),
    attention_resolutions=[],
    class_num=10
)
model.to(device)

gaussian_diffusion = GaussianDiffusion(timesteps=timesteps)
image = next(iter(train_loader))[0][0].squeeze()
label = next(iter(train_loader))[1][0].squeeze()

x_start = image

gaussian_diffusion = GaussianDiffusion(timesteps=500, beta_schedule='linear')

plt.figure(figsize=(16, 5))
for idx, t in enumerate([0, 100, 300, 400, 499]):
    x_noisy = gaussian_diffusion.q_sample(x_start.to(device), t=torch.tensor([t]).to(device))
    noisy_image = (x_noisy.squeeze() + 1) * 127.5
    if idx==0:
        noisy_image = (x_start.squeeze() + 1) * 127.5
    noisy_image = noisy_image.cpu().numpy().astype(np.uint8)
    plt.subplot(1, 5, 1 + idx)
    plt.imshow(noisy_image, cmap='gray')
    plt.axis("off")
    plt.title(f"t={t}")

plt.show()

# train
epochs = 10
p_uncound = 0.2
len_data = len(train_loader)
time_end = time.time()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(epochs):
    for step, (images, labels) in enumerate(train_loader):     
        time_start = time_end
        
        optimizer.zero_grad()
        
        batch_size = images.shape[0]
        images = images.to(device)
        labels = labels.to(device)
        
        # random generate mask
        z_uncound = torch.rand(batch_size)
        batch_mask = (z_uncound>p_uncound).int().to(device)
        
        # sample t uniformally for every example in the batch
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()
        
        loss = gaussian_diffusion.train_losses(model, images, t, labels, batch_mask)
        
        if step % 100 == 0:
            time_end = time.time()
            print("Epoch{}/{}\t  Step{}/{}\t Loss {:.4f}\t Time {:.2f}".format(epoch+1, epochs, step+1, len_data, loss.item(), time_end-time_start))
            
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), save_dir + f"model_{epoch}.pth")
    print('saved model at ' + save_dir + f"model_{epoch}.pth")

torch.save(model.state_dict(), save_dir + f"model.pth")

model.load_state_dict(torch.load('./diffusion_ep_models/model.pth'))

gaussian_diffusion = GaussianDiffusion(timesteps=500, beta_schedule='linear')
generated_images = gaussian_diffusion.sample(model, 28, batch_size=64, channels=1, n_class=10, w=2, mode='random', clip_denoised=False)

# generate new images
fig = plt.figure(figsize=(12, 12), constrained_layout=True)
gs = fig.add_gridspec(8, 8)

imgs = generated_images[-1].reshape(8, 8, 28, 28)
for n_row in range(8):
    for n_col in range(8):
        f_ax = fig.add_subplot(gs[n_row, n_col])
        f_ax.imshow((imgs[n_row, n_col]+1.0) * 255 / 2, cmap="gray")
        f_ax.axis("off")

plt.savefig("./draw/generated_images.jpg")



ddim_generated_images = gaussian_diffusion.ddim_sample(model, 28, batch_size=64, channels=1, ddim_timesteps=50, n_class=10,
                                                       w=2, mode='random', ddim_discr_method='quad', ddim_eta=0.0, clip_denoised=False)
# ddim generate new images
fig = plt.figure(figsize=(12, 12), constrained_layout=True)
gs = fig.add_gridspec(8, 8)

imgs = ddim_generated_images.reshape(8, 8, 28, 28)
for n_row in range(8):
    for n_col in range(8):
        f_ax = fig.add_subplot(gs[n_row, n_col])
        f_ax.imshow((imgs[n_row, n_col]+1.0) * 255 / 2, cmap="gray")
        f_ax.axis("off")

plt.savefig("./draw/generated_ddim_images.jpg")

gif_generated_images = gaussian_diffusion.ddim_sample(model, 28, batch_size=40, channels=1, ddim_timesteps=100, n_class=10,
                                                       w=2, mode='all', ddim_discr_method='quad', ddim_eta=0.0, clip_denoised=False)
# ddim generate 0 1 2 3 4 5 6 7 8 9
fig = plt.figure(figsize=(12, 5), constrained_layout=True)
gs = fig.add_gridspec(4, 10)

imgs = gif_generated_images[-1].reshape(4, 10, 28, 28)
for n_row in range(4):
    for n_col in range(10):
        f_ax = fig.add_subplot(gs[n_row, n_col])
        f_ax.imshow((imgs[n_row, n_col]+1.0) * 255 / 2, cmap="gray")
        f_ax.axis("off")

plt.savefig("./draw/generated_class_ddim_images.jpg")        

