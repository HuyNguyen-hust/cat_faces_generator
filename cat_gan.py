import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import os

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as T
from torchvision import datasets
from torchvision.utils import make_grid

img_size = 64
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transforms = T.Compose([
                   T.Resize(img_size),
                   T.CenterCrop(img_size),
                   T.ToTensor(),
                   T.Normalize(mean, std)
])

print(os.listdir())

def denorm(img_tensors):
    return img_tensors * std[0] + mean[0]

def show_images(images, nmax = 64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow = 8).permute(1, 2, 0))

def show_batch(dl, nmax = 64):
    for images, _ in dl:
        show_images(images, nmax)
        break

latent_dim = 128

generator = nn.Sequential(
    # in: latent_size x 1 x 1

    nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    # out: 512 x 4 x 4

    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    # out: 256 x 8 x 8

    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    # out: 128 x 16 x 16

    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    # out: 64 x 32 x 32

    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
    # out: 3 x 64 x 64
)

discriminator = nn.Sequential(
    # in: 3 x 64 x 64

    nn.Conv2d(3, 64, kernel_size = 4, stride = 2, padding = 1, bias = False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2, inplace = True),
    # out: 64 x 32 x 32

    nn.Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 1, bias = False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace = True),
    # out: 128 x 16 x 16

    nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1, bias = False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace = True),
    # out: 256 x 8 x 8

    nn.Conv2d(256, 512, kernel_size = 4, stride = 2, padding = 1, bias = False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2, inplace = True),
    # out: 512 x 4 x 4
     
    nn.Conv2d(512, 1, kernel_size = 4, stride = 1, padding = 0, bias = False),
    # out: 1 x 1 x 1

    nn.Flatten(),
    nn.Sigmoid()
)

xb = torch.randn(batch_size, latent_dim, 1, 1) # random latent tensors
fake_images = generator(xb)
print(fake_images.shape)
show_images(fake_images)

def train_discriminator(real_images, opt_d):
    # Clear discriminator gradients
    opt_d.zero_grad()

    # Pass real images through discriminator
    real_preds = discriminator(real_images)
    real_targets = torch.ones(real_images.size(0), 1)
    real_loss = F.binary_cross_entropy(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()
        
    # Generate fake images
    latent = torch.randn(batch_size, latent_dim, 1, 1)
    fake_images = generator(latent)

    # Pass fake images through discriminator
    fake_targets = torch.zeros(fake_images.size(0), 1)
    fake_preds = discriminator(fake_images)
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()

    # Update discriminator weights
    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()
    return loss.item(), real_score, fake_score

def train_generator(opt_g):
    # Clear generator gradients
    opt_g.zero_grad()
        
    # Generate fake images
    latent = torch.randn(batch_size, latent_dim, 1, 1)
    fake_images = generator(latent)
        
    # Try to fool the discriminator
    preds = discriminator(fake_images)
    targets = torch.ones(batch_size, 1)
    loss = F.binary_cross_entropy(preds, targets)
        
    # Update generator weights
    loss.backward()
    opt_g.step()
        
    return loss.item()


def fit(epochs, lr, start_idx=1):
    torch.cuda.empty_cache()
    
    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []
    
    # Create optimizers
    opt_d = torch.optim.Adam(discriminator.parameters(), lr = lr, betas = (0.5, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), lr = lr, betas = (0.5, 0.999))
    
    for epoch in range(epochs):
        for real_images, _ in tqdm(data_loader):
            # Train discriminator
            loss_d, real_score, fake_score = train_discriminator(real_images, opt_d)
            # Train generator
            loss_g = train_generator(opt_g)
            
        # Record losses & scores
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)
        
        # Log losses & scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch+1, epochs, loss_g, loss_d, real_score, fake_score))
    
        # Save generated images
        save_samples(epoch + start_idx, fixed_latent, show = False)

        #save checkpoint
        '''
        if (start_idx + epoch - 1) % 5 == 0:
          path = '/model/model_after_' + str(epoch + start_idx - 1) + '_epochs.pt'

          torch.save({
            'epoch': epoch,
            'gen_state_dict': generator.state_dict(),
            'dis_state_dict': discriminator.state_dict(),
            'optd_state_dict': opt_d.state_dict(),
            'optg_state_dict': opt_g.state_dict(),
            'losses_g': losses_g,
            'losses_d': losses_d,
            }, path)
        '''
    
    return losses_g, losses_d, real_scores, fake_scores

lr = 0.0002
epochs = 50

history = fit(epochs, lr)


losses_g, losses_d, _, _ = history
plt.plot(idx, losses_g, color = 'red', label = 'generator')
plt.plot(idx, losses_d, color = 'green', label = 'discriminator')
plt.legend()

plt.savefig('loss.jpg')
plt.show()