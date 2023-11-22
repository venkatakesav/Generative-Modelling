# %%
import torch
import torchvision
# import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# %%
import torch
import torch.nn.functional as F

# Write the Stdout to a file
# import sys
# sys.stdout = open('output.txt','w')

class DiffusionModel:
    def __init__(self, timesteps, start=0.0001, end=0.02):
        self.timesteps = timesteps
        self.beta_schedule = torch.linspace(start, end, timesteps)

        # Pre-calculate different terms for closed form
        self.alphas = 1. - self.beta_schedule
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.beta_schedule * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def linear_beta_schedule(self, start=0.0001, end=0.02):
        return torch.linspace(start, end, self.timesteps)

    @staticmethod
    def get_index_from_list(vals, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_diffusion_sample(self, x_0, t, device="cpu"):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        """
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        # mean + variance
        return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

# Example usage:
T = 300
diffusion_model = DiffusionModel(timesteps=T)
beta_schedule = diffusion_model.linear_beta_schedule()

# %%
from torchvision import transforms 
from torch.utils.data import DataLoader
import numpy as np

IMG_SIZE = 64
BATCH_SIZE = 128

# def show_tensor_image(image):
#     reverse_transforms = transforms.Compose([
#         transforms.Lambda(lambda t: (t + 1) / 2),
#         transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
#         transforms.Lambda(lambda t: t * 255.),
#         transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
#         transforms.ToPILImage(),
#     ])

#     # Take first image of batch
#     if len(image.shape) == 4:
#         image = image[0, :, :, :] 
#     plt.imshow(reverse_transforms(image))

# %%
data_transforms = [
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # Scales data into [0,1] 
    transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
]
data_transform = transforms.Compose(data_transforms)

train = torchvision.datasets.CIFAR10(root="./data", download=True, 
                                     transform=data_transform, train=True)

test = torchvision.datasets.CIFAR10(root="./data", download=True, 
                                     transform=data_transform, train=False)

data = torch.utils.data.ConcatDataset([train, test])

dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# %%
from torch import nn
import math


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3 
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])
        
        # Edit: Corrected a bug found by Jakub C (see YouTube comment)
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
        return self.output(x)

model = SimpleUnet()
print("Num params: ", sum(p.numel() for p in model.parameters()))
model

# %%
def get_loss(model, x_0, t):
    x_noisy, noise = diffusion_model.forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)

# %%
@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = diffusion_model.get_index_from_list(beta_schedule, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = diffusion_model.get_index_from_list(
        diffusion_model.sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = diffusion_model.get_index_from_list(diffusion_model.sqrt_recip_alphas, t, x.shape)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = diffusion_model.get_index_from_list(diffusion_model.posterior_variance, t, x.shape)
    
    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

# @torch.no_grad()
# def sample_plot_image():
#     # Sample noise
#     img_size = IMG_SIZE
#     img = torch.randn((1, 3, img_size, img_size), device=device)
#     plt.figure(figsize=(15,15))
#     plt.axis('off')
#     num_images = 10
#     stepsize = int(T/num_images)

#     for i in range(0,T)[::-1]:
#         t = torch.full((1,), i, device=device, dtype=torch.long)
#         img = sample_timestep(img, t)
#         # Edit: This is to maintain the natural range of the distribution
#         img = torch.clamp(img, -1.0, 1.0)
#         if i % stepsize == 0:
#             plt.subplot(1, num_images, int(i/stepsize)+1)
#             show_tensor_image(img.detach().cpu())
#     plt.show()            

save_folder = "saved_images"
os.makedirs(save_folder, exist_ok=True)  # Create the folder if it doesn't exist

@torch.no_grad()
def save_sample_image(epoch_num):
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)
    num_images = 10
    stepsize = int(T / num_images)

    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            pil_img = transforms.ToPILImage()(img[0].detach().cpu())
            pil_img.save(f"{save_folder}/image_{epoch_num}_timestep_{i}.png")

# %%
from torch.optim import Adam

# Write the Losses to a file
with open('diffusion.txt','w') as f:
    f.write("Losses: \n")

model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 50 # Try more!

for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
      optimizer.zero_grad()

      t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
      loss = get_loss(model, batch[0], t)
      loss.backward()
      optimizer.step()
      print("{step} of {total_steps} | Loss: {loss}".format(step=step, total_steps=len(dataloader), loss=loss.item()))
      # Write the same print statement to the file
      with open('diffusion.txt','a') as f:
        f.write("{step} of {total_steps} | Loss: {loss}".format(step=step, total_steps=len(dataloader), loss=loss.item()))
        f.write("\n")

    #   if epoch % 10 == 0 and step == 0:
    save_sample_image(epoch)
    print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")


