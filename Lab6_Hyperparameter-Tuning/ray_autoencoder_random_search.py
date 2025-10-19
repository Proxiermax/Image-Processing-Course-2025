import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Subset,Dataset

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import os
import cv2
from skimage.util import random_noise
from sklearn.model_selection import train_test_split
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import ray
from ray import tune
from ray.air import session
from ray.tune import Tuner
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle

seed = 4912
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


### START CODE HERE ###
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, gauss_noise=False, gauss_blur=False, resize=128, center_crop=128, p=0.5):
        self.p = p
        self.resize = resize
        self.gauss_noise = gauss_noise
        self.gauss_blur = gauss_blur
        self.center_crop = center_crop
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        from PIL import Image
        gt_image = Image.fromarray(image).resize((self.resize, self.resize))
        noisy_image = gt_image.copy()

        if self.gauss_blur and np.random.rand() < self.p:
            kernel_size = np.random.choice(range(3, 12, 2)).item()
            blur_transform = transforms.GaussianBlur(kernel_size=kernel_size)
            noisy_image = blur_transform(noisy_image)

        if self.gauss_noise and np.random.rand() < self.p:
            noisy_image_np = np.array(noisy_image, dtype=np.float32)
            mean = np.random.randint(-50, 51)
            std = 25
            noise = np.random.normal(mean, std, noisy_image_np.shape).astype(np.float32)
            noisy_image_np = np.clip(noisy_image_np + noise, 0, 255)
            noisy_image = Image.fromarray(noisy_image_np.astype(np.uint8))

        image = self.transform(noisy_image)
        gt_image = self.transform(gt_image)

        return image, gt_image
    
### END CODE HERE ###


### START CODE HERE ###
def imshow_grid(images, title="Images", rows=2, cols=4, figsize=(12, 6)):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if rows * cols > 1 else [axes]
    
    for i, img in enumerate(images[:rows*cols]):
        if torch.is_tensor(img):
            img = img.permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'Image {i+1}')
    
    for i in range(len(images), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
### END CODE HERE ###


### START CODE HERE ###
# data_dir = './data/img_align_celeba'
# image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
# image_paths = [os.path.join(data_dir, f) for f in image_files]

# dataset = CustomImageDataset(image_paths, 
#                            gauss_noise=True, 
#                            gauss_blur=True, 
#                            resize=128, 
#                            p=0.7)
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# ### END CODE HERE ###


# ### START CODE HERE ###
# batch, gt_img = next(iter(dataloader))

# print(f"Batch shape: {batch.shape}")
# print(f"Ground truth shape: {gt_img.shape}")

# # imshow_grid(batch, "Noisy/Blurry Images", rows=2, cols=4)
# # imshow_grid(gt_img, "Ground Truth Images", rows=2, cols=4)

# ### END CODE HERE ###


### START CODE HERE ###
class DownSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DownSamplingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class UpSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(UpSamplingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.upsample(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, channels=[64, 128, 256], input_channels=3, output_channels=3):
        super().__init__()
        
        self.conv_in = nn.Conv2d(input_channels, channels[0], kernel_size=3, stride=1, padding=1)
        self.bn_in = nn.BatchNorm2d(channels[0])
        
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.encoder_blocks.append(
                DownSamplingBlock(channels[i], channels[i+1])
            )
        
        self.bottleneck = nn.Conv2d(channels[-1], channels[-1], kernel_size=3, stride=1, padding=1)
        self.bn_bottleneck = nn.BatchNorm2d(channels[-1])
        
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.decoder_blocks.append(
                UpSamplingBlock(channels[i], channels[i-1])
            )
        
        self.conv_out = nn.Conv2d(channels[0], output_channels, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))

        for block in self.encoder_blocks:
            x = block(x)
        x = F.relu(self.bn_bottleneck(self.bottleneck(x)))
        
        for block in self.decoder_blocks:
            x = block(x)
        x = self.sigmoid(self.conv_out(x))
        
        return x

### END CODE HERE ###


### START CODE HERE ###
def train(model, opt, loss_fn, train_loader, test_loader, epochs=10, checkpoint_path=None, device='cpu'):
    print("🤖Training on", device)
    print(f"Training batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    model = model.to(device)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'🚀Training Epoch [{epoch+1}/{epochs}]', unit='batch')
        
        batch_count = 0
        for images, gt in train_bar:
            images, gt = images.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            
            opt.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, gt)
            loss.backward()
            opt.step()
            
            train_loss += loss.item()
            batch_count += 1
            
            train_bar.set_postfix(loss=f'{loss.item():.4f}', batch=f'{batch_count}/{len(train_loader)}')
            
            # if batch_count >= 10:
            #     break
        
        avg_train_loss = train_loss / batch_count
        
        model.eval()
        test_loss = 0.0
        test_count = 0
        
        with torch.no_grad():
            for images, gt in tqdm(test_loader, desc='📄Testing', unit='batch'):
                images, gt = images.to(device, non_blocking=True), gt.to(device, non_blocking=True)
                outputs = model(images)
                loss = loss_fn(outputs, gt)
                test_loss += loss.item()
                test_count += 1
                
                # if test_count >= 5:
                #     break
        
        avg_test_loss = test_loss / test_count
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
        
        if checkpoint_path and epoch == epochs - 1:
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved to {checkpoint_path}")
                
### END CODE HERE ###


### START CODE HERE ###
# data_dir = 'data/img_align_celeba'
# files = os.listdir(data_dir)
# files = [os.path.join(data_dir, file) for file in files if file.endswith('.jpg')]

# files = files[:16]
# print(f"Using {len(files)} images for training")

# train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)

# train_dataset = CustomImageDataset(train_files, gauss_noise=True, gauss_blur=True, resize=128, p=0.7)
# test_dataset = CustomImageDataset(test_files, gauss_noise=True, gauss_blur=True, resize=128, p=0.7)
# trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
# testloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

### END CODE HERE ###


### START CODE HERE ###
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = Autoencoder()
# opt = optim.Adam(model.parameters(), lr=0.001)
# loss_fn = nn.MSELoss()

# train(model, opt, loss_fn, trainloader, testloader, epochs=20, 
#       checkpoint_path='autoencoder_model.pth', device=device)

### END CODE HERE ###

import ray
from ray import tune
from ray.air import session

ray.shutdown()

### START CODE HERE ###
def train_raytune(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Current directory:", os.getcwd())

    data_dir = 'C:/Users/IMG_PRO/Desktop/Image-Processing-Course-2025/Lab6_Hyperparameter-Tuning/data/img_align_celeba'
    #data_dir = "./Desktop/Image-Processing-Course-2025/Lab6_Hyperparameter-Tuning/data/img_align_celeba"
    # data_dir = 'C://Users//IMG_PRO//Desktop//Image-Processing-Course-2025//Lab6_Hyperparameter-Tuning//data//img_align_celeba'
    # data_dir = r"C:\Users\IMG_PRO\Desktop\Image-Processing-Course-2025\Lab6_Hyperparameter-Tuning\data\img_align_celeba"
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, file) for file in files if file.endswith('.jpg')]
    # files = files[:50]
    
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
    
    train_dataset = CustomImageDataset(train_files, gauss_noise=True, gauss_blur=True, 
                                     resize=64, p=0.7)
    test_dataset = CustomImageDataset(test_files, gauss_noise=True, gauss_blur=True, 
                                    resize=64, p=0.7)
    
    trainloader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                           shuffle=True, num_workers=0)
    testloader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                          shuffle=False, num_workers=0)
    
    model = Autoencoder(channels=config['architecture']).to(device)
    
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for images, gt in trainloader:
            images, gt = images.to(device), gt.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, gt)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # if train_batches >= 20:
            #     break
        
        avg_train_loss = train_loss / train_batches
        
        model.eval()
        val_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, gt in testloader:
                images, gt = images.to(device), gt.to(device)
                outputs = model(images)
                loss = criterion(outputs, gt)
                
                val_loss += loss.item()
                val_batches += 1
                
                for i in range(outputs.shape[0]):
                    output_img = outputs[i].cpu().numpy().transpose(1, 2, 0)
                    gt_img = gt[i].cpu().numpy().transpose(1, 2, 0)
                    
                    output_img = np.clip(output_img, 0, 1)
                    gt_img = np.clip(gt_img, 0, 1)
                    
                    total_psnr += psnr(gt_img, output_img, data_range=1.0)
                    total_ssim += ssim(gt_img, output_img, data_range=1.0, 
                                     multichannel=True, channel_axis=2)
                
                # if val_batches >= 10:
                #     break
        
        avg_val_loss = val_loss / val_batches
        avg_psnr = total_psnr / (val_batches * config['batch_size'])
        avg_ssim = total_ssim / (val_batches * config['batch_size'])
        
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     patience_counter = 0
        # else:
        #     patience_counter += 1
        #     if patience_counter >= patience:
        #         print(f"Early stopping at epoch {epoch}")
        #         break

        session.report({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_psnr": avg_psnr,
            "val_ssim": avg_ssim,
        })
        
### END CODE HERE ###

### START CODE HERE ###
ray.shutdown()
ray.init(num_gpus=1 if torch.cuda.is_available() else 0, ignore_reinit_error=True)

config = {
    'architecture': tune.choice([[32, 64, 128], [32, 64, 128, 256]]),
    'lr': tune.randn(1e-3, 1e-4),
    'batch_size': tune.randint(16, 32),
    'num_epochs': tune.randint(10, 20),
    'optimizer': tune.choice(['Adam', 'SGD'])
}

scheduler = ASHAScheduler(
    metric="val_loss",
    mode="min",
    # max_t=100,
    grace_period=55,
    reduction_factor=2
)

tuner = tune.Tuner(
    # train_raytune,
    tune.with_resources(train_raytune, resources={"cpu": 2, "gpu": 0.5}),
    param_space=config,
    tune_config=tune.TuneConfig(
        scheduler=scheduler,
        # num_samples=1,
    ),
    num_samples=80,
    # run_config=ray.air.RunConfig(
    #     name="autoencoder_grid_search",
    #     local_dir="./ray_results"
    # )
)

print("Starting grid search...")
result = tuner.fit()

### END CODE HERE ###

best_result = result.get_best_result(metric="val_loss", mode="min")
# print("Best config is:", best_result.config)
# print("Best result metrics:", {
#     "val_loss": best_result.metrics["val_loss"],
#     "val_psnr": best_result.metrics["val_psnr"], 
#     "val_ssim": best_result.metrics["val_ssim"]
# })

df = result.get_dataframe()
df.to_csv('grid_search_results-lastday.csv', index=False)
# print("Results saved to 'grid_search_results.csv'")

# print("\nTop 3 results by validation loss:")
# top_results = df.nsmallest(3, 'val_loss')
# for i, (idx, row) in enumerate(top_results.iterrows()):
#     print(f"{i+1}. Val Loss: {row['val_loss']:.4f}, PSNR: {row['val_psnr']:.2f}, SSIM: {row['val_ssim']:.3f}")
#     print(f"   Config: arch={row.get('config/architecture', 'N/A')}, lr={row.get('config/lr', 'N/A')}, "
#           f"batch={row.get('config/batch_size', 'N/A')}, opt={row.get('config/optimizer', 'N/A')}")