import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Subset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report


import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset,random_split


import ray
from ray import tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter

# Ensure the environment variable is set before any imports
import os
# os.environ["OMP_NUM_THREADS"] = "64"

def set_seed(seed):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

class InvertIntensity:
    def __call__(self, img):
        """
        Inverts the intensity of an image.

        Args:
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Inverted image.
        """
        if isinstance(img, torch.Tensor):
            return 1 - img
        else:
            return transforms.functional.invert(img) # for PIL image

def load_data(path, seed=42, batch_size = 16):
    # Set the random seed
    set_seed(seed)

    ### START CODE HERE ###
    transform = transforms.Compose([
        transforms.Resize((232, 232)),
        transforms.ToTensor(),
        InvertIntensity(),
        transforms.Pad(padding=231, padding_mode = 'reflect'),
        transforms.RandomAffine(degrees=45, translate=(0.1, 0.1),scale=(0.8, 1.2), shear=45),
        transforms.CenterCrop((224, 224))
    ])

    full_dataset = datasets.ImageFolder(root=path, transform=transform)

    # Calculate the number of samples for training and testing
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    # Split the dataset into training and testing sets
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    ### END CODE HERE ###
    
    return train_loader, test_loader


class MLP(nn.Module):
    def __init__(self,h_dims=[16,32],dropout=0.2,input_size=(3,224,224)):
        super(MLP, self).__init__()
        ### START CODE HERE ###

        self.flatten = nn.Flatten()
        layers = []
        for i, hdim in enumerate(h_dims):
            if i == 0:
                layers.append(nn.Linear(input_size[0] * input_size[1] * input_size[2], h_dims[0]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            else:
                layers.append(nn.Linear(h_dims[i-1], hdim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

        self.hidden_fc = nn.Sequential(*layers)

        self.fc_out = nn.Linear(h_dims[-1], 10)
        ### END CODE HERE ###

    def forward(self, x):
        ### START CODE HERE ###
        flatten = self.flatten(x)
        x1 = self.hidden_fc(flatten)
        out = self.fc_out(x1)
        return out
    


    
class CNN(nn.Module):
    def __init__(self,num_ch=[16,32],h_dims=[16,32],input_size=(3,224,224)):
        super(CNN, self).__init__()
        ### START CODE HERE ###

        features = []
        in_ch = input_size[0]
        for out_ch in num_ch:
            features.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            features.append(nn.ReLU())
            features.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_ch = out_ch

        
        self.features = nn.Sequential(*features)
        self.flatten = nn.Flatten()
        fc_input_size = self._get_input_size_fc1(input_size)
        
        layers = []
        for i, hdim in enumerate(h_dims):
            if i == 0:
                layers.append(nn.Linear(fc_input_size, hdim))
            else:
                layers.append(nn.Linear(h_dims[i-1], hdim))
            layers.append(nn.ReLU())

        self.fc = nn.Sequential(*layers)
        self.fc_out = nn.Linear(h_dims[-1], 10)
        ### END CODE HERE ###

    
    def _get_input_size_fc1(self, input_shape):
        # Create a dummy input tensor with the same shape as the input images
        dummy_input = torch.zeros(1, *input_shape)
        
        # Pass the dummy input through the convolutional layers
        dummy_output = self.features(dummy_input)
        
        # Flatten the output and get the size
        return int(torch.prod(torch.tensor(dummy_output.shape[1:])))

    def forward(self, x):
        conv1 = self.features(x)
        flatten = self.flatten(conv1)
        fc1 = self.fc(flatten)
        out = self.fc_out(fc1)
        return out

def create_model(model_name,num_ch=None,h_dims=None,input_size=(1,100,100)):
    if model_name == 'cnn':
        model = CNN(num_ch=num_ch,h_dims=h_dims,input_size=input_size)
    elif model_name == 'mlp':
        model = MLP(h_dims=h_dims,dropout=0.5,input_size=input_size)
    return model


def train_raytune(config):
    ### START CODE HERE ###
    loss_fn = nn.CrossEntropyLoss()
    image_dir = "/home/pawit/Lab/Lab8/thai-handwriting-number.appspot.com"
    train_loader, test_loader = load_data(image_dir, batch_size=config['batch_size'])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define the model
    if config['model_name'] == 'cnn':
        model = CNN(h_dims=config['h_dims'], num_ch=config['num_ch']).to(device)

    # model = model.to(device)
    # Define the optimizer
    if config['optimizer'] == 'Adam':
        opt = optim.Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer'] == 'SGD':
        opt = optim.SGD(model.parameters(), lr=config['lr'])

    for epoch in range(config['num_epochs']):
        model.train()

        running_loss = 0
        total = 0
        correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            num_classes = 10
            labels = torch.nn.functional.one_hot(labels, num_classes).float()

            opt.zero_grad()
            logits = model(images).to(device)
            outputs = F.softmax(logits, dim=1)
            
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()

            running_loss += loss.item()
            predicted = torch.argmax(outputs, 1)
            total += labels.size(0)
            correct += (predicted == torch.argmax(labels, 1)).sum().item()

        train_accuracy = 100 * correct / total
        avg_training_loss = running_loss /  len(train_loader)

        model.eval()
        test_loss = 0
        total = 0
        correct = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images).to(device)
                outputs = F.softmax(logits, dim=1)

                loss = loss_fn(logits, labels)
                test_loss += loss.item()
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_accuracy = 100 * correct / total
        test_loss = test_loss / len(test_loader)

        session.report({
            "train_loss": avg_training_loss,
            "train_accuracy": train_accuracy,
            "val_loss": test_loss,
            "val_accuracy": test_accuracy,
        })
    ### END CODE HERE ###


print("Start init ray...")

### START CODE HERE ###
ray.init(num_gpus=1)

print("init succesfully")

models_ls = ["cnn"]
n_layers = None
n_nodes = None
h_dims = [[16, 32], [16, 32, 64], [64, 128], [64, 128, 256], [64, 128, 256, 512]]
num_ch = [[16, 32], [16, 32, 64], [64, 128], [64, 128, 256], [64, 128, 256, 512]]
lr = [1e-3]
batch_size = [16, 32]
num_epochs = [100]
opts = ["Adam"]

# h_dims = [[16, 32],]
# lr = [1e-3]
# batch_size = [16, ]
# num_epochs = [2,]
# opts = ["Adam","SGD"]

config = {
    'model_name': tune.grid_search(models_ls),
    'h_dims': tune.grid_search(h_dims),
    'num_ch': tune.grid_search(num_ch),
    'optimizer': tune.grid_search(opts),
    'lr': tune.grid_search(lr),
    'batch_size': tune.grid_search(batch_size),
    'num_epochs': tune.grid_search(num_epochs),
}

scheduler = ASHAScheduler(
    # metric="val_loss",
    # mode="min",
    # max_t=max(num_epochs),
    grace_period=7,
    reduction_factor=2
)

print("start tuner...")

tuner = tune.Tuner(
    tune.with_resources(train_raytune, resources={"cpu": 0, "gpu": 0.5}),
    tune_config=tune.TuneConfig(
        metric="val_loss",
        mode="min",
        scheduler=scheduler,
        # local_dir="/home/pawit/ray_results/ray_cnn_cpu"
        # max_concurrent_trials=2,
        # num_samples=2
    ),
    param_space=config,
)
result = tuner.fit()

### END CODE HERE ###

print("🎉[INFO] Training is done!")
print("Best config is:", result.get_best_result().config)
print("Best result is:", result.get_best_result())
df = result.get_dataframe()
df.to_csv("/home/pawit/Lab/Lab8/ray_cnn_srun.csv")

ray.shutdown()