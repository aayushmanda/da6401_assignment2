import os
import wandb
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import yaml

# Define directories
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


categories=['Amphibia','Animalia','Arachnida','Aves','Fungi','Insecta','Mammalia','Mollusca','Plantae','Reptilia']
train_dir = config['train_dir']
test_dir = config['test_dir']
batch_size = config['batch_size']

# Define transformations with additional augmentations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]),
}

# Create datasets
image_datasets = {
    'train': datasets.ImageFolder(train_dir, data_transforms['train']),
    'val': datasets.ImageFolder(test_dir, data_transforms['val'])
}

# Split validation dataset into two equal halves
val_dataset = image_datasets['val']
val_size = len(val_dataset)
half_size = val_size // 2  # Integer division to split evenly
# If val_size is odd, one half will have one more sample
val_half1, val_half2 = random_split(val_dataset, [half_size, val_size - half_size])

# Create dataloaders for train, val_half1, and val_half2
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
    'val': DataLoader(val_half1, batch_size=batch_size, shuffle=False, num_workers=2),
    'test': DataLoader(val_half2, batch_size=batch_size, shuffle=False, num_workers=2)
}

# Get dataset sizes and class names
dataset_sizes = {
    'train': len(image_datasets['train']),
    'val': len(val_half1),
    'test': len(val_half2)
}
class_names = image_datasets['train'].classes

# Set device for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Print summary information
print(f"Training dataset size: {dataset_sizes['train']}")
print(f"Validation half 1 size: {dataset_sizes['val']}")
print(f"Validation half 2 size: {dataset_sizes['test']}")
print(f"Class names: {class_names}")
print(f"Using device: {device}")

import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleCNN(nn.Module):
    def __init__(self,
                 input_channels=3,
                 num_classes=10,
                 num_conv_layers=5,
                 conv_filters=32,
                 filter_size=3,
                 activation_func=nn.ReLU,
                 dense_neurons=512,
                 dropout_rate=0.25):

        super().__init__()

        self.conv_blocks = nn.ModuleList()
        in_channels = input_channels

        # Adding convolution-activation-batchnorm-maxpool blocks
        for _ in range(num_conv_layers):
            self.conv_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=conv_filters,
                          kernel_size=filter_size, stride=1, padding=filter_size//2),
                nn.BatchNorm2d(conv_filters),  # Adding Batch Normalization
                activation_func(),
                nn.MaxPool2d(kernel_size=2)
                # nn.Dropout2d(0.4)
            ))
            in_channels = conv_filters

        # Flatten layer
        self.flatten = nn.Flatten()

        # Calculating input size for the first fully connected layer
        input_size = conv_filters * (224 // (2**num_conv_layers))**2  # Calculate input size dynamically

        # Fully connected layers with Dropout
        self.fc1 = nn.Linear(input_size, dense_neurons)  # Use calculated input size
        self.dropout = nn.Dropout(dropout_rate)  # Adding Dropout layer
        self.fc2 = nn.Linear(dense_neurons, num_classes)

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply Dropout during training
        x = self.fc2(x)
        return x

# Example instantiation
model = FlexibleCNN(input_channels=3, num_classes=10, num_conv_layers=5,
                    conv_filters=48, filter_size=5, activation_func=nn.ReLU,
                    dense_neurons=512)

print(model)


import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim

# Hyperparameters
num_epochs = config["epoch"]  # Number of epochs to train
learning_rate = config["learning_rate"]  # Learning rate 

# Load pre-trained ResNet50
# model = models.resnet50(weights=True)
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)

# Ensure the new layer is trainable
for param in model.fc.parameters():
    param.requires_grad = True

# Set up the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Loss function for multi-class classification
optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)  # Optimize only the final layer

# Move model to the appropriate device (assuming device is defined, e.g., 'cuda' or 'cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training loop with test evaluation
lossi = []
best_val_acc = 0.0

for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    print('-' * 10)

    # Training and validation phases
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    lossi.append(loss.item())

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        print(f'{phase} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

        # Track best validation accuracy
        if phase == 'val' and epoch_acc > best_val_acc:
            best_val_acc = epoch_acc



plt.plot(torch.tensor(lossi))