import os
import wandb
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

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

# Example instantiation
from model import FlexibleCNN
model = FlexibleCNN(input_channels=3, num_classes=10, num_conv_layers=5,
                    conv_filters=48, filter_size=5, activation_func=nn.ReLU,
                    dense_neurons=512)

print(model)



import torch.optim as optim
import time

# Hyperparameters
num_epochs = config[""]  # Number of epochs to train
learning_rate = 1e-4  # Learning rate

# Set up the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Loss function for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

# Move model to the appropriate device
model = FlexibleCNN(input_channels=3, num_classes=10, num_conv_layers=5,
                    conv_filters=48, filter_size=5, activation_func=nn.ReLU,
                    dense_neurons=512)

def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

model.apply(initialize_weights)

model.to(device)


# Training loop with test evaluation
lossi = []
best_val_acc = 0.0


######### Training loop #########
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