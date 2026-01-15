import os
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import kagglehub
import torch
import random
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

class CataractDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for label_name in self.classes:
            label_dir = os.path.join(data_dir, label_name)
            for img_name in os.listdir(label_dir):
                self.image_paths.append(os.path.join(label_dir, img_name))
                self.labels.append(self.class_to_idx[label_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def train_and_evaluate(model, optimizer, train_loader, val_loader, criterion, num_epochs, device, patience=5):
    best_val_accuracy = 0.0  # Initialize variable to track the best validation accuracy
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')


        # Checkpoint the model if validation accuracy improved
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            epochs_without_improvement = 0

            
            checkpoint_path = f'mobilenet_v4_{epoch+1:02d}_{val_acc:.3f}.pth' # Changing version to 4 for data augmentations
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')

        else:
            epochs_without_improvement += 1
            print(f'⚠️ No improvement for {epochs_without_improvement} epoch(s)')
        
            if epochs_without_improvement >= patience:
                print(f'🛑 Early stopping triggered after {epoch+1} epochs')
                break


def make_model(learning_rate=0.001, size_inner=500, droprate=0.2):
    model = CataractDetectorMobileNet(
        num_classes=2,
        size_inner=size_inner,
        droprate=droprate
    )
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer

class CataractDetectorMobileNet(nn.Module):
    def __init__(self, size_inner=500, droprate=0.2, num_classes=2):
        super(CataractDetectorMobileNet, self).__init__()
        
        # Load pre-trained MobileNetV2
        self.base_model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Remove original classifier
        self.base_model.classifier = nn.Identity()
        
        # Add custom layers
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.inner = nn.Linear(1280, size_inner)  # New inner layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(droprate)  # Add dropout
        self.output_layer = nn.Linear(size_inner, num_classes)

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.global_avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.inner(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.output_layer(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CataractDetectorMobileNet(num_classes=2)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()


def loader_func(train_dir, test_dir, train_transforms, val_transforms, test_transforms):
    train_dataset = CataractDataset(
    data_dir=train_dir,
    transform=train_transforms
    )
    
    val_dataset = CataractDataset(
        data_dir=train_dir,
        transform=val_transforms
    )
    
    test_dataset = CataractDataset(
        data_dir=test_dir,
        transform=test_transforms
    )
    
    indices = list(range(len(train_dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, random_state=42
    )
    
    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)
    
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader

input_size = 224

# ImageNet normalization values
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Training transforms WITH augmentation 
train_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),

    transforms.RandomRotation(10),

    transforms.GaussianBlur(
    kernel_size=3,
    sigma=(0.1, 1.0)
    ),

    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

val_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

test_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

droprate = 0.2
size_inner = 500
learning_rate = 1e-4 # 0.001
num_epochs = 25


train_loader, val_loader, test_loader = loader_func(train_dir, test_dir, train_transforms, val_transforms, test_transforms)

print(f'\n=== With Data Augmentation with the Final Pipeline using Random Rotation and Gaussian Blur : ===')
model, optimizer = make_model(learning_rate=learning_rate, size_inner=size_inner,droprate=droprate)
train_and_evaluate(model, optimizer, train_loader, val_loader, criterion, num_epochs, device)