import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from PIL import Image
from tqdm import tqdm
import glob
import random

# Define paths
DATA_DIRS = ['dataset_resplit_aug_1', 'dataset_resplit_aug_2']
CHECKPOINT_PATH = 'last_model.pth'
BEST_MODEL_PATH = 'best_model.pth'
ONNX_MODEL_PATH = 'best_model.onnx'
CONFUSION_MATRIX_PATH = 'confusion_matrix.png'

# Parameters
NUM_CLASSES = 6
BATCH_SIZE = 16
NUM_EPOCHS = 100 
PATIENCE = 20
LEARNING_RATE = 0.001
THRESHOLD_ACCURACY = 0.80

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class YoloCropDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = item['img_path']
        cls = item['cls']
        x_c, y_c, w, h = item['bbox']
        
        try:
            image = Image.open(img_path).convert('RGB')
            img_w, img_h = image.size
            
            # Convert YOLO to Box
            left = (x_c - w / 2) * img_w
            top = (y_c - h / 2) * img_h
            right = (x_c + w / 2) * img_w
            bottom = (y_c + h / 2) * img_h
            
            # Square Crop with Padding logic (similar to inference)
            box_w = right - left
            box_h = bottom - top
            max_dim = max(box_w, box_h)
            center_x = left + box_w / 2
            center_y = top + box_h / 2
            
            padding = 30 # Add extra padding
            half_dim = (max_dim + padding) / 2
            
            left = max(0, center_x - half_dim)
            top = max(0, center_y - half_dim)
            right = min(img_w, center_x + half_dim)
            bottom = min(img_h, center_y + half_dim)

            # Avoid empty crops
            if right - left < 5 or bottom - top < 5:
                print(f"Warning: Invalid crop in {img_path}. Using full image.")
                crop = image
            else:
                crop = image.crop((left, top, right, bottom))
            
            if self.transform:
                crop = self.transform(crop)
                
            return crop, cls
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            dummy = torch.zeros((3, 224, 224))
            return dummy, cls

def scan_all_samples(root_dirs):
    all_samples = []
    splits_to_scan = ['train', 'valid', 'test', 'val'] # Scan everything
    
    print("Scanning all datasets...")
    for root_dir in root_dirs:
        for split in splits_to_scan:
            img_dir_path = os.path.join(root_dir, split, 'images')
            label_dir_path = os.path.join(root_dir, split, 'labels')
            
            if not os.path.exists(img_dir_path):
                continue
                
            image_files = glob.glob(os.path.join(img_dir_path, '*.*'))
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
            image_files = [f for f in image_files if f.lower().endswith(valid_extensions)]
            
            for img_path in image_files:
                basename = os.path.splitext(os.path.basename(img_path))[0]
                label_path = os.path.join(label_dir_path, basename + '.txt')
                
                if os.path.exists(label_path):
                    try:
                        with open(label_path, 'r') as f:
                            lines = f.readlines()
                        
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cls = int(float(parts[0]))
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                                
                                if 0 <= cls < NUM_CLASSES:
                                    all_samples.append({
                                        'img_path': img_path,
                                        'cls': cls,
                                        'bbox': (x_center, y_center, width, height)
                                    })
                    except Exception as e:
                        print(f"Error reading {label_path}: {e}")
                        
    print(f"Total samples found: {len(all_samples)}")
    return all_samples

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25, patience=20):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming from {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # Reset optimizer for new strategy
        print("NOTE: Optimizer and Scheduler states were NOT loaded (Resetting for improvement).")
        
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        print(f"Resuming from epoch {start_epoch}, Best Acc: {best_acc:.4f}")

    epochs_no_improve = 0
    history = {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': []}

    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} Epoch {epoch}"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                # StepLR usually steps here, but ReduceLROnPlateau steps after validation
                if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
                # Step ReduceLROnPlateau here using validation accuracy
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_acc)
                
                if epoch_acc >= THRESHOLD_ACCURACY:
                    print(f"Threshold of {THRESHOLD_ACCURACY} reached/exceeded.")
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), BEST_MODEL_PATH)
                    print(f"New best model saved! Acc: {best_acc:.4f}")
                    epochs_no_improve = 0
                    
                    # Save ONNX
                    dummy_input = torch.randn(1, 3, 224, 224, device=device)
                    try:
                        torch.onnx.export(model, dummy_input, ONNX_MODEL_PATH, verbose=False, input_names=['input'], output_names=['output'])
                        print(f"Model exported to {ONNX_MODEL_PATH}")
                    except Exception as e:
                        print(f"Failed to export ONNX: {e}")
                    
                else:
                    epochs_no_improve += 1
                    print(f"No improvement for {epochs_no_improve} epochs.")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc
        }, CHECKPOINT_PATH)
        
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, history

def plot_confusion_matrix(model, dataloader, class_names, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    print("Generating Confusion Matrix...")
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Conf Matrix"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(CONFUSION_MATRIX_PATH)
    print(f"Confusion matrix saved to {CONFUSION_MATRIX_PATH}")

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Augmentation & Normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)), 
            transforms.RandomRotation(45), # Increased from 30
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)), # Harder zoom (was 0.8)
            transforms.RandomHorizontalFlip(), 
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1), # Increased jitter
            transforms.RandomGrayscale(p=0.1), # 10% chance of B/W
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), # Simulate webcam blur
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            AddGaussianNoise(0., 0.05) 
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 1. Load ALL data
    all_samples = scan_all_samples(DATA_DIRS)
    
    # 2. Shuffle
    random.seed(42)
    random.shuffle(all_samples)
    
    # 3. Split 70/15/15
    total_len = len(all_samples)
    train_len = int(total_len * 0.7)
    val_len = int(total_len * 0.15)
    test_len = total_len - train_len - val_len
    
    train_samples = all_samples[:train_len]
    val_samples = all_samples[train_len:train_len+val_len]
    test_samples = all_samples[train_len+val_len:]
    
    print(f"Split Results: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
    
    # 4. Create Datasets
    train_dataset = YoloCropDataset(train_samples, transform=data_transforms['train'])
    val_dataset = YoloCropDataset(val_samples, transform=data_transforms['val'])
    test_dataset = YoloCropDataset(test_samples, transform=data_transforms['val']) 

    if len(train_dataset) == 0:
        print("Error: No training data found.")
        exit()

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    }
    
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    class_names = [str(i) for i in range(NUM_CLASSES)]
    
    model_ft = models.efficientnet_b0(pretrained=True)
    
    # Increase Dropout
    # EfficientNet B0 classifier is Sequential(Dropout, Linear)
    model_ft.classifier[0].p = 0.5 
    
    num_ftrs = model_ft.classifier[1].in_features
    model_ft.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES) 
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Switch to AdamW
    optimizer_ft = optim.AdamW(model_ft.parameters(), lr=1e-4, weight_decay=1e-4) # Lower LR for AdamW

    # Switch to ReduceLROnPlateau
    exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='max', factor=0.1, patience=5, verbose=True)

    try:
        print("Starting training with AdamW and ReduceLROnPlateau...")
        # Note: train_model needs small adjustment for ReduceLROnPlateau which steps based on metric
        model_ft, history = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                               dataloaders, dataset_sizes, device,
                               num_epochs=NUM_EPOCHS, patience=PATIENCE)
    
        plot_confusion_matrix(model_ft, dataloaders['val'], class_names, device)
        
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving current state...")
        torch.save({
            'model_state_dict': model_ft.state_dict(),
            'optimizer_state_dict': optimizer_ft.state_dict(),
        }, 'interrupted_checkpoint.pth')


