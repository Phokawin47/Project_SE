import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import copy
import os

# ==========================================
# 1. SETTINGS
# ==========================================
DATA_DIR = 'Dataset_Cleaned' 
BATCH_SIZE = 32
LEARNING_RATE = 0.00005 
WEIGHT_DECAY = 1e-4 
NUM_EPOCHS = 50       
EARLY_STOP_PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOAD_FROM_CHECKPOINT = True  # <--- ตั้งเป็น True เพื่อให้โหลดอันเก่ามาเทรนต่อ

# ==========================================
# 2. DATA PREPARATION
# ==========================================
class SubsetWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.subset)

def get_dataloaders():
    print(f"Loading data from: {DATA_DIR}")
    
    # Config เดิมที่ทำให้ Loss ลงดีแล้ว (ยังไม่ต้องเปิด Normalize)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    try:
        full_dataset = datasets.ImageFolder(root=DATA_DIR)
    except FileNotFoundError:
        print(f"ERROR: ไม่เจอโฟลเดอร์ {DATA_DIR}")
        exit()
        
    classes = full_dataset.classes
    print(f"Found classes: {classes}")

    total = len(full_dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    train_data = SubsetWrapper(train_subset, transform=train_transform)
    val_data   = SubsetWrapper(val_subset,   transform=val_transform)
    test_data  = SubsetWrapper(test_subset,  transform=val_transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, classes

# ==========================================
# 3. TRAINING
# ==========================================
def train_model():
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    
    print("Initializing EfficientNet...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, len(class_names))
    )
    
    model = model.to(DEVICE)
    
    # --- ส่วนที่เพิ่มมา: เช็คและโหลดโมเดลเก่า ---
    save_dir = os.path.join('weights', 'efficientnet')
    last_model_path = os.path.join(save_dir, 'last_model.pth')
    
    if LOAD_FROM_CHECKPOINT and os.path.exists(last_model_path):
        print(f"\n!!! Found checkpoint: {last_model_path}")
        print("!!! Resuming training from last save...")
        model.load_state_dict(torch.load(last_model_path))
    else:
        print("\n!!! No checkpoint found (or LOAD_FROM_CHECKPOINT=False). Starting FRESH training.")
    # ----------------------------------------

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_loss = float('inf')
    counter = 0
    
    print("-" * 30)
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        print(f"Val   Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

        # Save Logic
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        torch.save(model.state_dict(), last_model_path) # Save ล่าสุดเสมอ

        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f"--> Saved BEST (Loss: {best_loss:.4f})")
            counter = 0
        else:
            counter += 1
            print(f"--> No improvement ({counter}/{EARLY_STOP_PATIENCE})")

        if counter >= EARLY_STOP_PATIENCE:
            print("Early Stopping!")
            break
        print("-" * 30)

    return class_names

if __name__ == '__main__':
    train_model()