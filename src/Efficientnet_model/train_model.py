import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import os

# Import โมเดลจาก model.py
from model import get_efficientnet_model

# ================= CONFIG =================
DATA_DIR = 'Dataset_Cleaned'
BATCH_SIZE = 16           # ปรับเป็น 16 เพื่อแก้ CUDA Out of Memory
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOAD_CHECKPOINT = True    # <--- เปิดระบบโหลดเซฟเก่า

# ================= DATA WRAPPER =================
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

# ================= PREPARE DATA =================
def get_dataloaders():
    # Train Transform
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Val Transform
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not os.path.exists(DATA_DIR):
        print(f"Error: ไม่เจอโฟลเดอร์ {DATA_DIR}")
        exit()

    full_dataset = datasets.ImageFolder(root=DATA_DIR)
    classes = full_dataset.classes
    print(f"Classes: {classes}")

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

# ================= TRAINING LOOP =================
def train():
    print(f"Using Device: {DEVICE}")
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    
    model = get_efficientnet_model(len(class_names), DEVICE)
    
    # --- ส่วนที่เพิ่ม: โหลดเซฟเก่า ---
    save_dir = os.path.join('weights', 'efficientnet')
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    last_model_path = os.path.join(save_dir, 'last_model.pth')

    if LOAD_CHECKPOINT and os.path.exists(last_model_path):
        print(f"Found checkpoint: {last_model_path}")
        print("resume training...")
        try:
            # weights_only=False แก้ warning
            model.load_state_dict(torch.load(last_model_path, map_location=DEVICE, weights_only=False))
        except Exception as e:
            print(f"Load failed ({e}). Starting fresh.")
    # --------------------------------

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_loss = float('inf')
    
    print("-" * 30)
    print("🚀 Starting Training...")
    
    for epoch in range(NUM_EPOCHS):
        # --- Train ---
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
        
        # --- Validate ---
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
        
        # --- ส่วนที่แก้: เพิ่ม Train Loss ใน Print ---
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.4f}")
        
        # Save Best
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f"-->Saved Best Model (Loss: {best_loss:.4f})")
            
        # Save Last
        torch.save(model.state_dict(), last_model_path)

if __name__ == "__main__":
    train()