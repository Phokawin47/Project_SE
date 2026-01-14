import torch
import torch.nn as nn
from torchvision import models

def get_efficientnet_model(num_classes, device):
    """
    สร้าง Model EfficientNet-B0 ที่พร้อมสำหรับ Transfer Learning
    """
    # 1. โหลดโมเดลตั้งต้น (Pre-trained)
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.3), 
        nn.Linear(num_ftrs, num_classes)
    )
    
    # 3. ส่งโมเดลไปที่ GPU/CPU
    model = model.to(device)
    return model