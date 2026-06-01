
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F

from warnings import filterwarnings
filterwarnings("ignore", category=DeprecationWarning)
filterwarnings("ignore", category=FutureWarning)
filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", category=RuntimeWarning)



# --- Model Definition ---
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average pooling along channel dimension
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        # Max pooling along channel dimension
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate along channel dimension
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        # Apply convolution and sigmoid
        attention = self.sigmoid(self.conv(pooled))
        return x * attention

class DoubleConvWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.attention = SpatialAttention()

    def forward(self, x):
        x = self.double_conv(x)
        x = self.attention(x)
        return x

class ImprovedUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConvWithAttention(in_channels, 64)
        self.enc2 = DoubleConvWithAttention(64, 128)
        self.enc3 = DoubleConvWithAttention(128, 256)
        self.enc4 = DoubleConvWithAttention(256, 512)

        # Decoder
        self.dec4 = DoubleConvWithAttention(512 + 256, 256)
        self.dec3 = DoubleConvWithAttention(256 + 128, 128)
        self.dec2 = DoubleConvWithAttention(128 + 64, 64)
        self.dec1 = nn.Conv2d(64, out_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Dropout layers for regularization
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool(enc1)
        x = self.dropout(x)

        enc2 = self.enc2(x)
        x = self.pool(enc2)
        x = self.dropout(x)

        enc3 = self.enc3(x)
        x = self.pool(enc3)
        x = self.dropout(x)

        x = self.enc4(x)

        # Decoder
        x = self.upsample(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec4(x)

        x = self.upsample(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec3(x)

        x = self.upsample(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec2(x)

        x = self.dec1(x)
        return x

class ImprovedWNet(nn.Module):
    def __init__(self, in_channels, num_classes=5):
        super().__init__()
        self.unet1 = ImprovedUNet(in_channels, num_classes)
        self.unet2 = ImprovedUNet(num_classes, in_channels)

    def forward(self, x):
        # Segmentation (UNet1)
        seg_logits = self.unet1(x)
        seg_probs = F.softmax(seg_logits, dim=1)

        # Reconstruction (UNet2)
        rec = self.unet2(seg_probs)

        return seg_probs, rec
    
