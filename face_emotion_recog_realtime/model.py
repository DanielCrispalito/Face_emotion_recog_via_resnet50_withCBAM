import torch
import torch.nn as nn
from torchvision import models

# ===== CHANNEL ATTENTION =====
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out) * x


# ===== SPATIAL ATTENTION =====
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out) * x



# ===== CBAM BLOCK =====
class CBAMBlock(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# ===== RESNET50 + LIGHT CBAM =====
class ResNet50_LightCBAM(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )
        last_block = self.resnet.layer4[-1]
        self.resnet.layer4[-1] = nn.Sequential(
            last_block,
            CBAMBlock(last_block.conv3.out_channels)
        )

        self.resnet.fc = nn.Linear(
            self.resnet.fc.in_features, num_classes
        )

    def forward(self, x):
        return self.resnet(x)
