import torch
import torch.nn as nn

class RCAB(nn.Module):
    def __init__(self, channels=96, reduction=8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        # Channel attention
        weights = self.avg_pool(out)
        weights = self.attention(weights)
        out = out * weights

        # Residual scaling
        out = out * 0.1

        # Skip connection
        out = out + residual
        return out


class RCAN(nn.Module):
    def __init__(self, num_blocks=12, channels=96):
        super().__init__()
        # 4-band Sentinel-2 input
        self.head = nn.Conv2d(4, channels, 3, padding=1)

        # RCAB blocks
        self.body = nn.Sequential(
            *[RCAB(channels) for _ in range(num_blocks)]
        )

        self.body_conv = nn.Conv2d(channels, channels, 3, padding=1)

        # 2x upsampling
        self.up1 = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )

        # another 2x = total 4x
        self.up2 = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )

        # 4-band output
        self.tail = nn.Conv2d(channels, 4, 3, padding=1)

    def forward(self, x):
        # Initial feature extraction
        features = self.head(x)
        
        # Global residual
        residual = features

        # RCAB stack
        features = self.body(features)
        features = self.body_conv(features)

        # Global residual connection
        features = features + residual

        # 4x upsampling
        features = self.up1(features)
        features = self.up2(features)

        # Reconstruction
        output = self.tail(features)
        
        return output

# ============================================================
# RCAN v2 CLASSES
# ============================================================

class RCAB_v2(nn.Module):
    def __init__(self, channels=96, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        res = res * self.ca(res)
        return x + res


class ResidualGroup(nn.Module):
    def __init__(self, channels=96, blocks=8):
        super().__init__()
        self.rcabs = nn.Sequential(
            *[RCAB_v2(channels) for _ in range(blocks)]
        )
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        res = self.rcabs(x)
        res = self.conv(res)
        return x + res


class RCAN48(nn.Module):
    def __init__(self, groups=6, blocks_per_group=8, channels=96):
        super().__init__()
        self.head = nn.Conv2d(4, channels, 3, padding=1)

        self.body = nn.Sequential(
            *[ResidualGroup(channels, blocks_per_group) for _ in range(groups)]
        )
        self.body_conv = nn.Conv2d(channels, channels, 3, padding=1)

        self.up1 = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )

        self.tail = nn.Conv2d(channels, 4, 3, padding=1)

    def forward(self, x):
        f0 = self.head(x)
        
        f = self.body(f0)
        f = self.body_conv(f)
        f = f + f0

        f = self.up1(f)
        f = self.up2(f)

        return self.tail(f)
