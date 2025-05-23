import torch.nn as nn
import torch

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                   stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x

class MobileNetV1(nn.Module):
    def __init__(self, num_classes=4):
        super(MobileNetV1, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),  # 224x224 -> 112x112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            DepthwiseSeparableConv(32, 64, stride=1),    # 112x112
            DepthwiseSeparableConv(64, 128, stride=2),   # 112x112 -> 56x56
            DepthwiseSeparableConv(128, 128, stride=1),  # 56x56
            DepthwiseSeparableConv(128, 256, stride=2),  # 56x56 -> 28x28
            DepthwiseSeparableConv(256, 256, stride=1),  # 28x28
            DepthwiseSeparableConv(256, 512, stride=2),  # 28x28 -> 14x14

            # 5x Depthwise Separable blocks with same shape
            DepthwiseSeparableConv(512, 512, stride=1),  # 14x14
            DepthwiseSeparableConv(512, 512, stride=1),  # 14x14
            DepthwiseSeparableConv(512, 512, stride=1),  # 14x14
            DepthwiseSeparableConv(512, 512, stride=1),  # 14x14
            DepthwiseSeparableConv(512, 512, stride=1),  # 14x14

            DepthwiseSeparableConv(512, 1024, stride=2),  # 14x14 -> 7x7
            DepthwiseSeparableConv(1024, 1024, stride=1), # 7x7
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Output: [batch, 1024, 1, 1]
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = MobileNetV1(num_classes=4)
    x = torch.randn(1, 3, 224, 224)  # Example input
    y = model(x)
    print(f"Output shape: {y.shape}")  # Should be [1, 4]