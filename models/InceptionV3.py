import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models.inception import Inception3
import torch.nn.functional as F

class InceptionV3(nn.Module):
    def __init__(self, num_classes=8,
                 dropout=0):
        super(InceptionV3, self).__init__()

        self.stem = self.make_stem()

        self.inceptionA_1 = InceptionA(192, 32)
        self.inceptionA_2 = InceptionA(256, 64)
        self.inceptionA_3 = InceptionA(288, 64) # N, 35,35, 288
        # Reduce spatial dimensions
        self.grid_reduction_a = GridReductionA(288) # N, 17,17, 768

        self.inceptionB_1 = InceptionB(768, 128)
        self.inceptionB_2 = InceptionB(768, 160)
        self.inceptionB_3 = InceptionB(768, 160)
        self.inceptionB_4 = InceptionB(768, 192) # N, 17,17, 768

        self.grid_reduction_b = GridReductionB(768) # N, 8,8, 768

        self.inceptionC_1 = InceptionC(1280)
        self.inceptionC_2 = InceptionC(2048)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(2048, 1000)
        self.fc_out = nn.Linear(1000, num_classes) # Added this layer since num_classes is only 4

        self.init_weights()

    def init_weights(self, kaiming=False):
        if kaiming:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)

        else: # Use PyTorch's truncated normal initialization
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    stddev = float(m.stddev) if hasattr(m, "stddev") else 0.1  # type: ignore
                    torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=stddev, a=-2, b=2)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)                

    def make_stem(self):
        stem = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3),
            BasicConv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            BasicConv2d(64, 80, kernel_size=1),
            BasicConv2d(80, 192, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        return stem
    
    def transform_input(self, x):
        IMGNET_MEAN = [0.485, 0.456, 0.406]
        IMGNET_STD  = [0.229, 0.224, 0.225]

        effective_mean = [(0.5 - m)/s for m, s in zip(IMGNET_MEAN, IMGNET_STD)]
        effective_std  = [0.5/s for s in IMGNET_STD]

        # This transform replicates your current transform_input method
        # assuming input is a tensor in [0,1] range
        custom_normalization_transform = transforms.Normalize(mean=effective_mean, 
                                                              std=effective_std)
        
        return custom_normalization_transform(x)


    def forward(self, x):
        x = self.transform_input(x)

        x = self.stem(x)

        x = self.inceptionA_1(x)
        x = self.inceptionA_2(x)
        x = self.inceptionA_3(x)

        x = self.grid_reduction_a(x)

        x = self.inceptionB_1(x)
        x = self.inceptionB_2(x)
        x = self.inceptionB_3(x)
        x = self.inceptionB_4(x)

        x = self.grid_reduction_b(x)

        x = self.inceptionC_1(x)
        x = self.inceptionC_2(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.fc_out(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs) # Could add padding_mode='replicate'
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()

        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1) # 1x1 convolution

        self.branch1x3_1 = BasicConv2d(in_channels, 48, kernel_size=1) # 1x1 convolution
        self.branch1x3_2 = BasicConv2d(48, 64, kernel_size=3, padding=1) # 3x3 convolution with padding

        self.branch1x3x3_1 = BasicConv2d(in_channels, 64, kernel_size=1) # 1x1 convolution
        self.branch1x3x3_2 = BasicConv2d(64, 96, kernel_size=3, padding=1) # 3x3 convolution with padding
        self.branch1x3x3_3 = BasicConv2d(96, 96, kernel_size=3, padding=1) # 3x3 convolution with padding

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1) # 1x1 convolution

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch1x3 = self.branch1x3_1(x)
        branch1x3 = self.branch1x3_2(branch1x3)

        branch1x3x3 = self.branch1x3x3_1(x)
        branch1x3x3 = self.branch1x3x3_2(branch1x3x3)
        branch1x3x3 = self.branch1x3x3_3(branch1x3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch1x3, branch1x3x3, branch_pool] #N, 224 + pool_features, 35, 35
        return torch.cat(outputs, 1)

class GridReductionA(nn.Module):
    def __init__(self, in_channels):
        super(GridReductionA, self).__init__()

        self.branch3x3_1 = BasicConv2d(in_channels, in_channels, kernel_size=1, stride=1) # Ta7neka
        self.branch3x3_2 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch1x1_3x3_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch1x1_3x3_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch1x1_3x3_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch1x1_3x3 = self.branch1x1_3x3_1(x)
        branch1x1_3x3 = self.branch1x1_3x3_2(branch1x1_3x3)
        branch1x1_3x3 = self.branch1x1_3x3_3(branch1x1_3x3)

        pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch1x1_3x3, pool]
        return torch.cat(outputs, 1)

class InceptionB(nn.Module):
    def __init__(self, in_channels, channels_7x7):
        super(InceptionB, self).__init__()
        
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)

class GridReductionB(nn.Module):
    def __init__(self, in_channels):
        super(GridReductionB, self).__init__()

        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)

class InceptionC(nn.Module):
    def __init__(self, in_channels):
        super(InceptionC, self).__init__()

        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        # Split branches
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3), 
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=2)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)  

        
