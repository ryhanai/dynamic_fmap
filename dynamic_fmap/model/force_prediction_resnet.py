import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import ResNet50_Weights, resnet50


class UpSample(nn.Module):
    def __init__(self, target_size):
        super().__init__()
        self.resize = T.Resize(
            target_size, interpolation=T.InterpolationMode.BILINEAR
        )  # adding antialias=True option may result in autograd error

    def forward(self, x):
        return self.resize(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size, strides):
        super().__init__()

        if len(num_filters) == 1:
            num_filters = [num_filters[0], num_filters[0]]

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, num_filters[0], kernel_size, strides[0], padding="same")
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(num_filters[0])
        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], kernel_size, strides[0], padding="same")
        self.bn3 = nn.BatchNorm2d(num_filters[1])
        self.conv3 = nn.Conv2d(in_channels, num_filters[1], kernel_size, strides[0], padding="same")

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu(x1)
        x1 = self.conv1(x1)
        x1 = self.bn2(x1)
        x1 = self.relu(x1)
        x1 = self.conv2(x1)

        x = self.conv3(x)
        x = self.bn3(x)

        x += x1
        return x


class ForcePredictionResNet(nn.Module):
    def __init__(self, device=0, fine_tune_encoder=True):
        super().__init__()

        self.stdev = 0.02
        self.device = device

        self.augmenter = T.Compose(
            [
                # T.ToTensor(),
                T.Resize([360, 512], antialias=True),
                T.RandomAffine(degrees=(-3, 3), translate=(0.03, 0.03)),
                T.ColorJitter(hue=0.1, saturation=0.1),
                T.RandomAutocontrast(),
                T.ColorJitter(contrast=0.1, brightness=0.1),
            ]
        )

        resnet_classifier = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.encoder = torch.nn.Sequential(*(list(resnet_classifier.children())[:-2]))

        if not fine_tune_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.decoder = nn.Sequential(
            ResBlock(2048, [1024, 512], 3, strides=[1, 1]),
            UpSample([24, 32]),
            ResBlock(512, [256, 256], 3, strides=[1, 1]),
            UpSample([48, 64]),
            ResBlock(256, [128, 128], 3, strides=[1, 1]),
            UpSample([96, 128]),
            ResBlock(128, [128, 128], 3, strides=[1, 1]),
            nn.ConvTranspose2d(128, 180, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            T.Resize([120, 120], antialias=True),
            nn.Unflatten(dim=1, unflattened_size=(3, 60)),
        )

        # self.decoder = nn.Sequential(
        #     ResBlock(2048, [1024, 512], 3, strides=[1, 1]),
        #     UpSample([24, 32]),
        #     ResBlock(512, [256, 128], 3, strides=[1, 1]),
        #     UpSample([48, 64]),
        #     ResBlock(128, [64, 64], 3, strides=[1, 1]),
        #     UpSample([96, 128]),
        #     ResBlock(64, [32, 32], 3, strides=[1, 1]),
        #     nn.ConvTranspose2d(32, 30, kernel_size=3, stride=1, padding=1),
        #     nn.Sigmoid(),
        #     T.Resize([80, 80], antialias=True),
        # )

    def forward(self, x):
        if self.training:
            x = self.augmenter(x) + torch.normal(mean=0, std=self.stdev, size=[360, 512]).to(self.device)

        x = self.decoder(self.encoder(x))
        return x
