import torch
import torch.nn as nn


class vgg16_bn_Geno(nn.Module):
    def __init__(self, numClasses):
        super(vgg16_bn_Geno, self).__init__()
        in_channels = 3

        self.features = nn.Sequential(

            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64), #or 1d?
            nn.Dropout(0.3, inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64), #or 1d?

            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128), #or 1d?
            nn.Dropout(0.4, inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128), #or 1d?

            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256), #or 1d?
            nn.Dropout(0.4, inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256), #or 1d?
            nn.Dropout(0.4, inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256), #or 1d?

            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512), #or 1d?
            nn.Dropout(0.4, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512), #or 1d?
            nn.Dropout(0.4, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512), #or 1d?

            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512), #or 1d?
            nn.Dropout(0.4, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512), #or 1d?
            nn.Dropout(0.4, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512), #or 1d?

            nn.MaxPool2d(kernel_size=14, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Dropout(0.5, inplace=True)

        )

        self.classifierG = nn.Sequential(

            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),

            nn.Dropout(0.5, inplace=True),
            nn.Linear(512, numClasses),

        )

    # Defining the forward pass
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifierG(x)
        return x


