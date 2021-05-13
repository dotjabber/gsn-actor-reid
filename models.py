import torch
import torch.nn as nn
import torchvision.models as models


class SimpleNet(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        
        self.core = torch.nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(128, self.embedding_size)
        
    def forward(self, x):
        x = self.core(x)
        x = x.flatten(1)

        x = self.fc(x)
        x = torch.nn.functional.normalize(x)
        return x

class BackboneResnet18(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size

        self.core = models.resnet18(pretrained=True)
        self.core.fc = nn.Linear(in_features=512, out_features=512, bias=True)

        self.fc1 = nn.Linear(in_features=512, out_features=512, bias=True)
        self.fc2 = nn.Linear(in_features=512, out_features=embedding_size, bias=True)

        # unlock
        for p in self.core.parameters():
          p.requires_grad = True

    def forward(self, x):
        x = self.core(x)

        x = self.fc1(x)
        x = self.fc2(x)
        
        x = x.flatten(1)
        x = torch.nn.functional.normalize(x)
        return x

class BackboneResnet152(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size

        self.core = torch.hub.load('pytorch/vision:v0.9.0', 'resnet152', pretrained=True)
        self.core.fc = nn.Linear(in_features=2048, out_features=2048, bias=True)

        self.fc1 = nn.Linear(in_features=2048, out_features=2048, bias=True)
        self.fc2 = nn.Linear(in_features=2048, out_features=embedding_size, bias=True)

        # unlock
        for p in self.core.parameters():
          p.requires_grad = True

    def forward(self, x):
        x = self.core(x)

        x = self.fc1(x)
        x = self.fc2(x)
        
        x = x.flatten(1)
        x = torch.nn.functional.normalize(x)
        return x

class BackboneVgg19(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size

        self.core = models.vgg16(pretrained=True)
        self.core.classifier[6] = nn.Linear(4096, 1024)

        self.fc1 = nn.Linear(in_features=1024, out_features=1024, bias=True)
        self.fc2 = nn.Linear(in_features=1024, out_features=embedding_size, bias=True)

        # unlock
        for p in self.core.parameters():
          p.requires_grad = True

    def forward(self, x):
        x = self.core(x)

        x = self.fc1(x)
        x = self.fc2(x)
        
        x = x.flatten(1)
        x = torch.nn.functional.normalize(x)
        return x