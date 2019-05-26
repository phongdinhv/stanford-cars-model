import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torchvision.models import resnet34, resnet18, squeezenet, inception_v3


class ResNet34(BaseModel):

    def __init__(self, num_classes=196, use_pretrained=True):
        super(BaseModel, self).__init__()
        self.model = resnet34(pretrained=use_pretrained)

        # replace last layer with total cars classes
        n_inputs = self.model.fc.in_features
        classifier = nn.Sequential(nn.Linear(n_inputs, num_classes))
        self.model.fc = classifier

    def forward(self, x):

        return self.model(x)


class ResNet18(BaseModel):

    def __init__(self, num_classes=196, use_pretrained=True):
        super(BaseModel, self).__init__()
        self.model = resnet18(pretrained=use_pretrained)

        # replace last layer with total cars classes
        n_inputs = self.model.fc.in_features
        classifier = nn.Sequential(nn.Linear(n_inputs, num_classes))
        self.model.fc = classifier

    def forward(self, x):

        return self.model(x)


class InceptionV3(BaseModel):

    def __init__(self, num_classes=196, use_pretrained=True):
        super(BaseModel, self).__init__()
        self.model = inception_v3(pretrained=use_pretrained)

        # replace last layer with total cars classes
        n_inputs = self.model.fc.in_features
        classifier = nn.Sequential(nn.Linear(n_inputs, num_classes))
        self.model.fc = classifier

    def forward(self, x):
        return self.model(x)


class SqueezeNet(BaseModel):
    def __init__(self, num_classes=196, use_pretrained=False):
        super(BaseModel, self).__init__()
        self.model = squeezenet.squeezenet1_0(pretrained=use_pretrained)

        self.model.num_classes = num_classes

        # # replace last layer with total cars classes
        self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.model.forward(x)
