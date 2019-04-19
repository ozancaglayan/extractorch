# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F

from torchvision import models


class DenseNet:
    """Model types are: densenet{121,161,169,201}."""
    def __init__(self, **kwargs):
        self.model_type = kwargs['model_type'].lower()
        print('Creating CNN instance "{}"'.format(self.model_type))

        self.model = getattr(models, self.model_type)(True)

        # Turn on eval mode
        self.model.train(False)

    def set_device(self, device):
        self.model.to(device)

    def __call__(self, x):
        return self.model.features(x)

    def output_shape(self, width, height):
        dummy = torch.zeros(1, 3, width, height, device='cpu')
        out = self(dummy)
        return out.size()[1:]
