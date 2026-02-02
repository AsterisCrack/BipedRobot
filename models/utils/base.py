import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional

class Torso(nn.Module, ABC):
    """Abstract base class for network backbones (torsos)."""
    def __init__(self, observation_normalizer: Optional[nn.Module] = None):
        super().__init__()
        self.observation_normalizer = observation_normalizer

    @abstractmethod
    def forward(self, x, **kwargs):
        pass

    def normalize(self, x):
        if self.observation_normalizer is not None:
            return self.observation_normalizer(x)
        return x

class Head(nn.Module, ABC):
    """Abstract base class for output heads."""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

    @abstractmethod
    def forward(self, x, **kwargs):
        pass
