import torch
from torch.distributions import beta, uniform
from torch.nn.functional import one_hot

class MixUp:
    def __init__(self, num_classes, alpha=0.5, uniform_range=None, seed=123):
        None


    def __call__(self, images, labels):
        images = None
        labels = None

        return images, labels