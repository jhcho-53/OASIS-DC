"""
Weight initialization utilities.
"""
import torch.nn as nn


def init_conv_weights(module):
    """Initialize convolutional layer weights with Kaiming normal."""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def init_bn_weights(module):
    """Initialize batch normalization weights."""
    if isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def init_weights(module):
    """Initialize weights for common layer types."""
    init_conv_weights(module)
    init_bn_weights(module)


def apply_weight_init(model):
    """Apply weight initialization to all modules in a model."""
    model.apply(init_weights)