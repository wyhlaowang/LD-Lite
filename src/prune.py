import torch
import torch.nn as nn
from model import *


def compute_l1_norm(weight):
    return weight.abs().sum(dim=(1, 2, 3))


def prune_conv(layer, prune_in=True, prune_out=True, pruning_ratio=0.2):
    weight = layer.weight.data
    bias = layer.bias.data if layer.bias is not None else None

    in_l1_norm = compute_l1_norm(weight.transpose(1,0))
    out_l1_norm = compute_l1_norm(weight) 

    if prune_in:
        prune_in_channels = torch.argsort(in_l1_norm, descending=True)[:max(int(len(in_l1_norm)*(1-pruning_ratio)), 1)]
    else:
        prune_in_channels = torch.argsort(in_l1_norm, descending=True)

    if prune_out:
        prune_out_channels = torch.argsort(out_l1_norm, descending=True)[:max(int(len(out_l1_norm)*(1-pruning_ratio)), 1)]
    else:
        prune_out_channels = torch.argsort(out_l1_norm, descending=True)
        
    pruned_layer = nn.Conv2d(
        in_channels=len(prune_in_channels),
        out_channels=len(prune_out_channels),
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        groups=layer.groups,
        bias=True if bias is not None else False
    )

    pruned_weight = weight[prune_out_channels, :, :, :][:, prune_in_channels, :, :]
    pruned_layer.weight.data = pruned_weight

    if bias is not None:
        pruned_layer.bias.data = bias[prune_out_channels]

    return pruned_layer, prune_in_channels, prune_out_channels


def prune_convtranspose(layer, prune_in=True, prune_out=True, pruning_ratio=0.2):
    weight = layer.weight.data
    bias = layer.bias.data if layer.bias is not None else None

    in_l1_norm = compute_l1_norm(weight)
    out_l1_norm = compute_l1_norm(weight.transpose(1,0))

    if prune_in:
        num_in = max(int((len(in_l1_norm)//2)*(1-pruning_ratio))*2, 1)
        prune_in_channels = torch.argsort(in_l1_norm, descending=True)[:num_in]
    else:
        prune_in_channels = torch.argsort(in_l1_norm, descending=True)

    if prune_out:
        num_out = max(int((len(out_l1_norm)//2)*(1-pruning_ratio))*2, 1)
        prune_out_channels = torch.argsort(out_l1_norm, descending=True)[:num_out]
    else:
        prune_out_channels = torch.argsort(out_l1_norm, descending=True)

    pruned_layer = nn.ConvTranspose2d(
        in_channels=len(prune_in_channels),
        out_channels=weight.shape[1],
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        output_padding=layer.output_padding,
        groups=layer.groups,
        bias=True if bias is not None else False
    )

    pruned_weight = weight[prune_in_channels, :, :, :][:, prune_out_channels, :, :]
    pruned_layer.weight.data = pruned_weight

    if bias is not None:
        pruned_layer.bias.data = bias[prune_out_channels]

    return pruned_layer, prune_in_channels, prune_out_channels


def prune_layernorm(layer, prune_channels):
    gamma = layer.gamma
    beta = layer.beta

    pruned_gamma = gamma[prune_channels]
    pruned_beta = beta[prune_channels]

    pruned_layer = LayerNorm(num_features=len(pruned_gamma))
    pruned_layer.gamma = torch.nn.Parameter(pruned_gamma)
    pruned_layer.beta = torch.nn.Parameter(pruned_beta)

    return pruned_layer


def prune_residual_block(module, pruning_ratio):
    for name, submodule in module.named_children():
        if isinstance(submodule, nn.Conv2d):
            pruned_layer = prune_conv(submodule, pruning_ratio=pruning_ratio)[0]
            setattr(module, name, pruned_layer)


def prune_module(module, pruning_ratio):
    for name, submodule in module.named_children():
        if isinstance(submodule, nn.Conv2d):
            pruned_layer = prune_conv(submodule, pruning_ratio=pruning_ratio)[0]
            setattr(module, name, pruned_layer)
        elif isinstance(submodule, nn.ConvTranspose2d):
            convt_name = name
            pruned_layer, _, prune_out_channels = prune_convtranspose(submodule, pruning_ratio=pruning_ratio)
            setattr(module, name, pruned_layer)
        elif isinstance(submodule, LayerNorm) and int(name) == int(convt_name)+1:
            pruned_layer = prune_layernorm(submodule, prune_out_channels)
            setattr(module, name, pruned_layer)
        elif isinstance(submodule, ResidualBlock):
            pruned_layer = prune_residual_block(submodule.block, pruning_ratio=pruning_ratio)


def prune_model(model, pruning_ratio):
    prune_module(model.encoder.content_encoder1.model, pruning_ratio)
    prune_module(model.encoder.content_encoder2.model, pruning_ratio)
    prune_module(model.decoder.model_up, pruning_ratio)

    return model



