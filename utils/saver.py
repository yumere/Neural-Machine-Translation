import os
import torch
import torch.nn as nn
import torch.optim as optim


def load_checkpoint(filename: str, model: nn.Module, optimizer: optim.Optimizer):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        return start_epoch, model, optim
