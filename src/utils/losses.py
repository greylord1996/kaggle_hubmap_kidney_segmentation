from torch import nn
from torch.nn import functional as F
from src.segloss.hausdorff import HausdorffDTLoss
from src.segloss.lovasz_loss import LovaszSoftmax


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        # inputs = inputs > 0.5
        targets = targets.view(-1)
        # print("targets = ", targets)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1.0 - dice


class DiceBCELoss(nn.Module):
    # Formula Given above.
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # inputs = inputs > 0.5

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class Hausdorff_loss(nn.Module):
    def __init__(self):
        super(Hausdorff_loss, self).__init__()

    def forward(self, inputs, targets):
        return HausdorffDTLoss()(inputs, targets)


class Lovasz_loss(nn.Module):
    def __init__(self):
        super(Lovasz_loss, self).__init__()

    def forward(self, inputs, targets):
        return LovaszSoftmax()(inputs, targets)
