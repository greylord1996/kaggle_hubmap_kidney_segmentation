from torch import nn
from torch.nn import functional as F


class DiceCoef(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceCoef, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # print("1 inputs = ", inputs)
        inputs = F.sigmoid(inputs)
        # print("2 inputs = ", inputs)
        # print('targets = ', targets)
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        inputs_new = inputs > 0.5
        # print("3 inputs = ", inputs)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        intersection_new = (inputs_new * targets).sum()
        dice_new = (2. * intersection_new + smooth) / (inputs_new.sum() + targets.sum() + smooth)

        return dice, dice_new