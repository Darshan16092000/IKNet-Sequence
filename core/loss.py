import numpy as np
import torch
import scipy
import torch.nn.functional as F
import torch.nn as nn

from utils.rotations import axis_angle_to_quaternion, quaternion_to_matrix, matrix_to_rotation_6d
EPS = 1e-6


class PoseLoss(nn.Module):
    def __init__(self, use_target_weight):
        super(PoseLoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target):
        B = target.shape[0]
        
        output = output.view(B, -1)
        target = target.view(B, -1)

        loss = self.criterion(output, target)
        return loss.mean() 
