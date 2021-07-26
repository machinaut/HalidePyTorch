import math
from torch import nn
from torch.autograd import Function
import torch

import vadd_cuda

torch.manual_seed(42)


class VAddFunction(Function):
    @staticmethod
    def forward(A, B):
        return vadd_cuda.forward(A, B)
