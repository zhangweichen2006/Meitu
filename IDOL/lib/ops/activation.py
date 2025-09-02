import math
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd


class _trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)  # cast to float32
    def forward(ctx, x):
        exp_x = torch.exp(x)
        ctx.save_for_backward(exp_x)
        return exp_x

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        exp_x = ctx.saved_tensors[0]
        return g * exp_x.clamp(min=1e-6, max=1e6)


trunc_exp = _trunc_exp.apply


class TruncExp(nn.Module):

    @staticmethod
    def forward(x):
        return _trunc_exp.apply(x)
