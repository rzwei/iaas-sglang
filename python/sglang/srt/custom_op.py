import torch
from torch import nn

import os

_is_cuda = torch.cuda.is_available() and torch.version.cuda
_is_rocm = torch.cuda.is_available() and torch.version.hip


class CustomOp(nn.Module):
    def __init__(self):
        super().__init__()
        self._forward_method = self.dispatch_forward()

    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    def forward_native(self, *args, **kwargs):
        raise NotImplementedError

    def forward_cuda(self, *args, **kwargs):
        raise NotImplementedError

    def forward_triton(self, *args, **kwargs):
        raise NotImplementedError

    def forward_hip(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)

    def forward_xpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_hpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_triton(self, *args, **kwargs):
        return self.forward_triton(*args, **kwargs)

    def dispatch_forward(self):
        if os.environ.get("SGL_USE_TRITON_NON_ATTN", "").lower() in ("true", "1"):
            return self.forward_triton

        if _is_cuda:
            return self.forward_cuda
        elif _is_rocm:
            return self.forward_hip
        else:
            return self.forward_native
