# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Fused operators for normalization layers."""

import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from sglang.srt.utils import is_cuda_available

if is_cuda_available():
    from sgl_kernel import (
        fused_add_rmsnorm,
        gemma_fused_add_rmsnorm,
        gemma_rmsnorm,
        rmsnorm,
    )

from sglang.srt.custom_op import CustomOp

import math
import triton
import triton.language as tl
from sglang.srt.utils import libentry
import hcdbg

logger = logging.getLogger(__name__)


# From FlagGems
@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_kernel(
    Y,  # pointer to the output
    X,  # pointer to the input
    W,  # pointer to the weights
    y_stride_r,
    y_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    Y += pid * y_stride_r
    X += pid * x_stride_r

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)

    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = (x * rrms).to(Y.dtype.element_ty) * w
    tl.store(Y + cols * y_stride_c, y, mask=mask)

# Integrated from FlagGems
class _RmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps=1e-5):
        dim = x.ndim - len(normalized_shape)
        M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)

        BLOCK_SIZE = triton.next_power_of_2(N)
        x = x.contiguous()
        weight = weight.contiguous()
        y = torch.empty_like(x)

        rms_norm_kernel[M,](y, x, weight, N, 1, N, 1, N, eps, BLOCK_SIZE)
        return y


def _rms_norm(x, normalized_shape, weight, eps=1e-5):
    return _RmsNorm.apply(x, normalized_shape, weight, eps)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_kernel(
    input_ptr,   # [..., hidden_size]
    residual_ptr,  # [..., hidden_size]
    weight_ptr,  # [hidden_size]
    y_stride_r,
    y_stride_c,
    x_stride_r,  # stride for input rows
    x_stride_c,  # stride for input columns
    N,           # hidden_size
    eps,         # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    input_ptr += pid * y_stride_r
    residual_ptr += pid * y_stride_r

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)

    # Load data from input and residual, then add them together
    x = tl.load(input_ptr + cols * x_stride_c, mask, other=0.0).to(tl.float32)
    r = tl.load(residual_ptr + cols * x_stride_c, mask, other=0.0).to(tl.float32)
    z = x + r
    tl.store(residual_ptr + cols * y_stride_c, z, mask=mask)

    # Compute variance
    var = tl.sum(z * z, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)

    # Load weight and apply RMS normalization
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    normed_z = (z * rrms).to(input_ptr.dtype.element_ty) * weight

    # Store result back to input
    tl.store(input_ptr + cols * y_stride_c, normed_z, mask=mask)


class _FusedAddRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, residual, normalized_shape, weight, eps=1e-5):
        dim = input.ndim - len(normalized_shape)
        M = math.prod(input.shape[:dim])
        N = math.prod(normalized_shape)

        BLOCK_SIZE = triton.next_power_of_2(N)
        input = input.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()

        # Launch the Triton kernel
        fused_add_rms_norm_kernel[(M,)](
            input, residual, weight, N, 1, N, 1, N, eps, BLOCK_SIZE
        )
        return input


def _fused_add_rms_norm(input, residual, normalized_shape, weight, eps=1e-5):
    return _FusedAddRMSNorm.apply(input, residual, normalized_shape, weight, eps)


class RMSNorm(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hcdbg.jack_print(f'hcdbg: (python-sglang) RMSNorm:forward_cuda():') # debug

        if residual is not None:
            fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            return x, residual
        out = rmsnorm(x, self.weight.data, self.variance_epsilon)
        return out

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hcdbg.jack_print(f'hcdbg: (python-sglang) RMSNorm:forward_native():') # debug
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = (x * self.weight).to(orig_dtype)
        if residual is None:
            return x
        else:
            return x, residual

    def forward_triton(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hcdbg.jack_print(f'hcdbg: (python-sglang) RMSNorm:forward_triton(): DONE (Triton)') # debug
        if residual is not None:
            # fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            _fused_add_rms_norm(
                x,
                residual,
                [len(self.weight.data)],
                self.weight.data,
                self.variance_epsilon,
            )
            return x, residual

        # out = rmsnorm(x, self.weight.data, self.variance_epsilon)
        # out = torch.empty_like(x)        
        # This is for vllm-cuda because vllm-cuda's rms_norm requires 'out' to be passed in.  
        # However, it seems that Triton originally did not pass in 'out'.  
        # So I removed it to check the result → The result is correct.  
        # The previous vllm code had redundancy, specifically:  
        # - "out = torch.empty_like(x)"
        # - "from xxx import op"
        #
        # testing the Chinese characters
        # 這是vllm-cuda的, 因為vllm-cuda的rms_norm需要傳入out.
        # 但似乎本來triton就沒傳入out. 所以先拿掉看結果->結果正確 之前vllm推出去的code有冗余,
        # 分別是"out = torch.empty_like(x)"" and "from xxx import op"" 
        out = _rms_norm(
                x,
                [len(self.weight.data)],
                self.weight.data,
                self.variance_epsilon
        )
        return out


class GemmaRMSNorm(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        if residual is not None:
            x = x + residual
            residual = x

        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x * (1.0 + self.weight.float())
        x = x.to(orig_dtype)
        return x if residual is None else (x, residual)

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            gemma_fused_add_rmsnorm(
                x, residual, self.weight.data, self.variance_epsilon
            )
            return x, residual
        out = gemma_rmsnorm(x, self.weight.data, self.variance_epsilon)
        return out


if not is_cuda_available():
    logger.info(
        "sgl-kernel is not available on Non-NV platforms. Fallback to other kernel libraries."
    )
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
