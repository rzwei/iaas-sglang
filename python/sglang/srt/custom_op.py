import torch
from torch import nn

from sglang.srt.server_args import ServerArgs
import os
import hcdbg

_is_cuda = torch.cuda.is_available() and torch.version.cuda
_is_rocm = torch.cuda.is_available() and torch.version.hip


class CustomOp(nn.Module):
    def __init__(self):
        super().__init__()
        self._forward_method = self.dispatch_forward()
        # error
        # hcdbg.jack_print(f'hcdbg: (python-sglang) CustomOp:__init__
        # (): self._forward_method = {self._forward_method}') # debug

    def forward(self, *args, **kwargs):
        # from python.sglang.srt.entrypoints.engine import Engine
        # # print(f'jack123 {Engine.jack_server_args["temperature"]}')
        # print(f'jack123 {Engine.g_server_args}')
        # /data00/jack/sglang_bdiaas-upstream/python/sglang/srt/entrypoints/engine.py
        # from python.sglang.srt.entrypoints.engine import g_server_args
        # /data00/jack/sglang_bdiaas-upstream/python/sglang/srt/managers/scheduler.py
        #from python.sglang.srt.managers.scheduler import g_server_args
        # /data00/jack/sglang_bdiaas-upstream/python/sglang/srt/model_executor/model_runner.py
        from python.sglang.srt.model_executor.model_runner import g_server_args
        # print(f'jack123 {g_server_args}')

        # print(f'jack123 {Engine.jack_server_args["temperature"]}')
        # print(f'jack456 {ServerArgs.attention_backend} {ServerArgs.use_triton_primative_kernels}')

        hcdbg.jack_print(f'hcdbg: (python-sglang) CustomOp:forward(): self._forward_method = {self._forward_method}') # debug

        # import sglang as sg
        # server_args = sg.get_context().server_args
        # server_args
        # print(f"Global Server Args: {.server_args}")
        # if ServerArgs.use_triton_primative_kernels:
        #     hcdbg.jack_print(f'hcdbg: (python-sglang) CustomOp:forward(): ✅IS use_triton_primative_kernels') # debug
        # else:
        #     hcdbg.jack_print(f'hcdbg: (python-sglang) CustomOp:forward(): ❌NOT use_triton_primative_kernels') # debug

        # if ServerArgs.enable_torch_compile:
        #     hcdbg.jack_print(f'hcdbg: test1 ✅') # debug
        # else:
        #     hcdbg.jack_print(f'hcdbg: test1 ❌') # debug

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
        if os.environ.get("VLLM_USE_TRITON_NON_ATTN", "").lower() in ("true", "1"):
            return self.forward_triton
            
        if _is_cuda:
            return self.forward_cuda
        elif _is_rocm:
            return self.forward_hip
        else:
            return self.forward_native
