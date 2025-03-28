import argparse
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, TypedDict
from tqdm import tqdm

import torch
import triton
from transformers import AutoConfig

from sglang.srt.layers.moe.ep_moe.kernels import (
    grouped_gemm_triton,
    get_config_dtype_str,
    get_config_file_name,
    get_default_grouped_gemm_config,
    get_grouped_gemm_configs,
)
from sglang.srt.utils import is_hip
from sglang.srt.layers.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
    per_token_group_quant_fp8,
)

_is_hip_ = is_hip()
_is_cuda = not _is_hip_

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tuning.log'),
        logging.StreamHandler()
    ]
)

class BenchmarkConfig(TypedDict):
    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    BLOCK_SIZE_K: int
    num_warps: int
    num_stages: int

def initialize_scales(a, b, block_shape):
    """Initialize scale tensors for block-wise quantization."""
    if block_shape is None:
        return None, None

    block_n, block_k = block_shape
    device = a.device
    dtype = a.dtype

    # Initialize scale_a
    num_blocks_k = triton.cdiv(a.shape[-1], block_k)
    scale_a = torch.empty((a.shape[0], num_blocks_k), device=device, dtype=dtype)
    logging.info(f"Initialized scale_a with shape: {scale_a.shape}")

    # Initialize scale_b
    num_blocks_n = triton.cdiv(b.shape[-2], block_n)
    num_experts = b.shape[0]
    scale_b = torch.empty((num_experts, num_blocks_n, num_blocks_k), device=device, dtype=dtype)
    logging.info(f"Initialized scale_b with shape: {scale_b.shape}")

    return scale_a, scale_b

def validate_tensors(a, b, c, block_shape=None):
    """Validate tensor shapes and block shape compatibility."""
    if a is None or b is None or c is None:
        raise ValueError("Input tensors cannot be None")
    
    if block_shape is not None:
        if len(block_shape) != 2:
            raise ValueError(f"block_shape must have length 2, got {len(block_shape)}")
        block_n, block_k = block_shape
        
        if a.shape[-1] % block_k != 0:
            raise ValueError(f"Input tensor a's last dimension ({a.shape[-1]}) must be divisible by block_k ({block_k})")
        if b.shape[-2] % block_n != 0:
            raise ValueError(f"Weight tensor b's second-to-last dimension ({b.shape[-2]}) must be divisible by block_n ({block_n})")
        if b.shape[-1] % block_k != 0:
            raise ValueError(f"Weight tensor b's last dimension ({b.shape[-1]}) must be divisible by block_k ({block_k})")

def benchmark_config(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    batch_size: int,
    weight_column_major: bool,
    seg_indptr: Optional[torch.Tensor] = None,
    weight_indices: Optional[torch.Tensor] = None,
    use_fp8_w8a8: bool = False,
    scale_a: Optional[torch.Tensor] = None,
    scale_b: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    config: Optional[BenchmarkConfig] = None,
    num_iters: int = 10,
) -> float:
    try:
        # Validate inputs
        validate_tensors(a, b, c, block_shape)
        logging.info(f"Input validation passed. Shapes: a={a.shape}, b={b.shape}, c={c.shape}")
        if block_shape:
            logging.info(f"Using block_shape: {block_shape}")

        torch.cuda.empty_cache()

        # Initialize scales if needed
        if block_shape is not None and (scale_a is None or scale_b is None):
            logging.info("Initializing scale tensors...")
            scale_a, scale_b = initialize_scales(a, b, block_shape)

        # Handle FP8 data type
        if use_fp8_w8a8 and block_shape is not None:
            try:
                logging.info("Attempting FP8 quantization...")
                logging.info(f"Input shapes before quantization: a={a.shape}, scale_a={scale_a.shape if scale_a is not None else None}")
                a_quant, scale_a = per_token_group_quant_fp8(a, block_shape[1])
                a = a_quant
                logging.info(f"FP8 quantization successful. New shapes: a={a.shape}, scale_a={scale_a.shape}")
            except Exception as e:
                logging.error(f"FP8 quantization failed: {e}")
                raise

        # Warmup
        logging.info("Starting warmup phase...")
        for i in range(3):
            try:
                logging.info(f"Warmup iteration {i+1}: scale_a shape={scale_a.shape if scale_a is not None else None}, scale_b shape={scale_b.shape if scale_b is not None else None}")
                grouped_gemm_triton(
                    a,
                    b,
                    c,
                    batch_size,
                    weight_column_major,
                    seg_indptr,
                    weight_indices,
                    use_fp8_w8a8,
                    scale_a,
                    scale_b,
                    block_shape,
                )
                logging.info(f"Warmup iteration {i+1} successful")
            except Exception as e:
                logging.error(f"Error during warmup iteration {i+1}: {e}")
                raise

        # Benchmark
        logging.info("Starting benchmark phase...")
        try:
            torch.cuda.synchronize()
            start = time.time()
            for i in range(num_iters):
                grouped_gemm_triton(
                    a,
                    b,
                    c,
                    batch_size,
                    weight_column_major,
                    seg_indptr,
                    weight_indices,
                    use_fp8_w8a8,
                    scale_a,
                    scale_b,
                    block_shape,
                )
                if i % 5 == 0:
                    logging.info(f"Completed benchmark iteration {i+1}/{num_iters}")
            torch.cuda.synchronize()
            end = time.time()
            avg_time = (end - start) / num_iters
            logging.info(f"Benchmark completed. Average time: {avg_time:.6f}s")
            return avg_time
        except Exception as e:
            logging.error(f"Error during benchmark: {e}")
            raise

    except Exception as e:
        logging.error(f"Benchmark failed: {e}")
        return float("inf")

def get_configs_compute_bound(
    M: int,
    N: int,
    K: int,
    dtype_str: str,
    block_shape: Optional[List[int]] = None,
) -> List[BenchmarkConfig]:
    if _is_hip_:
        # ROCm configuration with expanded search space
        return [
            BenchmarkConfig(
                BLOCK_SIZE_M=m,
                BLOCK_SIZE_N=n,
                BLOCK_SIZE_K=k,
                num_warps=w,
                num_stages=2,
            )
            for m in [16, 32, 64, 128]  # Added 16 and 128
            for n in [16, 32, 64, 128]  # Added 16 and 128
            for k in [16, 32, 64, 128]  # Added 16 and 128
            for w in [1, 2, 4, 8]  # Added 1 and 8
        ]
    else:
        # CUDA configuration with expanded search space
        return [
            BenchmarkConfig(
                BLOCK_SIZE_M=m,
                BLOCK_SIZE_N=n,
                BLOCK_SIZE_K=k,
                num_warps=w,
                num_stages=s,
            )
            for s in [1, 2, 3, 4]  # Added 1 and 4
            for m in [16, 32, 64, 128]  # Added 16 and 128
            for n in [16, 32, 64, 128]  # Added 16 and 128
            for k in [16, 32, 64, 128]  # Added 16 and 128
            for w in [1, 2, 4, 8]  # Added 1 and 8
        ]

def main(args):
    # Load model config
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    hidden_size = config.hidden_size
    num_experts = getattr(config, "n_routed_experts", 8)  # Default to 8 if not found
    dtype = torch.float16

    print(f"Model configuration:")
    print(f"- Hidden size: {hidden_size}")
    print(f"- Number of experts: {num_experts}")
    print(f"- Data type: {dtype}")
    if args.block_shape:
        print(f"- Block shape: {args.block_shape}")
    print(f"- Use FP8: {args.use_fp8_w8a8}")

    # Create sample inputs
    batch_sizes = [1, 2, 4, 8, 16, 32]
    device = torch.device("cuda")

    # Initialize tensors
    a = torch.randn(batch_sizes[-1], hidden_size, device=device, dtype=dtype)
    b = torch.randn(num_experts, hidden_size, hidden_size, device=device, dtype=dtype)
    c = torch.empty(batch_sizes[-1], hidden_size, device=device, dtype=dtype)

    # Initialize indices
    seg_indptr = torch.zeros(batch_sizes[-1] + 1, device=device, dtype=torch.int64)
    weight_indices = torch.zeros(batch_sizes[-1], device=device, dtype=torch.int64)

    # Get configurations
    config_dtype_str = get_config_dtype_str(
        use_fp8_w8a8=args.use_fp8_w8a8,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        dtype=dtype,
    )

    configs = get_configs_compute_bound(
        batch_sizes[-1],
        hidden_size,
        hidden_size,
        config_dtype_str,
        args.block_shape,
    )

    print(f"\nTesting {len(configs)} configurations...")

    # Run benchmarks
    results = []
    for config in tqdm(configs, desc="Benchmarking configurations"):
        latency = benchmark_config(
            a,
            b,
            c,
            batch_sizes[-1],
            True,
            seg_indptr,
            weight_indices,
            args.use_fp8_w8a8,
            None,
            None,
            args.block_shape,
            config,
        )
        results.append((config, latency))

    # Sort results by latency
    results.sort(key=lambda x: x[1])

    # Save results
    output_file = get_config_file_name(
        hidden_size,
        hidden_size,
        hidden_size,
        config_dtype_str,
        args.block_shape,
    )
    best_config = results[0][0]
    print(f"\nBest configuration:")
    print(f"- Latency: {results[0][1]*1000:.2f}ms")
    print(f"- Config: {best_config}")
    print(f"- Saving to: {output_file}")

    with open(output_file, "w") as f:
        json.dump(best_config, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--use-fp8-w8a8", action="store_true")
    parser.add_argument("--block-shape", type=int, nargs=2, default=None)
    args = parser.parse_args()
    main(args) 