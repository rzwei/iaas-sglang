import os
import subprocess
from pathlib import Path
import json
import argparse
from datetime import datetime
import logging

# 设置日志记录
def setup_logging():
    """设置日志记录"""
    log_dir = Path("tuning_logs")
    log_dir.mkdir(exist_ok=True)
    
    # 设置日志格式
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )

def get_model_list():
    """获取/data00/models/下的模型列表"""
    model_dir = Path("/data00/models")
    models = []
    
    logging.info(f"Scanning for MoE models in {model_dir}")
    
    # 遍历目录下的所有模型
    for model_path in model_dir.iterdir():
        if model_path.is_dir():
            # 检查是否有config.json文件
            config_path = model_path / "config.json"
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                        logging.debug(f"Reading config from {config_path}")
                        # 检查是否是MoE模型
                        if any(arch in config.get("architectures", []) for arch in [
                            "MixtralForCausalLM",
                            "DbrxForCausalLM",
                            "JambaForCausalLM",
                            "Qwen2MoeForCausalLM",
                            "DeepseekV2ForCausalLM",
                            "DeepseekV3ForCausalLM",
                            "Grok1ForCausalLM",
                            "Grok1ImgGen",
                            "Grok1AForCausalLM"
                        ]):
                            models.append(str(model_path))
                            logging.info(f"Found MoE model: {model_path}")
                            logging.debug(f"Model config: {json.dumps(config, indent=2)}")
                except Exception as e:
                    logging.error(f"Error reading config for {model_path}: {e}")
    
    return models

def run_tuning(model_path: str, block_shape: tuple = None, use_fp8_w8a8: bool = False):
    """对单个模型运行调优"""
    logging.info(f"{'='*50}")
    logging.info(f"Starting tuning for model: {model_path}")
    logging.info(f"Block shape: {block_shape}")
    logging.info(f"Use FP8: {use_fp8_w8a8}")
    logging.info(f"{'='*50}")
    
    # 读取模型配置
    try:
        with open(Path(model_path) / "config.json") as f:
            config = json.load(f)
            logging.info("Model configuration:")
            logging.info(f"- Architecture: {config.get('architectures', ['Unknown'])[0]}")
            logging.info(f"- Hidden size: {config.get('hidden_size', 'Unknown')}")
            logging.info(f"- Number of experts: {config.get('num_local_experts', config.get('num_experts', 'Unknown'))}")
            logging.info(f"- Model type: {config.get('model_type', 'Unknown')}")
    except Exception as e:
        logging.error(f"Error reading model config: {e}")
    
    # 创建日志目录
    log_dir = Path("tuning_logs")
    log_dir.mkdir(exist_ok=True)
    
    # 生成日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(model_path).name
    log_file = log_dir / f"{model_name}_{timestamp}.log"
    
    # 构建调优命令
    cmd = [
        "python3", "tuning_grouped_gemm_triton.py",
        "--model-path", model_path,
    ]
    
    if block_shape:
        cmd.extend(["--block-shape", str(block_shape[0]), str(block_shape[1])])
    
    if use_fp8_w8a8:
        cmd.append("--use-fp8-w8a8")
    
    logging.info(f"Running command: {' '.join(cmd)}")
    
    # 运行调优命令并记录日志
    try:
        with open(log_file, "w") as f:
            f.write(f"Starting tuning for {model_path} at {timestamp}\n")
            f.write(f"Command: {' '.join(cmd)}\n\n")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 实时写入日志
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    output = output.strip()
                    print(output)
                    f.write(output + "\n")
                    logging.debug(f"Process output: {output}")
            
            # 获取错误输出
            error = process.stderr.read()
            if error:
                logging.error(f"Process error output:\n{error}")
                f.write(f"\nError output:\n{error}")
            
            # 获取返回码
            return_code = process.poll()
            if return_code == 0:
                logging.info(f"Tuning completed successfully for {model_path}")
                f.write(f"\nTuning completed successfully at {datetime.now().strftime('%Y%m%d_%H%M%S')}")
            else:
                logging.error(f"Tuning failed for {model_path} with return code {return_code}")
                f.write(f"\nTuning failed with return code {return_code}")
                
    except Exception as e:
        error_msg = f"Error during tuning: {str(e)}"
        logging.exception(error_msg)
        with open(log_file, "a") as f:
            f.write(f"\n{error_msg}")

def main():
    # 设置日志记录
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Tune MoE models")
    parser.add_argument("--block-shape", type=int, nargs=2, help="Block shape for FP8 quantization (e.g., 32 64)")
    parser.add_argument("--use-fp8-w8a8", action="store_true", help="Use FP8 quantization")
    args = parser.parse_args()
    
    logging.info("Starting tuning script")
    logging.info(f"Arguments: {args}")
    
    # 获取模型列表
    models = get_model_list()
    
    if not models:
        logging.warning("No MoE models found in /data00/models/")
        return
    
    logging.info(f"Found {len(models)} MoE models:")
    for model in models:
        logging.info(f"- {model}")
    
    # 对每个模型进行调优
    for model in models:
        try:
            run_tuning(model, args.block_shape, args.use_fp8_w8a8)
        except Exception as e:
            logging.exception(f"Error while tuning {model}")

if __name__ == "__main__":
    main() 