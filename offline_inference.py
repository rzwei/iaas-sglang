# launch the offline engine
#from sglang.utils import stream_and_merge, async_stream_and_merge
import sglang as sgl
import asyncio
import torch._dynamo
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

#export TORCH_LOGS="output_code"
#os.environ["TORCH_LOGS"] = 'output_code'

#os.environ["VLLM_USE_TRITON_NON_ATTN"] = 'false' # NEW
#os.environ["VLLM_USE_TRITON_NON_ATTN"] = 'true' # NEW
os.environ["VLLM_USE_TRITON_NON_ATTN"] = '1' # NEW

#os.environ["VLLM_USE_V1"] = '1'

#os.environ['VLLM_ATTENTION_BACKEND'] = 'TRITON_MHA'
# TRITON_MLA/FLASHMLA 似乎都是MLA專用 所以跟model有關 model本跟要造成use_mla(1) 才會正常啟動 所以不管V0/V1 用llama跑都掛
#os.environ['VLLM_ATTENTION_BACKEND'] = 'TRITON_MLA' #
#os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHMLA' #

# tried nothing worked
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'spawn'
#export PYTORCH_CUDA_ALLOC_CONF=spawn

# os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "True"

# Set CUDA_LAUNCH_BLOCKING to 1 for debugging
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Optionally, enable device-side assertions if needed
#os.environ['TORCH_USE_CUDA_DSA'] = '1'

# python3 -m sglang.launch_server --model-path /data00/jack/models/Llama-2-7b-hf --tp 8 --trust-remote-code --port 8010 --mem-fraction-static 0.95 --disable-cuda-graph --disable-radix-cache
def main():
    #llm = sgl.Engine(model_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
    #llm = sgl.Engine(model_path="/data00/jack/models/models/Meta-Llama-3.1-8B-Instruct",
    #llm = sgl.Engine(model_path="/data00/jack/models/models/Meta-Llama-3-8B",
    #llm = sgl.Engine(model_path="/data00/jack/models/Meta-Llama-3.1-8B-Instruct",
    #llm = sgl.Engine(model_path="/data00/jack/models/Meta-Llama-3-8B-Instruct",
    #llm = sgl.Engine(model_path="/data00/jack/models/Meta-Llama-3-8B",
    #                 context_length=8192, mem_fraction_static=0.7, enable_torch_compile=False, disable_cuda_graph=False)
    #llm = sgl.Engine(model_path="/data00/jack/models/Llama-2-7b-hf",
    #                 enable_torch_compile=False, disable_cuda_graph=False)
    #llm = sgl.Engine(model_path="/data00/jack/models/Llama-2-7b-hf",
    #                 enable_torch_compile=False, disable_cuda_graph=True)
                     #use_triton_primative_kernels=True, enable_torch_compile=False, disable_cuda_graph=True)
    #llm = sgl.Engine(model_path="/data00/jack/models/Meta-Llama-3-8B",
    llm = sgl.Engine(model_path="/data00/jack/models/Meta-Llama-3-8B-Instruct",
                     enable_torch_compile=False, disable_cuda_graph=True)
                     #enable_torch_compile=False, disable_cuda_graph=False)
                     #enable_torch_compile=True, disable_cuda_graph=True)
                     #enable_torch_compile=True, disable_cuda_graph=False)

    # Can use graph mode to print out all the functions


    prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    ]

    sampling_params = {"temperature": 0.8, "top_p": 0.95}
    # Deterministic inference resutls
    #sampling_params = {
    #    "temperature": 0.0,  # 0 表示不使用隨機性
    #    "top_p": 0.95,        # 1.0 表示不過濾候選詞
    #    #"top_p": 1.0,        # 1.0 表示不過濾候選詞
    #    #"top_k": 1,          # 只選擇最高分數的 token
    #}


    outputs = llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")

# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    main()
