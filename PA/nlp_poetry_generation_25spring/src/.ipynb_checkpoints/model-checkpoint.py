import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- 修改开始 ---
# 将模型ID修改为本地模型文件所在的目录路径
local_model_path = "./pretrain/Qwen1.5-1.8B-Chat/"
# --- 修改结束 ---

# QLoRA 量化配置 (这部分保持不变)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 # 在支持的硬件上使用bfloat16以获得更好性能
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    local_model_path, # <--- 修改这里，使用本地路径
    quantization_config=bnb_config,
    device_map="auto" # 自动将模型分发到可用设备（GPU）
)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(local_model_path) # <--- 修改这里，使用本地路径
# Qwen1.5没有默认的pad_token，我们通常可以将其设置为eos_token
tokenizer.pad_token = tokenizer.eos_token

print(f"模型和分词器已从本地路径 '{local_model_path}' 加载。")