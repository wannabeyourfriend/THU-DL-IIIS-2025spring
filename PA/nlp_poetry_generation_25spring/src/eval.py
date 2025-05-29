
from transformers import pipeline
import torch

# 加载我们刚刚训练好的模型适配器进行推理
# 在SFTTrainer训练后，模型已在内存中准备好，可以直接使用
# 如果要从头加载，请参考下面的注释代码

# --- 如果在新的脚本中加载模型 ---
# from peft import AutoPeftModelForCausalLM
# trained_model = AutoPeftModelForCausalLM.from_pretrained(
#     "./final_poetry_adapter",
#     low_cpu_mem_usage=True,
#     return_dict=True,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )
# trained_tokenizer = AutoTokenizer.from_pretrained("./final_poetry_adapter")
# ------------------------------------

# 直接使用trainer中的模型和分词器
trained_model = trainer.model
trained_tokenizer = trainer.tokenizer

# 定义几个测试开头
prompts = [
    "白日依山尽，",
    "红豆生南国，",
    "床前明月光，", # 一个非常经典的例子
    "双燕东南飞",
]

for start_text in prompts:
    # 构造与训练时一致的输入格式
    messages = [
        {"role": "user", "content": f"请补全这首唐诗：{start_text}"}
    ]
    
    # 使用pipeline进行生成
    pipe = pipeline("text-generation", model=trained_model, tokenizer=trained_tokenizer)
    
    # apply_chat_template 会将消息列表转换为模型能理解的单个字符串
    prompt_for_model = trained_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print(f"--- 测试开头: {start_text} ---")
    
    outputs = pipe(
        prompt_for_model,
        max_new_tokens=60, # 生成的最大token数
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        eos_token_id=trained_tokenizer.eos_token_id,
        pad_token_id=trained_tokenizer.pad_token_id
    )
    
    generated_text = outputs[0]['generated_text']
    # 从生成结果中提取助手的回复
    assistant_response = generated_text.split("<|im_start|>assistant")[1].replace("<|im_end|>", "").strip()
    
    print(f"模型生成结果:\n{start_text}{assistant_response}\n")

