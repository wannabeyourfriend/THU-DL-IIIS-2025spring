from peft import LoraConfig, get_peft_model


lora_config = LoraConfig(
    r=16,  # LoRA的秩，一个关键超参数
    lora_alpha=32, # LoRA的alpha，通常是r的两倍
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ], # 指定要应用LoRA的模块
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 将LoRA适配器应用到模型上
model = get_peft_model(model, lora_config)

# 打印可训练参数，验证LoRA是否生效
model.print_trainable_parameters()
# 输出会显示可训练参数远小于总参数，例如:
# trainable params: 10,502,656 || all params: 1,847,199,744 || trainable%: 0.5685

from transformers import TrainingArguments
from trl import SFTTrainer

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results_qwen1.5_poetry",
    num_train_epochs=1,                # 训练轮次，对于SFT，1-3轮通常足够
    per_device_train_batch_size=4,     # 每个设备的批大小
    gradient_accumulation_steps=4,     # 梯度累积步数，有效批大小 = batch_size * accumulation_steps
    optim="paged_adamw_8bit",          # 使用8-bit AdamW优化器以节省内存
    logging_steps=20,                  # 每20步记录一次日志
    learning_rate=2e-4,                # 学习率
    lr_scheduler_type="cosine",        # 学习率调度器
    save_strategy="epoch",             # 每个epoch保存一次模型
    warmup_ratio=0.03,                 # 预热比例
    bf16=True,                         # 如果GPU支持，使用bfloat16混合精度训练
    report_to="wandb",                 # 将报告发送到wandb
    run_name="qwen1.5-1.8b-poetry-sft-run-1" # wandb上的运行名称
)

# 创建SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset,
    peft_config=lora_config,
    dataset_text_field="messages",     # 指定数据集中包含对话消息的字段
    max_seq_length=1024,               # 序列最大长度
    args=training_args,
    packing=True,                      # 将多个短序列打包成一个长序列，提高训练效率
)

print("开始微调...")
trainer.train()
print("微调完成！")

# 保存最终的模型适配器
trainer.save_model("./pretrain/final_poetry_adapter")
print("模型适配器已保存到 ./pretrain/final_poetry_adapter")
