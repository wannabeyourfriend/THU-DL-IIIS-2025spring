import wandb

# 登录到你的wandb账户
# 你需要从 https://wandb.ai/authorize 获取你的API密钥
try:
    wandb.login()
except Exception as e:
    print(f"Wandb login failed: {e}. Please run 'wandb login' in your terminal.")

# 设置你的W&B项目名称
import os
os.environ["WANDB_PROJECT"] = "qwen1.5-tang-poetry-sft"