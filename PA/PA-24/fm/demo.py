import os
import torch
import argparse
import datetime
from pathlib import Path
from gpt import GPT
from configs import get_configs
import tiktoken

def setup_args():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(description="Interactive demo for GPT model")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="checkpoints/sft", 
        help="Path to the checkpoint directory"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="demo", 
        help="Directory to save interaction results"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=100, 
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7, 
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.9, 
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--model_type", 
        type=str, 
        default="gpt2", 
        help="Model type (gpt2, gpt2/dropout)"
    )
    return parser.parse_args()

def load_model(checkpoint_path, model_type="gpt2"):
    """Load model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    cfg = get_configs(model_type)
    model = GPT.from_pretrained(cfg, checkpoint_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"Model loaded successfully and moved to {device}")
    return model, device

def prepare_gpt2_input(prompt, device):
    enc = tiktoken.get_encoding("gpt2")
    # 修改这里：使用正确的特殊标记，并处理编码问题
    encode = lambda s: enc.encode(s, allowed_special={""})  
    decode = lambda l: enc.decode(l)
    indices = encode(prompt)
    x = (torch.tensor(indices, dtype=torch.long, device=device)[None, ...])
    return x, decode

def generate_response(model, prompt, device, max_new_tokens=100, temperature=0.7, top_p=0.9):
    x, decode = prepare_gpt2_input(prompt, device)
    
    with torch.no_grad():
        y = model.generate(
            x,
            max_new_tokens,
            temperature=temperature,
            top_k=int(top_p * 100)  
        )
    
    # 解码并返回生成的文本
    return decode(y[0].tolist())

def save_interaction(output_dir, prompt, response, session_id):
    """保存交互结果到文件"""
    # 如果输出目录不存在，则创建
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用会话 ID 创建文件路径
    file_path = os.path.join(output_dir, f"interaction_{session_id}.txt")
    
    # 修改这里：处理编码问题，确保写入文件时使用正确的编码
    try:
        # 尝试处理可能包含代理对的字符串
        prompt = prompt.encode('utf-8', errors='ignore').decode('utf-8')
        response = response.encode('utf-8', errors='ignore').decode('utf-8')
        
        # 将交互写入文件
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"Prompt: {prompt}\n\n")
            f.write(f"Response: {response}\n\n")
            f.write("-" * 50 + "\n\n")
    except Exception as e:
        print(f"Error saving interaction: {e}")
        # 创建一个简单的备份记录
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"Interaction occurred but could not be fully saved due to encoding issues.\n")
            f.write("-" * 50 + "\n\n")
    
    return file_path

def main():
    args = setup_args()
    
    checkpoint_dir = Path(args.checkpoint)
    checkpoint_files = list(checkpoint_dir.glob("*.pt"))
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return
    
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    print(f"Using latest checkpoint: {latest_checkpoint}")
    
    model, device = load_model(latest_checkpoint, args.model_type)
    
    session_id = datetime.datetime.now().strftime("%B_%d_%Y_%H_%M_%S")
    print(f"Session ID: {session_id}")
    
    output_dir = os.path.join(args.output_dir, session_id)
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 50)
    print("Interactive GPT Demo")
    print("Type 'exit' or 'quit' to end the session")
    print("=" * 50 + "\n")
    
    while True:
        prompt = input("\nYou: ")
        
        if prompt.lower() in ["exit", "quit"]:
            print("Ending session. Goodbye!")
            break
        
        response = generate_response(
            model, 
            prompt, 
            device, 
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p 
        )
        
        print(f"\nGPT: {response}")
        
        file_path = save_interaction(output_dir, prompt, response, session_id)
        print(f"Interaction saved to {file_path}")

if __name__ == "__main__":
    main()