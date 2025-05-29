from datasets import load_dataset
import re
import os

local_dataset_dir = "./data/tang_poems/" # 假设脚本从 PA2 目录运行

dataset_files = {
    'test': os.path.join(local_dataset_dir, 'test-00000-of-00001-a794cd4c018c9326.parquet'),
    'train': os.path.join(local_dataset_dir, 'train-00000-of-00001-6914ee5fabc145c0.parquet')
}

all_splits = load_dataset('parquet', data_files=dataset_files)

train_dataset = all_splits['train']
test_dataset = all_splits['test']


def format_poetry_prompt(sample):
    """
    处理单条数据，将其格式化为模型输入格式。
    无效数据将返回 {"messages": None}。
    """
    poetry_column_name = 'paragraphs'
    
    # 修改点1：无效时返回 {"messages": None}
    if poetry_column_name not in sample or sample[poetry_column_name] is None:
        return {"messages": None} 
        
    raw_poem_data = sample[poetry_column_name]
    
    if isinstance(raw_poem_data, list):
        poem_text = "".join(raw_poem_data) 
    elif isinstance(raw_poem_data, str):
        poem_text = raw_poem_data
    else:
        return {"messages": None} # 修改点2

    poem_text = poem_text.strip()
    if not poem_text:
        return {"messages": None} # 修改点3

    parts = re.split(r'([，。？！])', poem_text)
    if len(parts) < 3:
        return {"messages": None} # 修改点4

    prompt_starter = parts[0] + parts[1]
    completion = "".join(parts[2:]).strip()
    
    if not prompt_starter or not completion:
        return {"messages": None} # 修改点5

    # 成功处理的情况
    messages_content = [
        {"role": "user", "content": f"请补全这首唐诗：{prompt_starter}"},
        {"role": "assistant", "content": completion}
    ]
    return {"messages": messages_content} # 保持不变

# 使用map函数处理训练数据集
formatted_train_dataset = train_dataset.map(
    format_poetry_prompt, 
    remove_columns=train_dataset.column_names 
)

# 修改点6：Filter现在主要依赖于 "messages" 键的值是否为 None
# x is not None 将始终为True，因为format_poetry_prompt不再返回顶层None
formatted_train_dataset = formatted_train_dataset.filter(
    lambda x: x.get("messages") is not None
)


if len(formatted_train_dataset) > 0:
    print("数据格式示例 (训练集):")
    print(formatted_train_dataset[0]['messages'])
    print(f"\n处理后的训练集样本数量: {len(formatted_train_dataset)}")
else:
    print("警告：处理后的训练数据集为空，请检查 format_poetry_prompt 函数、原始数据或 'paragraphs' 列的内容。")