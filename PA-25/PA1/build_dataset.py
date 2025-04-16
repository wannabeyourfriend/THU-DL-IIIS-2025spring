import os
import shutil
import random
from tqdm import tqdm
import numpy as np

def build_dataset(source_dir, target_dir, selected_classes, split_ratio=(0.6, 0.2, 0.2), seed=42):
    """
    从mini-imagenet数据集中构建训练、验证和测试数据集
    
    Args:
        source_dir: 源数据目录
        target_dir: 目标数据目录
        selected_classes: 选定的类别列表
        split_ratio: 训练、验证、测试集的比例
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # 创建目标目录结构
    os.makedirs(target_dir, exist_ok=True)
    
    # 创建训练、验证和测试集目录
    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'val')
    test_dir = os.path.join(target_dir, 'test')
    
    for directory in [train_dir, val_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # 处理每个选定的类别
    for class_id in tqdm(selected_classes, desc="处理类别"):
        # 获取类别名称
        class_name = class_id.split(':')[0].strip()
        
        # 创建类别目录
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        
        for directory in [train_class_dir, val_class_dir, test_class_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # 获取该类别的所有图像
        source_class_dir = os.path.join(source_dir, class_name)
        images = [f for f in os.listdir(source_class_dir) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]
        
        # 随机打乱图像顺序
        random.shuffle(images)
        
        # 计算分割点
        n_images = len(images)
        n_train = int(n_images * split_ratio[0])
        n_val = int(n_images * split_ratio[1])
        
        # 分割数据集
        train_images = images[:n_train]
        val_images = images[n_train:n_train+n_val]
        test_images = images[n_train+n_val:]
        
        # 复制图像到目标目录
        for img in train_images:
            shutil.copy2(os.path.join(source_class_dir, img), os.path.join(train_class_dir, img))
        
        for img in val_images:
            shutil.copy2(os.path.join(source_class_dir, img), os.path.join(val_class_dir, img))
        
        for img in test_images:
            shutil.copy2(os.path.join(source_class_dir, img), os.path.join(test_class_dir, img))
    
    # 统计数据集信息
    train_count = sum([len(os.listdir(os.path.join(train_dir, d))) for d in os.listdir(train_dir)])
    val_count = sum([len(os.listdir(os.path.join(val_dir, d))) for d in os.listdir(val_dir)])
    test_count = sum([len(os.listdir(os.path.join(test_dir, d))) for d in os.listdir(test_dir)])
    
    print(f"数据集构建完成！")
    print(f"训练集: {train_count}张图像")
    print(f"验证集: {val_count}张图像")
    print(f"测试集: {test_count}张图像")
    
    return train_dir, val_dir, test_dir

if __name__ == "__main__":
    # 源数据目录
    source_dir = os.path.join('data', 'mini-imagenet')
    
    # 目标数据目录
    target_dir = os.path.join('data', 'custom-dataset')
    
    # 选定的类别（细粒度和粗粒度分类）
    selected_classes = [
        # 细粒度分类类别(相似度高)
        "n02110341: dalmatian",
        "n02971356: carton",
        "n03127925: crate",
        "n03908618: pencil_box",
        "n04509417: unicycle",
        # 粗粒度分类类别(相似度低)
        "n02606052: rock_beauty",
        "n07613480: trifle",
        "n07697537: hotdog",
        "n01704323: triceratops",
        "n01749939: green_mamba"
    ]
    
    # 构建数据集，比例为4:1:1
    train_dir, val_dir, test_dir = build_dataset(
        source_dir, 
        target_dir, 
        selected_classes, 
        split_ratio=(0.6, 0.2, 0.2)
    )
    
    # 创建类别映射文件
    class_mapping = {i: cls.split(':')[0].strip() for i, cls in enumerate(selected_classes)}
    
    with open(os.path.join(target_dir, 'class_mapping.txt'), 'w') as f:
        for idx, class_id in class_mapping.items():
            f.write(f"{idx}: {class_id}\n")
    
    print("类别映射已保存到 class_mapping.txt")