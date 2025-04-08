# 实现MLP模型
# 实现MLP模型
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class MLP(nn.Module):
    def __init__(self, input_size=3*224*224, hidden_sizes=[1024, 512, 256], num_classes=10, dropout_rate=0.5):
        """
        多层感知机模型
        
        Args:
            input_size: 输入特征的维度
            hidden_sizes: 隐藏层的神经元数量列表
            num_classes: 分类类别数
            dropout_rate: Dropout比率,用于防止过拟合
        """
        super(MLP, self).__init__()
        
        # 构建网络层
        layers = []
        
        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # 构建中间隐藏层
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # 最后一个隐藏层到输出层
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        
        # 将所有层组合成一个Sequential模型
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """前向传播"""
        # 将输入展平
        x = x.view(x.size(0), -1)
        return self.model(x)

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=30, early_stopping_patience=5):
    """
    训练MLP模型
    
    Args:
        model: 模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 训练设备
        epochs: 训练轮数
        early_stopping_patience: 早停耐心值
    
    Returns:
        训练历史记录
    """
    model.to(device)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            pbar.set_postfix({'loss': train_loss/train_total, 'acc': 100.*train_correct/train_total})
        
        train_loss = train_loss / train_total
        train_acc = 100. * train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 统计
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                # 更新进度条
                pbar.set_postfix({'loss': val_loss/val_total, 'acc': 100.*val_correct/val_total})
        
        val_loss = val_loss / val_total
        val_acc = 100. * val_correct / val_total
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_mlp_model.pth')
            print(f'Model saved with Val Acc: {val_acc:.2f}%')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    return history

def evaluate(model, test_loader, criterion, device):
    """
    评估模型性能
    
    Args:
        model: 模型实例
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 评估设备
    
    Returns:
        测试损失和准确率
    """
    model.to(device)
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    
    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 统计总体准确率
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            
            # 统计每个类别的准确率
            correct = predicted.eq(targets).cpu().numpy()
            for i in range(inputs.size(0)):
                label = targets[i].item()
                class_correct[label] += correct[i]
                class_total[label] += 1
    
    test_loss = test_loss / test_total
    test_acc = 100. * test_correct / test_total
    
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    # 打印每个类别的准确率
    for i in range(10):
        print(f'Accuracy of class {i}: {100 * class_correct[i] / class_total[i]:.2f}%')
    
    return test_loss, test_acc

def plot_history(history):
    """
    绘制训练历史曲线
    
    Args:
        history: 训练历史记录
    """
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    plt.savefig('mlp_training_history.png')
    plt.show()

def get_data_loaders(data_dir='data/custom-dataset', batch_size=32, img_size=224):
    """
    获取数据加载器
    
    Args:
        data_dir: 数据集目录
        batch_size: 批量大小
        img_size: 图像大小
    
    Returns:
        训练,验证和测试数据加载器
    """
    # 数据预处理和增强
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    train_dataset = datasets.ImageFolder(root=f'{data_dir}/train', transform=train_transform)
    val_dataset = datasets.ImageFolder(root=f'{data_dir}/val', transform=val_test_transform)
    test_dataset = datasets.ImageFolder(root=f'{data_dir}/test', transform=val_test_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_data_loaders()
    
    # 创建模型
    input_size = 3 * 224 * 224  # 假设输入图像大小为224x224
    hidden_sizes = [2048, 1024, 512]
    num_classes = 10
    model = MLP(input_size, hidden_sizes, num_classes, dropout_rate=0.5)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    # 训练模型
    history = train(model, train_loader, val_loader, criterion, optimizer, scheduler, device)
    
    # 绘制训练历史
    plot_history(history)
    
    # 加载最佳模型并评估
    model.load_state_dict(torch.load('best_mlp_model.pth'))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)