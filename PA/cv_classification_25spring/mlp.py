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
    def __init__(self, input_size=3*84*84, hidden_sizes=[2048, 1024, 512, 256], num_classes=10, dropout_rate=0.5):
        super(MLP, self).__init__()
        
        self.input_dropout = nn.Dropout(0.2)  # 输入层dropout
        
        # 第一层
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # 第二层
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # 第三层
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # 第四层
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.bn4 = nn.BatchNorm1d(hidden_sizes[3])
        self.dropout4 = nn.Dropout(dropout_rate)
        
        # 输出层
        self.fc_out = nn.Linear(hidden_sizes[3], num_classes)
        
        # 残差连接的投影层
        self.shortcut1 = nn.Linear(input_size, hidden_sizes[1])
        self.shortcut2 = nn.Linear(hidden_sizes[1], hidden_sizes[3])
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 将输入展平
        x = x.view(x.size(0), -1)
        identity = x
        
        # 输入dropout
        x = self.input_dropout(x)
        
        # 第一层
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # 第二层 + 残差连接1
        identity1 = self.shortcut1(identity)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x + identity1)  # 残差连接
        x = self.dropout2(x)
        identity2 = x
        
        # 第三层
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # 第四层 + 残差连接2
        identity2 = self.shortcut2(identity2)
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x + identity2)  # 残差连接
        x = self.dropout4(x)
        
        # 输出层
        x = self.fc_out(x)
        
        return x
    def __init__(self, input_size=3*84*84, hidden_sizes=[1024, 512, 256, 128], num_classes=10, dropout_rate=0.75):
        """
        多层感知机模型
        
        Args:
            input_size: 输入特征的维度
            hidden_sizes: 隐藏层的神经元数量列表1                 
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

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=50):
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
    
    Returns:
        训练历史记录
    """
    model.to(device)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    best_val_acc = 0
    
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
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
        
        # 保存最佳模型 (不使用早停)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_mlp_model.pth')
            print(f'Model saved with Val Acc: {val_acc:.2f}%')
    
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

# 修改plot_history函数，使其返回figure对象而不是直接显示
def plot_history(history, save_fig=True, show_fig=False):
    """
    绘制训练历史曲线
    
    Args:
        history: 训练历史记录
        save_fig: 是否保存图像到文件
        show_fig: 是否显示图像
    
    Returns:
        matplotlib figure对象
    """
    fig = plt.figure(figsize=(12, 5))
    
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
    
    if save_fig:
        plt.savefig('mlp_training_history.png')
    
    if show_fig:
        plt.show()
    
    return fig

# 添加一个main函数，用于在Jupyter中调用
def main(img_size=84, 
         batch_size=32, 
         data_dir='data/custom-dataset', 
         num_classes=10,
         hidden_sizes=[2048, 1024, 512, 256],
         dropout_rate=0.5,
         learning_rate=0.001,
         weight_decay=0.01,
         epochs=50,
         show_plots=False):
    """
    运行MLP实验的主函数
    
    Args:
        img_size: 输入图像大小
        batch_size: 批量大小
        data_dir: 数据集目录
        num_classes: 分类类别数
        hidden_sizes: 隐藏层大小列表
        dropout_rate: Dropout比率
        learning_rate: 学习率
        weight_decay: 权重衰减
        epochs: 训练轮数
        show_plots: 是否显示图形
    
    Returns:
        包含训练历史、测试结果和图形的字典
    """
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        img_size=img_size
    )
    
    # 创建模型
    input_size = 3 * img_size * img_size
    model = MLP(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 定义优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # 定义学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )
    
    # 训练模型
    print("开始训练MLP模型...")
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=epochs
    )
    
    # 绘制训练历史
    fig = plot_history(history, save_fig=True, show_fig=show_plots)
    
    # 加载最佳模型并评估
    print("加载最佳模型并进行评估...")
    model.load_state_dict(torch.load('best_mlp_model.pth'))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    # 返回结果
    results = {
        'history': history,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'fig': fig
    }
    
    return results

# 修改if __name__ == '__main__'部分
if __name__ == '__main__':
    results = main()

def get_data_loaders(data_dir='data/custom-dataset', batch_size=32, img_size=84):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5)  # 随机擦除增强
    ])
    """
    获取数据加载器
    
    Args:
        data_dir: 数据集目录
        batch_size: 批量大小
        img_size: 图像大小
    
    Returns:
        训练,验证和测试数据加载器
    """
    # 数据预处理和增强 (减少增强强度)
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.3),  # 降低翻转概率
        transforms.RandomRotation(5),  # 减小旋转角度
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # 减小颜色变化
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
    
    # 创建数据加载器 (统一worker数量)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, test_loader