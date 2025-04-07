这个题目的关键在于理解`d2l`中默认的`trainer`函数的方法：

在老师给出的代码实现中， SoftmaxRegressionScratch 类继承自 Classifier ，且没有显式定义 loss 方法，模型会使用父类的默认损失函数实现，，可以通过如下代码得到：

```python
import inspect
print(inspect.getsource(d2l.Classifier.loss))
```

输出为

```python
def loss(self, Y_hat, Y, averaged=True):
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    return F.cross_entropy(Y_hat, Y, reduction='mean' if averaged else 'none')
```

父类的损失函数期望的是原始的logits（未经过softmax处理的值），因为 F.cross_entropy 内部会自动应用softmax.

而由于我们forward 方法输出的是经过自定义 softmax 函数处理的概率值

```python
def forward(self, X):
    X = X.reshape((-1, self.W.shape[0]))
    return self.softmax(torch.matmul(X, self.W) + self.b)
```

这会导致：

1. 数值不稳定性：对已经是概率的值再次应用softmax
2. 梯度计算错误：双重softmax会影响梯度传播
3. 训练效果不佳：损失值计算不准确









用数学表达式来解释这个双重softmax的问题：

1. 正常的softmax回归过程：
   - 线性变换：$o = Wx + b$ (logits)
   - softmax函数：$p_i = \frac{e^{o_i}}{\sum_j e^{o_j}}$
   - 交叉熵损失：$L = -\sum_i y_i \log(p_i)$

2. 双重softmax问题：
   - 第一次softmax（在forward中）：$p_i = \frac{e^{o_i}}{\sum_j e^{o_j}}$
   - 第二次softmax（在损失函数中）：$q_i = \frac{e^{p_i}}{\sum_j e^{p_j}}$
   - 最终损失：$L = -\sum_i y_i \log(q_i)$

这会导致以下问题：
1. 数值不稳定：$e^{e^x}$ 容易导致数值溢出
2. 梯度链式法则：$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial q} \frac{\partial q}{\partial p} \frac{\partial p}{\partial o} \frac{\partial o}{\partial W}$
   - 多了一个不必要的$\frac{\partial q}{\partial p}$项
   - 使梯度计算变得不准确且不稳定

这就是为什么我们应该只应用一次softmax的原因。

我们可以做实验来证明我们的说法：

```python
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from d2l import torch as d2l

class FashionMNIST(d2l.DataModule):
    """Fashion-MNIST数据集类"""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=True)

    def text_labels(self, indices):
        """返回文本标签"""
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [labels[int(i)] for i in indices]

    def get_dataloader(self, train):
        """获取数据加载器"""
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train)

    def train_dataloader(self):
        """获取训练数据加载器"""
        return self.get_dataloader(train=True)
    
    def val_dataloader(self):
        """获取验证数据加载器"""
        return self.get_dataloader(train=False)

class Classifier(d2l.Module):  #@save
    """The base class of classification models."""
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)
        
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)
    
    def accuracy(self, Y_hat, Y, averaged=True):
        """Compute the number of correct predictions."""
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        preds = Y_hat.argmax(axis=1).type(Y.dtype)
        compare = (preds == Y.reshape(-1)).type(torch.float32)
        return compare.mean() if averaged else compare


class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs), requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)

    # 正确的实现方式
    # def loss(self, Y_hat, Y):  
    #     return cross_entropy(Y_hat, Y)
    
    # 模拟错误情形
    def loss(self, Y_hat, Y):
        # 第二次softmax (模拟错误情况)
        second_softmax = self.softmax(Y_hat)
        # 计算交叉熵损失
        return -torch.log(second_softmax[range(len(Y_hat)), Y]).mean()
    
    def training_step(self, batch):  # 需要添加
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        self.plot('acc', self.accuracy(self(*batch[:-1]), batch[-1]), train=True)
        return l
    
    def parameters(self):
        return [self.W, self.b]
    
    def forward(self, X):
        X = X.reshape((-1, self.W.shape[0]))
        return self.softmax(torch.matmul(X, self.W) + self.b)
    
    def softmax(self, X):
        X_exp = torch.exp(X)
        partition = X_exp.sum(1, keepdims=True)
        return X_exp / partition
```

结果

| ---                                                          | ---                                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20250402193341404](C:\Users\35551\Desktop\assets\image-20250402193341404.png) | ![image-20250402193413237](C:\Users\35551\Desktop\assets\image-20250402193413237.png) |

