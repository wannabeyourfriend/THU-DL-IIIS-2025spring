# Foundation Model 
## Overview
这是一个基于GPT-2架构的语言模型实现项目，主要包含了模型定义、训练流程和评估方法。项目实现了两种训练方式：
1. SFT (Supervised Fine-Tuning) - 监督微调
2. DPO (Direct Preference Optimization) - 直接偏好优化

## 文件结构

项目主要包含以下文件：

- `gpt.py`: GPT模型的核心实现，包括Transformer架构
- `attention.py`: 注意力机制的实现
- `configs.py`: 配置类和配置获取函数
- `dataset.py`: 数据集类的实现
- `loss.py`: 损失函数的实现
- `trainers.py`: 训练器类的实现
- `train_sft.py`: SFT训练的入口脚本
- `train_dpo.py`: DPO训练的入口脚本
- `test.py`: 模型测试脚本
- `tokenizer.py`: 分词器实现
- `requirements.txt`: 项目依赖

## 核心组件分析

### 1. 模型架构 (gpt.py)

GPT模型采用标准的Transformer解码器架构，主要组件包括：

- `MaskedMultiheadSelfAttention`: 实现带掩码的多头自注意力
- `FeedForwardNetworks`: 前馈神经网络
- `TransformerDecoderBlock`: Transformer解码器块
- `TransformerDecoder`: 完整的Transformer解码器
- `GPT`: 最终的GPT模型，包含了解码器和语言模型头

模型支持从预训练权重加载，并提供了生成文本的功能。

### 2. 注意力机制 (attention.py)

`multi_head_self_attention`函数实现了多头自注意力机制，但目前是一个待实现的TODO项。

### 3. 配置系统 (configs.py)

使用`TrainingConfig`数据类来管理模型和训练的配置参数，包括：
- 模型结构参数（层数、头数、嵌入维度等）
- 训练参数（学习率、批量大小等）
- SFT和DPO特定参数

`get_configs`函数提供了预定义的配置，如"gpt2"和"gpt2/dropout"。

### 4. 数据集 (dataset.py)

实现了两种数据集：
- `EYLSFTStaticDataset`: 用于SFT训练的数据集
- `DahoasRMStaticDataset`: 用于DPO训练的数据集，包含偏好数据（正例和负例）

### 5. 损失函数 (loss.py)

- `CrossEntropyLoss`: 用于SFT训练的交叉熵损失
- `DPOLoss`: 用于DPO训练的损失函数，目前是一个待实现的TODO项

### 6. 训练器 (trainers.py)

- `Trainer`: 基础训练器类，提供通用功能
- `SFTTrainer`: SFT训练器，实现了监督微调训练流程
- `DPOTrainer`: DPO训练器，实现了直接偏好优化训练流程，其中`shared_step`方法是待实现的TODO项

### 7. 训练脚本

- `train_sft.py`: SFT训练入口，设置配置、加载模型和数据集，并启动训练
- `train_dpo.py`: DPO训练入口，类似于SFT但针对偏好优化

### 8. 测试脚本 (test.py)

提供了模型测试功能，可以生成文本并评估模型性能。

## 待实现部分

项目中有几个关键的TODO项需要实现：

1. `attention.py`中的`multi_head_self_attention`函数
2. `gpt.py`中的掩码构建（mask construction）
3. `loss.py`中的`DPOLoss`前向传播实现
4. `trainers.py`中的`DPOTrainer.shared_step`方法

## 训练流程

### SFT训练流程

1. 加载预训练的GPT-2模型
2. 使用`EYLSFTStaticDataset`准备训练和测试数据
3. 使用`SFTTrainer`进行训练，应用交叉熵损失
4. 定期保存模型检查点

### DPO训练流程

1. 加载SFT训练后的模型
2. 使用`DahoasRMStaticDataset`准备包含偏好对的训练和测试数据
3. 使用`DPOTrainer`进行训练，应用DPO损失
4. 定期评估和保存模型

## 实现思路

要完成项目，需要实现以下关键部分：

1. 在`attention.py`中实现多头自注意力机制，包括注意力计算、掩码应用和dropout
2. 在`gpt.py`中正确构建因果掩码（causal mask）
3. 在`loss.py`中实现DPO损失函数，考虑KL散度正则化
4. 在`trainers.py`中实现DPO训练的单步逻辑，处理正例和负例

这些实现将使模型能够正确学习语言建模任务，并通过人类偏好进行优化。

## 总结

该项目是一个完整的语言模型训练框架，实现了从模型定义到训练和评估的全流程。通过SFT和DPO两种训练方法，可以先让模型学习基本的语言生成能力，然后根据人类偏好进一步优化模型输出质量。完成TODO项后，该框架将能够有效训练出高质量的语言模型。


## 脚本功能说明

1. **加载预训练模型**：
   - 自动查找checkpoint文件夹中最新的模型权重文件
   - 支持通过命令行参数指定不同的checkpoint路径
   - 支持选择不同的模型类型（如gpt2或gpt2/dropout）

2. **交互式推理**：
   - 提供命令行交互界面，用户输入提示词，模型生成回复
   - 支持配置生成参数：最大生成token数、温度、top-p采样等
   - 支持退出命令（exit或quit）

3. **结果存储**：
   - 使用英文日期时间格式创建唯一的会话ID（如"January_01_2023_12_30_45"）
   - 在demo文件夹下创建以会话ID命名的子文件夹
   - 将每次交互的提示和回复保存到文本文件中

4. **其他功能**：
   - 自动检测并使用GPU（如果可用）
   - 提供丰富的命令行参数，方便调整模型行为
   - 友好的用户界面，显示会话进度和保存位置
## 创建虚拟环境
```bash
conda env create -f environment.yaml --prefix ./fm_env
```

## 使用方法

1. 基本使用：
```bash
python demo.py
```

2. 指定不同的checkpoint路径：
```bash
python demo.py --checkpoint path/to/your/checkpoint
```

3. 调整生成参数：
```bash
python demo.py --max_new_tokens 200 --temperature 0.9 --top_p 0.95
```

4. 指定输出目录：
```bash
python demo.py --output_dir custom_demo_folder
```

5. 选择不同的模型类型：
```bash
python demo.py --model_type gpt2/dropout
```

这个脚本设计考虑了易用性和可扩展性，您可以根据需要进一步调整参数或添加功能。

        