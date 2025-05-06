# Foundation Model - GPT2
## Overview
这是一个基于GPT-2架构的语言模型实现项目，主要包含了模型定义、训练流程和评估方法。项目实现了两种训练方式：
1. SFT (Supervised Fine-Tuning) - 监督微调
2. DPO (Direct Preference Optimization) - 直接偏好优化
这些实现将使模型能够正确学习语言建模任务，并通过人类偏好进行优化。

## Run
参考checkpoints/source.txt,下载模型权重

```bash
python demo.py
```

```bash
python demo.py -c path/to/your/checkpointfile.pt
```

```bash
python demo.py --max_new_tokens 200 --temperature 0.9 --top_p 0.95
```

```bash
python demo.py --output_dir custom_demo_folder
```

```bash
python demo.py --model_type gpt2/dropout
```
## Train
### SFT
```bash
python train_sft.py
```

### DPO
```bash
python train_dpo.py
```

### Model Behavior

|         | Positive Question                                            | Negative Question                                            |
| ------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Base    | ![image-20250506114016852](assets/image-20250506114016852.png) | ![image-20250506105236875](assets/image-20250506105236875.png) |
| **SFT** | ![image-20250506114246619](assets/image-20250506114246619.png) | ![image-20250506105357903](assets/image-20250506105357903.png) |
| **DPO** | ![image-20250506114333207](assets/image-20250506114333207.png) | ![image-20250506113131415](assets/image-20250506113131415.png) |

