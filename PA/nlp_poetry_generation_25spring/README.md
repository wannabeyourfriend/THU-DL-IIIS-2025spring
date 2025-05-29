# Tang  Poem Generation Problem

## Instruction

模型基于`Qwen1.5-1.8b-chat`进行SFT微调，在唐诗数据集上进行了微调

## Problem Statement

输入是唐诗的第一句

输出是整首唐诗

## Demo

查看[website](https://wannabeyourfriend.github.io/Qwen1.5-1.8B-chat-SFT-tang-poetry.github.io/)可以看到10个对比的例子；

```txt
https://wannabeyourfriend.github.io/Qwen1.5-1.8B-chat-SFT-tang-poetry.github.io/
```

## File Structure

```txt
(base) root@autodl-container-dbce458354-eb145c04:~/autodl-tmp/PA2# tree
.
├── PA2_backup.tar.gz
├── README.md
├── data
│   ├── README.md
│   ├── gitattributes
│   └── tang_poems
│       ├── test-00000-of-00001-a794cd4c018c9326.parquet
│       └── train-00000-of-00001-6914ee5fabc145c0.parquet
├── eval.ipynb
├── gitignore
├── pretrain
│   ├── Qwen1.5-1.8B-Chat
│   │   ├── LICENSE
│   │   ├── README.md
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── merges.txt
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
│   └── final_poetry_adapter
│       ├── README.md
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── added_tokens.json
│       ├── merges.txt
│       ├── special_tokens_map.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── training_args.bin
│       └── vocab.json
├── requirements.txt
├── results_qwen1.5_poetry
│   └── checkpoint-155
│       ├── README.md
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── added_tokens.json
│       ├── merges.txt
│       ├── optimizer.pt
│       ├── rng_state.pth
│       ├── scheduler.pt
│       ├── special_tokens_map.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── trainer_state.json
│       ├── training_args.bin
│       └── vocab.json
├── src
│   ├── dataset.py
│   ├── eval.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── train.ipynb
└── wandb
    ├── debug-internal.log -> run-20250529_205043-dxnbr2y3/logs/debug-internal.log
    ├── debug.log -> run-20250529_205043-dxnbr2y3/logs/debug.log
    ├── latest-run -> run-20250529_205043-dxnbr2y3
    └── run-20250529_205043-dxnbr2y3
        ├── files
        │   ├── output.log
        │   ├── requirements.txt
        │   └── wandb-metadata.json
        ├── logs
        │   ├── debug-core.log -> /root/.cache/wandb/logs/core-debug-20250529_205042.log
        │   ├── debug-internal.log
        │   └── debug.log
        ├── run-dxnbr2y3.wandb
        └── tmp
            └── code

15 directories, 57 files
```

## Check

您可以进入`train.ipynb`详细查看训练与评估情况

## Observation

微调后的模型具有更强的格律遵循的能力，显著提升效果。