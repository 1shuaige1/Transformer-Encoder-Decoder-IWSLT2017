# Transformer-Encoder-Decoder-IWSLT2017

本项目是《大模型基础与应用》课程的期中作业实现。  
实现完整 **Encoder–Decoder Transformer**，包含 **相对位置编码（T5 风格）** 与 **超参数敏感性分析**。

---

## 🏗️ 模型架构

### Transformer 架构
本项目实现了标准的 Transformer Encoder-Decoder 架构，包含以下核心组件：
- **多头注意力机制**：支持自注意力和交叉注意力
- **前馈神经网络**：位置前馈网络（Position-wise Feed-Forward Networks）
- **残差连接与层归一化**：增强梯度流动和训练稳定性
- **相对位置偏置**：T5 风格的相对位置编码，增强模型对序列位置信息的感知

### 相对位置偏置
采用 T5 风格的相对位置偏置实现：
- 创建大小为 (2*max_distance+1, num_heads) 的偏置表
- 对于位置 i 和 j 之间的注意力分数添加偏置 b_{i-j}
- 通过学习的方式获取相对位置信息，相比绝对位置编码更具表达能力

### 模型配置
默认模型配置：
- d_model: 256 (模型维度)
- num_layers: 4 (编码器和解码器层数)
- num_heads: 8 (注意力头数)
- d_ff: 1024 (前馈网络维度)
- dropout: 0.1 (Dropout 概率)
- max_len: 128 (最大序列长度)
- vocab_size: 16000 (词汇表大小)

---

## 📂 文件结构
```
src/
├── model_relpos.py           # 模型定义（Encoder–Decoder + 相对位置偏置）
├── data_iwslt.py             # 数据加载与 SentencePiece 分词器
├── train_mt.py               # 训练与验证
├── eval_bleu.py              # BLEU 评估
├── sample_mt.py              # 翻译示例
├── run_sensitivity.py        # 超参数敏感性分析脚本
├── ablation/                 # 消融实验相关文件
│   ├── model_ablation.py     # 支持消融实验的模型实现
│   ├── model_no_relpos.py    # 无相对位置偏置的模型变体
│   ├── train_ablation.py     # 消融实验训练脚本
│   ├── run_ablation_relpos.py# 相对位置偏置消融实验
│   └── run_comprehensive_ablation_v2.py # 综合消融实验
scripts/
├── run_iwslt.sh              # 一键运行脚本
results/
├── run_experiments/          # 各实验结果目录
│   ├── run_base/             # 基线模型结果
│   └── sensitivity/          # 超参分析实验结果
├── ablation_comprehensive_summary.csv # 消融实验结果汇总
├── sensitivity_d_model.csv   # d_model 敏感性分析结果
├── sensitivity_num_layers.csv# 层数敏感性分析结果
└── sensitivity_batch_size.csv# 批大小敏感性分析结果
```

---

## ⚙️ 环境配置

### 依赖项
项目依赖以下 Python 包：
- `torch>=2.0.0`：深度学习框架
- `datasets>=2.0.0`：数据集加载工具
- `sentencepiece`：子词分词器
- `sacrebleu`：BLEU 评分计算
- `numpy`：数值计算库
- `tqdm`：进度条显示
- `matplotlib`：结果可视化

### 安装依赖
```bash
python -m pip install -r requirements.txt
```

### 硬件要求
- GPU：推荐使用 CUDA 兼容的 GPU 以加速训练
- 内存：至少 8GB RAM
- 存储：至少 5GB 可用空间（用于数据集和模型存储）

## 🧪 示例用法

### 1. 训练模型：
```bash
bash scripts/run_iwslt.sh
```
此脚本将：
- 使用默认参数（d_model=256, num_layers=4, num_heads=8, epochs=20）训练模型
- 自动下载 IWSLT2017 英德数据集
- 训练 20 个 epoch 并保存检查点
- 默认使用 GPU 进行训练

### 2. 评估 BLEU：
```bash
python -m src.eval_bleu --ckpt results/run_experiments/run_base/ckpt_epoch20.pt --split validation
```
参数说明：
- `--ckpt`：模型检查点路径
- `--split`：评估数据集（validation/test）
- `--device`：计算设备（cuda/cpu，默认为cuda）

### 3. 翻译句子：
```bash
python -m src.sample_mt --ckpt results/run_experiments/run_base/ckpt_epoch20.pt --sentence "Hello, how are you?"
```
参数说明：
- `--ckpt`：模型检查点路径
- `--sentence`：待翻译的句子
- `--device`：计算设备（cuda/cpu，默认为cuda）

### 4. 自定义训练参数：
```bash
python -m src.train_mt --batch_size 64 --d_model 512 --num_layers 6 --epochs 20 --output_dir results/run_experiments/custom_run
```
常用参数：
- `--batch_size`：批处理大小（默认64）
- `--d_model`：模型维度（默认256）
- `--num_layers`：编码器/解码器层数（默认4）
- `--num_heads`：注意力头数（默认8）
- `--d_ff`：前馈网络维度（默认1024）
- `--epochs`：训练轮数（默认20）
- `--lr`：学习率（默认3e-4）
- `--output_dir`：输出目录路径

### 5. 运行消融实验：
```bash
# 相对位置偏置消融实验
python -m src.ablation.run_ablation_relpos

# 综合消融实验
python -m src.ablation.run_comprehensive_ablation_v2
```

---

## 🔍 超参数敏感性分析

为了探究模型性能对关键超参数的敏感程度，我们实现了 `run_sensitivity.py` 脚本，支持对单一变量进行批量实验，并记录 BLEU 分数与验证损失。

### 支持分析的参数

| 参数名        | 测试范围              |
|---------------|-----------------------|
| `d_model`     | 128, 256, 512         |
| `num_layers`  | 2, 4, 6               |
| `batch_size`  | 32, 64, 128           |

### 运行方法

```bash
python -m src.run_sensitivity --param d_model --output_summary results/sensitivity_d_model.csv
```

### 实验结果

#### d_model 敏感性分析
| d_model | BLEU  | Loss  |
|---------|-------|-------|
| 128     | 15.38 | 2.89  |
| 256     | 19.75 | 2.37  |
| 512     | 20.98 | 2.16  |

结果表明，随着模型维度的增加，BLEU 分数提升但提升幅度逐渐减小，同时计算开销显著增加。

#### num_layers 敏感性分析
| num_layers | BLEU  | Loss  |
|------------|-------|-------|
| 2          | 15.95 | 2.79  |
| 4          | 19.75 | 2.37  |
| 6          | 20.85 | 2.24  |

层数增加能提升性能，但 6 层相比 4 层的提升幅度较小，4 层在性能和效率间取得了较好平衡。

#### batch_size 敏感性分析
| batch_size | BLEU  | Loss  |
|------------|-------|-------|
| 32         | 20.89 | 2.28  |
| 64         | 19.75 | 2.37  |

较小的批大小（32）在本实验中表现略好，可能与梯度更新频率和正则化效应有关。

### 输出结果

- 每次实验的结果（BLEU、Loss）会保存为 `.csv` 文件。
- 自动生成 BLEU 与 Loss 随参数变化的趋势图。

示例图表路径：
- `results/sensitivity_d_model.png`
- `results/sensitivity_num_layers.png`
- `results/sensitivity_batch_size.png`

### 分析建议

通过对比不同参数下的 BLEU 分数与收敛速度，可以找出最优或性价比最高的超参组合，辅助后续模型优化决策。实验结果显示，d_model=512, num_layers=6 的配置获得最佳性能，但 d_model=256, num_layers=4 的配置在性能和效率间取得了良好平衡。

## 🧪 消融实验

我们进行了全面的消融实验，以评估模型各组件的重要性。

### 实验配置

| 实验名称 | 相对位置偏置 | Dropout | 激活函数 | 层数 | BLEU  | Loss  |
|----------|--------------|---------|----------|------|-------|-------|
| 基线模型 | ✓            | 0.1     | GELU     | 4    | 19.75 | 2.37  |
| 移除相对位置偏置 | ✗        | 0.1     | GELU     | 4    | 11.17 | 3.07  |
| 移除 Dropout | ✓          | 0.0     | GELU     | 4    | 19.30 | 2.35  |
| 替换激活函数 | ✓          | 0.1     | ReLU     | 4    | 19.14 | 2.47  |
| 浅层模型   | ✓            | 0.1     | GELU     | 2    | 15.95 | 2.79  |

### 结果分析

1. **相对位置偏置的重要性**：移除相对位置偏置导致 BLEU 分数从 19.75 降至 11.17，损失函数从 2.37 增加到 3.07，表明相对位置信息对翻译质量至关重要。

2. **Dropout 的作用**：移除 Dropout 后性能略有下降（19.75 → 19.30），说明 Dropout 在防止过拟合方面有一定作用。

3. **激活函数影响**：将 GELU 替换为 ReLU 后性能略有下降（19.75 → 19.14），表明 GELU 激活函数更适合本任务。

4. **模型深度**：从 4 层减少到 2 层导致显著性能下降（19.75 → 15.95），证明了足够模型深度的重要性。

实验结果充分验证了相对位置偏置在 Transformer 模型中的关键作用，其对模型性能的影响最为显著。
