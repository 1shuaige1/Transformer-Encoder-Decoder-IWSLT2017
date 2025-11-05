# Transformer-Encoder-Decoder-IWSLT2017

本项目实现了一个 从零构建的 Transformer 模型（Encoder-Decoder 架构），在 IWSLT2017 英德翻译数据集（en→de） 上进行训练与验证。

---

## 模型架构

模型实现了完整的：
- Multi-Head Self-Attention  
- Position-wise Feed Forward Network  
- Residual Connection + Layer Normalization  
- Sinusoidal 或 Learned 位置编码  

项目支持命令行超参配置、自动下载数据集、保存模型与训练验证曲线。

---

## 📂 项目结构
```
Transformer-IWSLT2017/
│
├── src/
│ ├── train.py # 主训练脚本：参数解析、训练循环、评估
│ ├── model.py # Transformer 模型定义（Encoder/Decoder）
│ └── data.py # 数据加载、分词、批处理封装
│
├── scripts/
│ └── run.sh # 一键训练脚本（含完整命令行）
│
├── results/ # 存放训练曲线(loss_curve.png)、模型权重(epochX.pt)
│
├── requirements.txt # 依赖库清单
└── README.md # 本文件
```

---

## ⚙️ 环境与硬件要求

| 组件 | 推荐版本 | 说明 |
|------|-----------|------|
| Python | ≥ 3.9 | 3.9~3.11均可 |
| PyTorch | ≥ 2.0 | 支持 CUDA |
| transformers | ≥ 4.44 | Hugging Face Tokenizer |
| datasets | ≥ 3.0 | 自动下载 IWSLT2017 |
| GPU | RTX 3060 / A100 / T4 | 推荐显存 ≥ 6GB |
| 操作系统 | Linux / Windows | 均可运行 |

---

## 📦 安装依赖

```bash
pip install -r requirements.txt
```

requirements.txt 内容：
torch>=2.0
datasets>=3.0.0
transformers>=4.44.0
tqdm
matplotlib
numpy

---

### 依赖项
项目依赖以下 Python 包：
- `torch>=2.0.0`：深度学习框架
- `datasets>=2.0.0`：数据集加载工具
- `sentencepiece`：子词分词器
- `sacrebleu`：BLEU 评分计算
- `numpy`：数值计算库
- `tqdm`：进度条显示
- `matplotlib`：结果可视化


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
