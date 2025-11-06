# Transformer-Encoder-Decoder-IWSLT2017

本项目实现了一个 从零构建的 Transformer 模型（Encoder-Decoder 架构），在 IWSLT2017 英德翻译数据集（en→de） 上进行训练与验证。

---

## 项目结构
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

## 环境与硬件要求

| 组件 | 推荐版本 | 说明 |
|------|-----------|------|
| Python | ≥ 3.9 | 3.9~3.11均可 |
| PyTorch | ≥ 2.0 | 支持 CUDA |
| transformers | ≥ 4.44 | Hugging Face Tokenizer |
| datasets | ≥ 3.0 | 自动下载 IWSLT2017 |
| GPU | RTX 3060 / A100 / T4 | 推荐显存 ≥ 6GB |
| 操作系统 | Linux / Windows | 均可运行 |

---

## 安装依赖

```bash
pip install -r requirements.txt
```

### requirements.txt 内容： 
```
torch>=2.0  
datasets>=3.0.0  
transformers>=4.44.0  
tqdm  
matplotlib  
numpy  
```

---

## 运行方式

### 方式一；直接运行脚本
```
bash scripts/run.sh
```

### 方式二：运行命令
```
python src/train.py \
  --epochs 10 \
  --batch_size 64 \
  --d_model 256 \
  --n_heads 4 \
  --n_layers 2 \
  --d_ff 1024 \
  --dropout 0.1 \
  --lr 3e-4 \
  --max_len 128 \
  --seed 42 \
  --limit_train_samples 48880 \
  --device cuda \
  --save_dir results
```
---

## 实验可复现性
为确保实验结果完全可重复，代码中固定了所有随机种子。
```
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```
- 数据加载与划分：使用 Hugging Face datasets 提供的官方 IWSLT2017 (en→de) 版本
- Tokenizer：Helsinki-NLP/opus-mt-en-de
- 样本数量：limit_train_samples = 48880
- 验证集使用官方 validation split
- 优化器：AdamW(lr=3e-4)
- Loss：CrossEntropy(ignore_index=pad_id)
- Gradient Clip：max_norm=1.0

---

## 输出结果

训练完成后，所有结果会自动保存在 `results/` 目录下，文件说明如下：

| 文件名 | 说明 |
|--------|------|
| `epoch1.pt`, `epoch2.pt`, ... | 各训练轮次保存的模型权重（PyTorch `state_dict` 格式） |
| `best_model.pt` | 验证集上表现最优的模型（最低验证损失） |
| `loss_curve.png` | 训练与验证损失曲线图，可用于观察模型收敛情况 |
| `train_log.txt` *(可选)* | 若开启日志保存，则记录每轮训练与验证损失、时间等信息 |

示例输出目录结构如下：
```
results/
│
├── epoch1.pt
├── epoch2.pt
├── ...
├── epoch10.pt
├── best_model.pt
├── loss_curve.png
└── train_log.txt
```

**曲线示例说明：**

- 蓝色曲线（Train Loss）：训练集损失，应随 epoch 稳定下降；
- 橙色曲线（Validation Loss）：验证集损失，通常在 5～8 轮后趋于平稳；
- 若验证损失上升，说明模型开始过拟合，可考虑增大 dropout、引入学习率调度或早停策略。

---

## 模型结构概览

- MultiHeadAttention：多头注意力机制（缩放点积）
- PositionwiseFeedForward：逐位置前馈网络（Linear→ReLU→Linear）
- PositionalEncoding：正弦/可学习位置编码
- EncoderLayer：Self-Attention + FFN + 残差 + LayerNorm
- DecoderLayer：Masked Self-Attn + Cross-Attn + FFN + 残差 + LayerNorm
- TransformerModel：整体 Encoder–Decoder 架构，含嵌入与输出投影
