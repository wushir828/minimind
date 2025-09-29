这是我使用 `minimind` 开源项目亲手训练的全套模型权重文件。
## 🙏 致谢与来源 (Acknowledgements & Source)

本项目是对 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 开源项目的学习、复现和实验。所有的核心代码和设计理念均归原作者 **Jingyao Gong** 所有。

我在此基础上，完整地搭建了环境，成功地进行了预训练、SFT、DPO和LoRA等一系列训练，并记录了详细的笔记和结果。这个仓库是我个人的学习和实践成果展示。

再次感谢原作者的无私开源，让我有机会深入学习大语言模型从零开始的全过程！

**训练包含内容:**
- `pretrain_512.pth`: 预训练模型
- `full_sft_512.pth`: SFT对话模型
- `rlhf_512.pth`: DPO优化模型
- `reason_512.pth`: 推理模型
- `lora/`: 包含 `lora_medical` 和 `lora_identity` 两个LoRA适配器
---

---

### 附录：LoRA超参数调优实验

为了探索超参数对模型训练效果的影响，我针对LoRA微调过程中的**学习率**进行了一组A/B对比实验。

#### 实验A：基准训练

首先，我使用`train_lora.py`脚本中默认的学习率 `1e-4` (0.0001) 进行了一次完整的LoRA训练。

**使用的命令行代码：**
```bash
# 使用默认学习率 1e-4 进行训练
torchrun --nproc_per_node 8 train_lora.py \
  --data_path ../dataset/lora_medical.jsonl \
  --lora_name "lora_medical_baseline" \
  --use_wandb \
  --save_interval 50
```
训练结果曲线图(学习率: 1e-4):<img width="1074" height="540" alt="Image" src="https://github.com/user-attachments/assets/144b0d71-6985-41a3-9d85-25cabc4bed7a" />
#### 实验B：调参训练
接下来，我将学习率增大了5倍至 5e-4 (0.0005)，其他所有参数保持不变，再次进行训练。
**使用的命令行代码：**
```bash
# 使用调整后的学习率 5e-4 进行训练
torchrun --nproc_per_node 8 train_lora.py \
  --data_path ../dataset/lora_medical.jsonl \
  --lora_name "lora_medical_tuned" \
  --use_wandb \
  --save_interval 50 \
  --learning_rate 5e-4
```
 训练结果曲线图 (学习率: 5e-4):<img width="1077" height="468" alt="Image" src="https://github.com/user-attachments/assets/36b4d1c9-1cce-4459-9c22-835a800a0bf7" />

#### 训练过程可视化 

本项目使用 **WandB (Weights & Biases)** 工具来记录和可视化训练过程。使用流程如下：
1.  **注册账号**: 访问 `wandb.ai` 网站免费注册一个账号，并在个人设置中获取API Key。
2.  **登录环境**: 在训练服务器的命令行中运行 `wandb login`，并根据提示粘贴API Key来完成登录。
3.  **启动训练**: 在`python`或`torchrun`训练命令的末尾添加 `--use_wandb` 参数。
4.  **查看结果**: 运行命令后，将命令行输出的WandB链接复制到浏览器中，即可访问实时更新的训练仪表盘，查看Loss、Learning Rate等指标的变化曲线，并进行不同实验之间的对比分析。

---

### 附录：LoRA训练静默失败问题排查

在进行LoRA微调时，我遇到了一个“静默失败”的问题：训练脚本（`train_lora.py`）能够无报错地运行结束，但在指定的`out/`目录下并未生成预期的`.pth`模型存档文件。

#### 1. 问题调查

通过 `ls -R out/` 命令检查，我确认了没有任何 `lora_*.pth` 文件被创建。这排除了文件被保存在非预期位置的可能性，并将问题指向了训练脚本的内部逻辑。

#### 2. 原因分析

通过审查`trainer/train_lora.py`的代码，我定位到了负责保存模型的 `if` 条件语句：
```python
if (step + 1) % args.save_interval == 0:
    # ... save model code ...
```
同时，我注意到脚本为 save_interval 设置的默认值是 100 步。
```python
parser.add_argument("--save_interval", type=int, default=100)
```

然而，通过对lora_medical.jsonl和lora_identity.jsonl这两个小型数据集的训练日志进行分析（例如 Epoch: [10/10] (0/1)），我发现它们每一轮（Epoch）的总训练步数远小于100（lora_identity甚至只有1步）。

结论：由于单轮的总训练步数小于默认的保存间隔，导致保存模型的条件永远无法被触发。因此，训练虽然正常结束，但模型从未被保存。
#### 3. 解决方案
解决方案是在运行训练命令时，通过命令行参数动态地覆盖默认的save_interval值，使其小于或等于单轮的总步数。

修正后的命令行指令示例：
```
# 修正 lora_medical 训练命令 (假设其总步数为99)
torchrun --nproc_per_node 8 train_lora.py \
  --data_path ../dataset/lora_medical.jsonl \
  --lora_name "lora_medical" \
  --use_wandb \
  --save_interval 50

# 修正 lora_identity 训练命令 (总步数为1)
torchrun --nproc_per_node 8 train_lora.py \
  --data_path ../dataset/lora_identity.jsonl \
  --lora_name "lora_identity" \
  --use_wandb \
  --save_interval 1
```

## ☁️ 云端迁移与分布式训练流程

本项目的预训练和SFT训练是在一台**单卡RTX 4090**服务器上完成的。为了加速后续的DPO和LoRA实验，我将整个工作环境和项目成果无缝迁移到了一台**8卡RTX 4090**服务器上。以下是完整的操作流程：

### 1. 成果打包与备份 

在单卡服务器上，首先等待所有训练任务结束。然后，利用**AutoDL平台的“保存镜像”功能**，将整个服务器的当前状态（包括项目代码、所有数据集、已配置好的Conda环境`minimind_env`、以及`out/`目录下的所有`.pth`模型存档）完整地打包成一个个人自定义镜像。

这是最可靠的备份和迁移方式，避免了重复上传数据和配置环境的繁琐工作。

### 2. 释放旧资源与创建新实例 (Releasing Old & Creating New Instance)

镜像保存成功后，彻底**`释放`**单卡服务器实例以停止计费。

随后，在“算力市场”租用一台8卡RTX 4090服务器。在创建实例时，从**“我的镜像”**中选择上一步保存好的自定义镜像进行部署。

### 3. 环境恢复与验证 (Environment Restoration and Verification)

启动新的8卡实例后，通过JupyterLab连接。由于使用了自定义镜像，无需任何额外操作，所有的项目文件和Conda环境都已自动恢复，实现了**“开箱即用”**。

在Terminal中，只需运行 `conda activate minimind_env` 即可直接进入之前配置好的工作环境。

### 4. 启动多卡训练 (Launching Multi-GPU Training)

对于后续的训练任务（如DPO、LoRA），使用了PyTorch的分布式训练启动器 `torchrun` 来利用全部8张GPU，从而大幅提升训练效率。

**8卡DPO训练启动命令示例：**
```bash
# 在 trainer 目录下，使用8卡GPU启动DPO训练
torchrun --nproc_per_node 8 train_dpo.py --use_wandb
```