# 🌋 LLaVA Fine-tuning Framework

> **By RuoChen from ZJU**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/CUDA-12.4-green.svg" alt="CUDA">
  <img src="https://img.shields.io/badge/Model-LLaVA--v1.6-purple.svg" alt="LLaVA">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

## 📋 目录

- [环境配置](#-环境配置)
- [模型下载](#-模型下载)
- [数据集准备](#-数据集准备)
- [训练与测试](#-训练与测试)
- [项目结构](#-项目结构)
- [致谢](#-致谢)

---

## 🚀 环境配置

### 系统要求
- **CUDA**: 12.4

### 安装步骤

在主目录下执行：

```bash
# 克隆LLaVA仓库
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA

# 创建虚拟环境
conda create -n llava python=3.10 -y
conda activate llava

# 安装依赖
pip install --upgrade pip
pip install -e .
```

---

## 📥 模型下载

在 `models` 目录下执行：

```bash
git lfs clone https://hf-mirror.com/llava-hf/llava-v1.6-vicuna-7b-hf
```

---

## 📊 数据集准备

### 下载数据集

在 `datas` 目录下执行：

```bash
# 下载LLaVA预训练数据集
git lfs clone https://hf-mirror.com/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K

# 解压图像文件
unzip images.zip -d images
```

### 💡 国内下载方案

如遇下载问题，可使用网盘代替：

> **通过网盘分享的文件**：Llava-next必要文件  
> 🔗 链接: https://pan.baidu.com/s/1WxmTcl-NVXqNFSDUG2XeoA?pwd=1234  
> 🔑 提取码: 1234

---

## 🎯 训练与测试

### 文件功能说明

| 文件名 | 功能描述 |
|--------|---------|
| `test.py` | 测试原始模型 |
| `train_Lora.py` | 使用LoRA方法训练模型 |
| `val_Lora.py` | 验证LoRA模型效果 |
| `data.py` | 定义自定义数据集及加载方式 |
| `data.ipynb` | Jupyter Notebook版本的data.py |

### 使用示例

```bash
# 测试原始模型
python test.py

# LoRA微调训练
python train_Lora.py

# 验证LoRA模型
python val_Lora.py
```

---

## 📁 项目结构

```
.
├── LLaVA/                    # LLaVA主代码库
├── models/
│   └── llava-v1.6-vicuna-7b-hf/  # LLaVA模型文件
├── datas/
│   ├── LLaVA-CC3M-Pretrain-595K/ # 预训练数据集
│   └── images/                    # 解压后的图像文件
├── test.py                   # 测试脚本
├── train_Lora.py            # LoRA训练脚本
├── val_Lora.py              # 验证脚本
├── data.py                  # 数据集定义
└── data.ipynb               # Notebook版本
```

---

## 🙏 致谢

<p align="center">
  <strong>This project would not be possible without the following codebases:</strong>
</p>

<p align="center">
  <a href="https://github.com/haotian-liu">haotian-liu</a> • 
  <a href="https://github.com/yuanzhoulvpi2017/zero_nlp">zero_nlp</a> • 
  <a href="https://space.bilibili.com/">B站：良睦路程序员</a>
</p>

---

<p align="center">
  <i>如有问题，欢迎提交 Issue 或 PR！</i>
</p>
