import torch
from accelerate.utils import memory

if not hasattr(memory, 'clear_device_cache'):
    memory.clear_device_cache = lambda: None
from transformers import (
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from data import LlavaDataset, TrainLLavaModelCollator

# 配置参数
MODEL_NAME = "./models/llava-v1.6-vicuna-7b-hf"
DATASET_DIR = "./data/LLaVA-CC3M-Pretrain-595K"
LORA_RANK = 4
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
IGNORE_INDEX = -100  # 与 collator 中的 ignore_index 一致

# 加载模型和处理器
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
processor = LlavaNextProcessor.from_pretrained(MODEL_NAME, use_fast=False)
model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 配置 LoRA
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 注意力层
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 检查可训练参数

# 准备数据集和数据整理器
train_dataset = LlavaDataset(DATASET_DIR)
data_collator = TrainLLavaModelCollator(processor=processor, IGNORE_INDEX=IGNORE_INDEX)

# 训练配置 - 修改为按步数保存
training_args = TrainingArguments(
    output_dir="./llava-lora-checkpoints",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    fp16=True,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,

    # 修改保存策略：按步数保存而不是按epoch
    save_strategy="steps",
    save_steps=50,  # 每50步保存一次，你可以根据需要调整这个数字

    # 可选：限制保存的检查点数量，避免占用太多磁盘空间
    save_total_limit=5,  # 只保留最近的5个检查点

    # 可选：如果你想要加载最佳模型
    load_best_model_at_end=False,  # 如果有验证集可以设为True

    remove_unused_columns=False,  # 保留自定义字段
    label_names=["labels"]
)

# 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

# 开始训练
trainer.train()