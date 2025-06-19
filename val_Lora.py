import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from peft import PeftModel

# 模型路径
base_model_path = "./models/llava-v1.6-vicuna-7b-hf"  # 基础模型路径
lora_model_path = "./llava-lora-checkpoints/checkpoint-50"  # 微调后的LoRA检查点路径

# 加载处理器
processor = LlavaNextProcessor.from_pretrained(
    base_model_path,
    use_fast=False
)

# 加载基础模型
print("加载基础模型...")
base_model = LlavaNextForConditionalGeneration.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 加载LoRA权重
print("加载LoRA权重...")
model = PeftModel.from_pretrained(base_model, lora_model_path)

# 可选：合并LoRA权重到基础模型（提高推理速度）
# model = model.merge_and_unload()

print("模型加载完成!")

# 加载图片
image = Image.open("./data/LLaVA-CC3M-Pretrain-595K/images/GCC_train_000000000.jpg")

# 设置提示词
prompt = "USER: <image>\n请详细描述这张图片的内容\nASSISTANT:"

# 正确的参数顺序：images在前，text在后
inputs = processor(
    images=image,      # 先传递图片
    text=prompt,       # 再传递文本
    return_tensors="pt"
).to(model.device)

print("输入信息:")
print(f"输入键: {inputs.keys()}")
print(f"图片张量形状: {inputs['pixel_values'].shape}")
print(f"图片尺寸: {inputs['image_sizes']}")

# 生成回复
print("开始生成回复...")
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=processor.tokenizer.eos_token_id  # 添加pad_token_id
    )

# 解码输出
response = processor.decode(output[0], skip_special_tokens=True)

# 提取助手回复
if "ASSISTANT:" in response:
    assistant_response = response.split("ASSISTANT:")[-1].strip()
else:
    assistant_response = response.split("</s>")[-1].strip()  # 备选分割方式

print("\n" + "="*50)
print("微调模型回复:")
print("="*50)
print(assistant_response)
print("="*50)