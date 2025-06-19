import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

# 加载模型
local_model_path = "./models/llava-v1.6-vicuna-7b-hf"

processor = LlavaNextProcessor.from_pretrained(
    local_model_path,
    use_fast=False
)

model = LlavaNextForConditionalGeneration.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 加载图片
image = Image.open("./data/LLaVA-CC3M-Pretrain-595K/images/GCC_train_000000000.jpg")  # 替换为您的图片路径
prompt = "USER: <image>\n请详细描述这张图片的内容\nASSISTANT:"

# 正确的参数顺序：images在前，text在后
inputs = processor(
    images=image,      # 先传递图片
    text=prompt,       # 再传递文本
    return_tensors="pt"
).to(model.device)
print(inputs.keys())  # 打印输入的键以确认
print(inputs['pixel_values'].shape)  # 打印图片张量的形状以确认
print(inputs['image_sizes'])  # 打印文本张量的形状以确认
# 生成回复
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

# 解码输出
response = processor.decode(output[0], skip_special_tokens=True)

if "ASSISTANT:" in response:
    assistant_response = response.split("ASSISTANT:")[-1].strip()
else:
    assistant_response = response.split("</s>")[-1].strip()  # 备选分割方式

print("模型回复:")
print(assistant_response)