import pandas as pd
import torch
import json
from PIL import Image
from dataclasses import dataclass
from torch.utils.data import Dataset
from pathlib import Path
from transformers import AutoProcessor
from transformers import LlavaNextProcessor
from dataclasses import dataclass


class LlavaDataset(Dataset):
    def __init__(self, dataset_dir: str) -> None:
        super().__init__()

        self.chat_data, self.image_dir = self.build_dataset(data_dir=dataset_dir)

    def build_dataset(self, data_dir: str) -> None:
        data_dir = Path(data_dir)
        image_dir = data_dir.joinpath("images")
        chat_file = Path(data_dir).joinpath("chat.json")

        # 修改：使用json模块而不是pandas来读取JSON文件
        try:
            with open(chat_file, 'r', encoding='utf-8') as f:
                chat_data = json.load(f)
        except Exception as e:
            print(f"Error reading JSON file: {e}")
            # 备用方案：尝试逐行读取
            chat_data = []
            with open(chat_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        chat_data.append(json.loads(line.strip()))

        return chat_data, image_dir

    def __len__(self):
        return len(self.chat_data)

    # 用这个函数组合一个自己的数据格式，第一个元素为人类输入，第二个元素为GPT输出，第三个元素为图片路径
    # 这里的图片路径是相对于data_dir的
    def __getitem__(self, index):
        cur_data = self.chat_data[index]
        human_input = cur_data['conversations'][0]['value']
        gpt_output = cur_data['conversations'][1]['value']
        image_path = self.image_dir.joinpath(cur_data['image'])
        return (human_input, gpt_output, image_path)


@dataclass
class QaImageOutput:
    """
    LLaVA-Next 格式的对话消息输出
    """
    q_input_ids: torch.Tensor
    pixel_values: torch.Tensor
    image_size: torch.Tensor
    GPT_a: torch.Tensor


def build_qa_image(processor, human_input, gpt_output, image_path):
    """
    构建符合 LLaVA-Next 格式的对话消息

    参数:
    processor - LlavaNextProcessor 实例
    human_input - 用户输入文本
    gpt_output - 之前的模型回复 (用于多轮对话)
    image_path - 图片路径

    返回:
    符合 LLaVA-Next 格式的提示字符串
    """
    # LLaVA-Next 使用固定的角色标识 "USER" 和 "ASSISTANT"
    # 图像标记 <image> 必须放在 USER 消息的开头

    # 构建多轮对话历史
    messages = []
    messages.append({"role": "USER", "content": f"{human_input}"})

    # 构建符合 LLaVA-Next 的提示字符串
    prompt = ""
    for msg in messages:
        if msg["role"] == "USER":
            prompt += f"USER: {msg['content']}\n"
        elif msg["role"] == "ASSISTANT":
            prompt += f"ASSISTANT: {msg['content']}</s>"

    # 添加 ASSISTANT: 提示模型生成回复
    prompt += "ASSISTANT:"

    image_file = image_path
    raw_image = Image.open(image_file)
    # print(raw_image.size)
    inputs = processor(
        images=raw_image,
        text=prompt,
        return_tensors="pt",
    )
    # 将回答转换为ids
    gpt_a = processor(text=gpt_output, )
    # print(inputs.image_sizes)
    # print(inputs.input_ids)
    # if inputs.pixel_values.dim() == 5:
    #     inputs.pixel_values = inputs.pixel_values[:, 0]  # 取第一个裁剪
    return QaImageOutput(
        q_input_ids=torch.tensor(inputs.input_ids),
        pixel_values=torch.tensor(inputs.pixel_values),
        image_size=torch.tensor(inputs.image_sizes),
        GPT_a=torch.tensor(gpt_a.input_ids)
    )


class TrainLLavaModelCollator:
    def __init__(self, processor: LlavaNextProcessor, IGNORE_INDEX: int) -> None:
        self.processor = processor
        self.ignore_index = IGNORE_INDEX

    def convert_one_piece(self,
                          q_input_ids: torch.Tensor,
                          a_input_ids: torch.Tensor, ) -> None:
        input_ids = torch.concat([
            q_input_ids,
            a_input_ids,
            torch.tensor([self.processor.tokenizer.eos_token_id]).reshape(1, 1)
        ], axis=1)

        labels = torch.concat([
            torch.full_like(q_input_ids, fill_value=self.ignore_index),
            a_input_ids,
            torch.tensor([self.processor.tokenizer.eos_token_id]).reshape(1, 1)
        ], axis=1)

        return input_ids, labels

    def __call__(self, features: list):
        input_ids_list = []
        labels_list = []
        pixel_values = []
        max_input_len_list = []
        image_sizes = []

        for feature in features:
            qaimage_output = build_qa_image(
                processor=self.processor,
                human_input=feature[0],
                gpt_output=feature[1],
                image_path=feature[2]
            )
            temp_input_ids, temp_labels = self.convert_one_piece(
                q_input_ids=qaimage_output.q_input_ids,
                a_input_ids=qaimage_output.GPT_a
            )
            max_input_len_list.append(temp_input_ids.shape[1])
            input_ids_list.append(temp_input_ids)
            labels_list.append(temp_labels)
            pixel_values.append(qaimage_output.pixel_values)
            image_sizes.append([224, 224])

        max_input_len = max(max_input_len_list)

        final_input_ids = torch.concat([
            torch.concat(tensors=[
                torch.full(
                    size=(1, max_input_len - max_input_len_list[index]),
                    fill_value=self.processor.tokenizer.pad_token_id),
                value,
            ], axis=1)
            for index, value in enumerate(input_ids_list)
        ])

        final_labels = torch.concat([torch.concat([
            torch.full(
                size=(1, max_input_len - max_input_len_list[index]), fill_value=self.ignore_index)
            , value
        ], axis=1) for index, value in enumerate(labels_list)])

        final_pixel_values = torch.concat(pixel_values, axis=0)

        attention_mask = torch.ones_like(final_input_ids)
        attention_mask[final_input_ids ==
                       self.processor.tokenizer.pad_token_ids] = 0

        return {
            "input_ids": final_input_ids,
            "labels": final_labels,
            "pixel_values": final_pixel_values,
            "attention_mask": attention_mask,
            "image_sizes": image_sizes  # 新增：返回图像尺寸列表
        }





