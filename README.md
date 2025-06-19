<span style="color:rgb(0,0,255)">By RuoChen from ZJU</span>
## CUDA = 12.4
# 在主目录下执行
## git clone https://github.com/haotian-liu/LLaVA.git
## cd LLaVA
## conda create -n llava python=3.10 -y
## conda activate llava
## pip install --upgrade pip  
## pip install -e .
---
# 在models目录下执行
## git lfs clone https://hf-mirror.com/llava-hf/llava-v1.6-vicuna-7b-hf
---
# 在datas目录下执行
## git lfs clone https://hf-mirror.com/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K
# 并解压images.zip
## unzip images.zip -d images
---
# 国内下载问题可以用网盘代替
## 通过网盘分享的文件：Llava-next必要文件
链接: https://pan.baidu.com/s/1WxmTcl-NVXqNFSDUG2XeoA?pwd=1234 提取码: 1234 

---
# 文件说明
## python test.py  # 测试原模型
## python train_Lora.py  # 用Lora训练模型
## python val_Lora.py # 验证Lora模型
## data.py # 定义自己的数据集及加载方式
## data.ipynb # jupyter notebook版本的data.py
---
# <span style="color:red">Acknowledgment</span>
## <span style="color:blue">This project is not possible without the following codebases.：</span>
https://github.com/haotian-liu

https://github.com/yuanzhoulvpi2017/zero_nlp

B站：良睦路程序员
