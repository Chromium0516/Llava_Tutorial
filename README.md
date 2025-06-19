<span style="color:rgb(0,0,255)">By RuoChen from ZJU</span>
## CUDA = 12.4
# Execute in the home directory
## git clone https://github.com/haotian-liu/LLaVA.git
## cd LLaVA
## conda create -n llava python=3.10 -y
## conda activate llava
## pip install --upgrade pip  
## pip install -e .
---
# Execute in the models directory
## git lfs clone https://hf-mirror.com/llava-hf/llava-v1.6-vicuna-7b-hf
---
# Execute in the datas directory
## git lfs clone https://hf-mirror.com/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K
# Unzip images.zip
## unzip images.zip -d images
 
---
# File Descriptions
## python test.py  # Test the original model
## python train_Lora.py  # Train the model using Lora
## python val_Lora.py # Validate the Lora model
## data.py # Define custom dataset and loading methods
## data.ipynb # Jupyter Notebook version of data.py
---
# <span style="color:red">Acknowledgment</span>
## <span style="color:blue">This project would not be possible without the following codebases:</span>
https://github.com/haotian-liu
https://github.com/yuanzhoulvpi2017/zero_nlp
