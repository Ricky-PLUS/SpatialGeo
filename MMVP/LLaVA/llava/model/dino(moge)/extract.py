import cv2
import torch
from PIL import Image
import numpy as np
import os
import pdb

device = torch.device("cuda")

# model_path = os.path.expanduser("./vit_model.pt")

state_dict = torch.load("./vit_model.pt")
print("参数名及形状：")
for key, value in state_dict.items():
    print(f"{key}: {tuple(value.shape)}")
# # Load the model from huggingface hub (or load from local).
# model = MoGeModel.from_pretrained(model_path).to(device)    
# print(model)

# # 提取 VIT 部分的参数
# vit_state_dict = {k.replace("backbone.", ""): v for k, v in model.state_dict().items() if k.startswith("backbone.")}
# # 保存 VIT 部分的参数
# torch.save(vit_state_dict, "./vit_model.pt")

# checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
