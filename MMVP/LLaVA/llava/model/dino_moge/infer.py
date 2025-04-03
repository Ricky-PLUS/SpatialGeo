import os
import torch
import numpy as np
from model.moge_model import MoGeModel

device = torch.device("cuda")

# Load the model from huggingface hub (or load from local).
model_path = "./vit_model.pt"
print(model_path)
model = MoGeModel.from_pretrained(model_path).to(device)   
print(model)                          
