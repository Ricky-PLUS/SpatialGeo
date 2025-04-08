import os
import torch
import numpy as np
from model.moge_model import MoGeModel

device = torch.device("cuda")

# Load the model from huggingface hub (or load from local).
model = MoGeModel()   
print(model)                          
