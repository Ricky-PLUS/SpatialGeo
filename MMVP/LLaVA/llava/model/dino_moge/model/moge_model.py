from typing import *
from numbers import Number
from pathlib import Path
import importlib

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils
import torch.utils.checkpoint
import torch.version
import torch.nn.functional as F
import numpy as np
from PIL import Image

from .utils import wrap_dinov2_attention_with_sdpa, wrap_module_with_gradient_checkpointing


class MoGeModel(nn.Module):
    image_mean: torch.Tensor
    image_std: torch.Tensor

    def __init__(self, 
        encoder: str = 'dinov2_vitl14', 
        intermediate_layers: Union[int, List[int]] = 1,
        trained_area_range: Tuple[Number, Number] = (500 * 500, 700 * 700),
        delay_load=False
    ):
        super(MoGeModel, self).__init__()

        self.encoder = encoder
        self.intermediate_layers = intermediate_layers
        self.trained_area_range = trained_area_range
        self.is_loaded = False
        
        import pdb
        pdb.set_trace()
        
        hub_loader = getattr(importlib.import_module(".dinov2.hub.backbones", __package__), encoder)
        self.backbone = hub_loader(pretrained=False)
        
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)
        
        if not delay_load:
            self.load_model()
        
        if torch.__version__ >= '2.0':
            self.enable_pytorch_native_sdpa()

    def load_model(self, pretrained_model_name_or_path: Union[str, Path, IO[bytes]] = "./vit_model.pt"):
        import pdb
        pdb.set_trace()
        if Path(pretrained_model_name_or_path).exists():
            checkpoint = torch.load(pretrained_model_name_or_path, map_location='cpu', weights_only=True)
            
        self.backbone.load_state_dict(checkpoint, strict=False)
        self.backbone.requires_grad_(False)
        
        self.is_loaded = True

    @staticmethod
    def cache_pretrained_backbone(encoder: str, pretrained: bool):
        _ = torch.hub.load('facebookresearch/dinov2', encoder, pretrained=pretrained)

    def load_pretrained_backbone(self):
        "Load the backbone with pretrained dinov2 weights from torch hub"
        state_dict = torch.hub.load('facebookresearch/dinov2', self.encoder, pretrained=True).state_dict()
        self.backbone.load_state_dict(state_dict)
    
    def enable_backbone_gradient_checkpointing(self):
        for i in range(len(self.backbone.blocks)):
            self.backbone.blocks[i] = wrap_module_with_gradient_checkpointing(self.backbone.blocks[i])

    def enable_pytorch_native_sdpa(self):
        for i in range(len(self.backbone.blocks)):
            self.backbone.blocks[i].attn = wrap_dinov2_attention_with_sdpa(self.backbone.blocks[i].attn)
            
    def process_image_(self, image: torch.Tensor, device):
        """
        moge处理image_tensor的方法

        Args:
            image: torch.Tensor

        Returns:
            image_14: image的w和h经过处理后，均为14的倍数。
        """
        
        if image.dim() == 3:
            image = image.unsqueeze(0)        

        original_height, original_width = image.shape[-2:]
        area = original_height * original_width

        expected_area = 500000

        if expected_area != area:
            expected_width, expected_height = int(original_width * (expected_area / area) ** 0.5), int(original_height * (expected_area / area) ** 0.5)
            image = F.interpolate(image, (expected_height, expected_width), mode="bicubic", align_corners=False, antialias=True)
            
        raw_img_h, raw_img_w = image.shape[-2:]
        patch_h, patch_w = raw_img_h // 14, raw_img_w // 14

        image = (image - self.image_mean) / self.image_std

        image_14 = F.interpolate(image, (patch_h * 14, patch_w * 14), mode="bilinear", align_corners=False, antialias=True)
        
        return image_14
    
          
    def processor_moge(self, image):
        """
        将moge的processor集成（旧版的process过程较为分散） 
        
        args:
            image:如果为路径，则读取图片。否则直接处理。
        
        return：dino(moge)_encoder的input
        
        """
        device_moge = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if isinstance(image, str):
            # dino(moge)部分的image processor

            pil_image = Image.open(image).convert('RGB')
            input_array = np.array(pil_image, dtype=np.float32)

            image = torch.tensor(input_array / 255.0, 
                                    dtype=torch.float32,
                                    device=device_moge).permute(2, 0, 1)  # HWC -> CHW
        
        image = image.unsqueeze(0)
        image_tensor = self.process_image_(image, device_moge)

        return image_tensor
    

    @torch.no_grad()
    def forward(self, image: torch.Tensor, mixed_precision: bool = False) -> Dict[str, torch.Tensor]:
        image_14 = self.processor_moge(image)

        # Get intermediate layers from the backbone
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=mixed_precision):
            features = self.backbone.get_intermediate_layers(image_14, self.intermediate_layers, return_class_token=False)

        return features
    
    @property
    def hidden_size(self):
        #return self.config.hidden_size
        return 1024


