import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils
import torch.utils.checkpoint
import torch.version

class FusionBlock(nn.Module):
    def __init__(self, num_blocks: int = 4):
        super().__init__()
        self.projects = nn.ModuleList([
            nn.Linear(1024, 1024) for _ in range(num_blocks)
        ])
    
    def forward(self, features):
        fusion_features = torch.stack([
            proj(feat) for proj, feat in zip(self.projects, features)
            ],dim = 1).sum(dim = 1)
        return fusion_features

class DownSampleBlock(nn.Module):
    def forward(self, x, h=None, w=None):
        vit_embeds = x
        
        if h is None or w is None:
            h = w = int(vit_embeds.shape[1] ** 0.5)
        else:
            assert h * w == vit_embeds.shape[1], \
                f"h({h})*w({w}) 必须等于输入序列长度 {vit_embeds.shape[1]}"
        
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.flat_square(vit_embeds)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        return vit_embeds

    def flat_square(self, x):
        n, w, h, c = x.size()
        if w % 2 == 1:
            x = torch.concat([x, torch.zeros((n, 1, h, c), dtype=x.dtype).to(x.device)], dim=1).contiguous()
            n, w, h, c = x.size()
        if h % 2 == 1:
            x = torch.concat([x, torch.zeros((n, w, 1, c), dtype=x.dtype).to(x.device)], dim=2).contiguous()
            n, w, h, c = x.size()
        x = x.view(n, w, int(h / 2), int(c * 2))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h / 2), int(w / 2), int(c * 4))
        return x

class MultimodalProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion = FusionBlock()
        self.downsample = DownSampleBlock()
        self.norm = nn.LayerNorm(4096)
        self.linear1 = nn.Linear(4096, 4096)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(4096, 4096)
        
        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                init.kaiming_normal_(
                    module.weight, 
                    mode='fan_in',          
                    nonlinearity='relu'    
                )
                if module.bias is not None:
                    init.zeros_(module.bias)

    def forward(self, x, h=None, w=None):
        x = self.fusion(x)
        x = self.downsample(x, h, w)
        x = self.norm(x)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x