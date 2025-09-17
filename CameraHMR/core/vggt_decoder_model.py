import torch
from .backbones.vggt import VGGT

class VGGTModel():
    def __init__(self, ckpt, device):
        self.model = VGGT().to(device).eval()
        if ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(ckpt)
        else:
            weight = torch.load(ckpt, map_location=device, weights_only=False)
        self.model.load_state_dict(weight)

    def forward(self, imgs):
        res = self.model(imgs)
        return res