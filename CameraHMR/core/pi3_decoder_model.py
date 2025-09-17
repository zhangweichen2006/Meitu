import torch
from .backbones.pi3 import Pi3

class Pi3Model():
    def __init__(self, ckpt, device):
        self.model = Pi3().to(device).eval()
        if ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(ckpt)
        else:
            weight = torch.load(ckpt, map_location=device, weights_only=False)

        self.model.load_state_dict(weight)

    def forward(self, imgs):
        try:
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        except Exception:
            dtype = torch.float16
        with torch.amp.autocast('cuda', dtype=dtype):
            res = self.model(imgs) #[None]

        return res