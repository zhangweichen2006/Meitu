# Copyright (c) 2023, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from transformers import Dinov2Backbone
from torchvision import transforms
from einops import rearrange

from torch import Tensor

def pretrain_forward(sp_lite, inputs: Tensor, layer_num: int, return_hidden_states=False) -> Tensor:
    B = inputs.size(0)
    patch_embed_output, _50, _51, _52, _53 = sp_lite.backbone.patch_embed(inputs)
    cls_token = sp_lite.backbone.cls_token.expand(B, -1, -1)
    x = torch.cat([cls_token, patch_embed_output], dim=1)

    cls_pos_embed, patch_pos_embed = sp_lite.backbone.pos_embed[:, 0:1, :], sp_lite.backbone.pos_embed[:, 1:, :]
    
    dim = cls_pos_embed.shape[-1]
    #64x64
    patch_pos_embed = patch_pos_embed.reshape(-1, 64, 64, dim)
    patch_pos_embed_ = patch_pos_embed.permute(0, 3, 1, 2)
    patch_pos_embed = torch.nn.functional.interpolate(
        patch_pos_embed_,
        size = (_52, _53),
        mode="bicubic",
        align_corners=False,
    )

    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(-1, _52 * _53, dim)
    patch_pos_embed =  torch.cat([cls_pos_embed, patch_pos_embed], dim=1)
    
    x = x + patch_pos_embed
    x = sp_lite.backbone.drop_after_pos(x)
    if return_hidden_states:
        hidden_states = []
        hidden_states.append(x)
    for i in range(layer_num):
        x = getattr(sp_lite.backbone.layers, str(i))(x)
        hidden_states.append(x)

    x = sp_lite.backbone.ln1(x)
    
    cls_output = x[:, 0]  # Assuming class token is at index 0
    patch_tokens = x[:, 1:]  # Remaining are patch tokens

    output = patch_tokens.view(B, _52, _53, -1).permute(0, 3, 1, 2)
    if return_hidden_states:
        return output, hidden_states
    return output
class SapiensWrapper_ts(nn.Module):

    """
    Sapiens wrapper using huggingface transformer implementation.
    """
    def __init__(self,
                 model_path: str = 'facebook/dinov2-base',
                 freeze=True,
                 img_size=None,
                 layer_num=None):
        super().__init__()
        if layer_num == None:
            if "0.3b" in model_path:
                self.layer_num = 24
            else:
                self.layer_num = 48
        else:
            self.layer_num = layer_num
        self.model = torch.jit.load(model_path)
        if img_size is None:
            self.my_processor = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        else:
            self.my_processor = transforms.Compose([
                transforms.Resize(size=img_size),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        self.interpolate_pos_encoding=True
        if freeze:
            self._freeze()
    def forward(self, image, use_my_proces=False, requires_grad=False, output_hidden_states=False):
        # image: [B, N, C, H, W]
        # RGB image with [0,1] scale and properly sized
        if image.ndim == 5:
            B, N, _, H, W = image.shape
            mv = True
            image = image.flatten(0, 1)
        if image.ndim == 4:
            N, _, H, W = image.shape
            B = None
        else:

            raise NotImplementedError
        device = image.device
        if not use_my_proces:
            inputs = self.image_processor(image, return_tensors="pt") 
            inputs['pixel_values'] = inputs['pixel_values'].to(device) 
        else:
            inputs = self.my_processor(image)
            inputs = {'pixel_values': inputs}
           

        if requires_grad==False:
            with torch.no_grad():
                outputs = pretrain_forward(self.model, inputs['pixel_values'], layer_num=self.layer_num, return_hidden_states=output_hidden_states)
        else:
            outputs = pretrain_forward(self.model, inputs['pixel_values'], layer_num=self.layer_num, return_hidden_states=output_hidden_states)
        last_feature_map = outputs[0]


        if not output_hidden_states:
            if B is  None: # dim = 5 
                last_feature_map = rearrange(last_feature_map, 'n dim h w -> n (h w) dim') # N, N_tk, C
            else:
                last_feature_map = rearrange(last_feature_map, 'bn  dim h w -> bn (h w) dim')
                last_feature_map = last_feature_map.reshape(B, N, last_feature_map.shape[-2], last_feature_map.shape[-1])

        if output_hidden_states:
            hidden_states = torch.stack(outputs[1], 0).permute(1, 0, 2, 3) # N, N_layer, N_tk, C
            hidden_states = hidden_states[:, :, 1:,:]  # N, N_layer, N_tk, C
        if output_hidden_states:
            return hidden_states
        else:
            return last_feature_map

    def _freeze(self):
        print(f"======== Freezing DinoWrapper ========")
        self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False

if __name__ == "__main__":
    model = SapiensWrapper_ts()
    model.eval()
    image = torch.rand(1, 3, 896, 640)
    output = model(image, use_my_proces=True,  output_hidden_states=True)
    output =  pretrain_forward(model, image, layer_num=24)
    print(output)
    print("done")