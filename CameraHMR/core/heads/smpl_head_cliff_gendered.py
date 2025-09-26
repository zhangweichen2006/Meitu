import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

from ..utils.geometry import rot6d_to_rotmat, aa_to_rotmat
from ..components.pose_transformer import TransformerDecoder
from ..constants import TRANSFORMER_DECODER, SMPL_MEAN_PARAMS_FILE, NUM_BETAS, NUM_POSE_PARAMS

def build_smpl_head_gendered():
    return GenderedSMPLTransformerDecoderHead()


class GenderedSMPLTransformerDecoderHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.joint_rep_dim = 6
        npose = self.joint_rep_dim * (NUM_POSE_PARAMS + 1)
        self.npose = npose
        transformer_args = dict(
            num_tokens=1,
            token_dim=(3 + npose + NUM_BETAS + 3),
            dim=1024,
        )
        transformer_args = (transformer_args | dict(TRANSFORMER_DECODER))
        self.transformer = TransformerDecoder(
            **transformer_args
        )
        dim=transformer_args['dim']

        # shared transformer but different fc decoders
        self.decpose = nn.ModuleList([nn.Linear(dim, npose) for _ in range(3)])
        self.decshape = nn.ModuleList([nn.Linear(dim, 10) for _ in range(3)])
        self.deccam = nn.ModuleList([nn.Linear(dim, 3) for _ in range(3)])
        self.deckp = nn.ModuleList([nn.Linear(dim, 88) for _ in range(3)])  

        mean_params = np.load(SMPL_MEAN_PARAMS_FILE)
        init_body_pose = torch.from_numpy(mean_params['pose'].astype(np.float32)).unsqueeze(0)
        init_betas = torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32)).unsqueeze(0)
        self.register_buffer('init_body_pose', init_body_pose)
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x, bbox_info, **kwargs):
        batch_size = x.shape[0]
        # vit pretrained backbone is channel-first. Change to token-first
        x = einops.rearrange(x, 'b c h w -> b (h w) c')

        init_body_pose = self.init_body_pose.expand(batch_size, -1)
        init_betas = self.init_betas.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)

        pred_body_pose = init_body_pose
        pred_betas = init_betas
        pred_cam = init_cam
        
        gendered_pred_smpl_params_list = []
        token = torch.cat([bbox_info, pred_body_pose, pred_betas, pred_cam], dim=1)[:,None,:]

        # Pass through transformer
        token_out = self.transformer(token, context=x)
        token_out = token_out.squeeze(1) # (B, C)

        for i in range(3):
            pred_body_pose_list = []
            pred_betas_list = []
            pred_cam_list = []
            # Decoder raw residuals
            
            resid_body_pose6d = self.decpose[i](token_out)
            resid_betas = self.decshape[i](token_out)
            resid_cam = self.deccam[i](token_out)
            resid_kp = self.deckp[i](token_out)

            # Readout from token_out (absolute, anchored at init means)
            pred_body_pose = resid_body_pose6d + init_body_pose.clone() # 1* 144 (6D*24)
            pred_betas = resid_betas + init_betas.clone()
            pred_cam = resid_cam + init_cam.clone()
            pred_kp = resid_kp
            pred_body_pose_list.append(pred_body_pose)
            pred_betas_list.append(pred_betas)
            pred_cam_list.append(pred_cam)

            joint_conversion_fn = rot6d_to_rotmat

            pred_smpl_params_list = {}
            pred_smpl_params_list['body_pose'] = torch.cat([joint_conversion_fn(pbp).view(batch_size, -1, 3, 3)[:, 1:, :, :] for pbp in pred_body_pose_list], dim=0) # 1*23*3*3, remove root
            pred_smpl_params_list['betas'] = torch.cat(pred_betas_list, dim=0) # 1*10
            pred_smpl_params_list['cam'] = torch.cat(pred_cam_list, dim=0) # 1*3
            # Extra tensors for residual-based fusion
            pred_smpl_params_list['body_pose6d'] = pred_body_pose.clone() # (B, 6*(NUM_POSE_PARAMS+1))
            pred_smpl_params_list['body_pose6d_residual'] = resid_body_pose6d.clone()
            pred_smpl_params_list['body_pose6d_residual'] = resid_body_pose6d.clone()
            pred_smpl_params_list['betas_residual'] = resid_betas.clone()
            pred_smpl_params_list['cam_residual'] = resid_cam.clone()
            pred_smpl_params_list['kp_residual'] = resid_kp.clone()
            pred_body_pose = joint_conversion_fn(pred_body_pose).view(batch_size, 24, 3, 3) # 1*24*3*3

            pred_smpl_params = {'global_orient': pred_body_pose[:, [0]], #root
                                'body_pose': pred_body_pose[:, 1:], # 23 joints
                                'betas': pred_betas}

            gendered_ret = pred_smpl_params, pred_cam, pred_smpl_params_list, pred_kp
            gendered_pred_smpl_params_list.append(gendered_ret)

        return gendered_pred_smpl_params_list
