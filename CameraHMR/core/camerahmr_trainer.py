import torch
import torch.nn.functional as F
from .utils.torch_compat import torch as _torch_compat  # registers safe globals on import
import pickle
from .utils.numpy_compat import ensure_numpy_legacy_aliases
ensure_numpy_legacy_aliases()
import smplx
import pytorch_lightning as pl
from typing import Dict
from yacs.config import CfgNode
from loguru import logger
import numpy as np

from .backbones import create_backbone
from .losses import (
    Keypoint3DLoss, Keypoint2DLoss, Keypoint2DLossScaled,
    ParameterLoss, VerticesLoss, TranslationLoss, PointToPlaneLoss
)
from .cam_model.fl_net import FLNet
from .smpl_wrapper import SMPL2 as SMPL, SMPLLayer
from .heads.smpl_head_cliff import build_smpl_head
from .heads.smpl_head_cliff_gendered import build_smpl_head_gendered
from .components import CrossAttentionNormalInjecter, FullyConnectedNormalInjecter, AdditionNormalInjecter
from .utils.train_utils import (
    trans_points2d_parallel, load_valid, perspective_projection,
    convert_to_full_img_cam
)
from .utils.eval_utils import pck_accuracy, reconstruction_error
from .utils.geometry import aa_to_rotmat
from .utils.pylogger import get_pylogger
from .utils.renderer_cam import render_image_group
from .constants import (
    NUM_JOINTS, H36M_TO_J14, CAM_MODEL_CKPT, DOWNSAMPLE_MAT,
    REGRESSOR_H36M, VITPOSE_BACKBONE, SMPL_MODEL_DIR, SMPLX2SMPL, SMPLX_MODEL_DIR, NUM_POSE_PARAMS, NUM_BETAS
)
from .losses import SMPLNormalLoss
# normals util
from .utils.smpl_utils import compute_normals_torch
# visualize images and normals
import cv2
from typing import List
# try:
#     from tools.vis import denorm_and_save_img, save_smpl
# except ImportError:
#     import os, sys
#     repo_root = os.path.dirname(os.path.dirname(__file__))  # .../CameraHMR
#     if repo_root not in sys.path:
#         sys.path.insert(0, repo_root)
#     from tools.vis import denorm_and_save_img, save_smpl

log = get_pylogger(__name__)

class CameraHMR(pl.LightningModule):
    """
    Pytorch Lightning Module for Camera Human Mesh Recovery (CameraHMR).
    This module integrates backbone feature extraction, camera modeling, SMPL fitting,
    and loss functions for training a 3D human mesh recovery pipeline.
    """

    def __init__(self, cfg: CfgNode):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=['init_renderer'])
        self.cfg = cfg

        # Backbone feature extractor
        self.backbone = create_backbone()
        if not self.cfg.MODEL.FINETUNE:
            self.backbone.load_state_dict(torch.load(VITPOSE_BACKBONE, map_location='cpu', weights_only=True)['state_dict'])

        # Camera model
        self.cam_model = FLNet()

        # SMPL Head (Neutral, Male, Female)
        self.enable_direct_gpu_process_gt_smpl_verts_and_normals = getattr(self.cfg.trainer, 'enable_direct_gpu_process_gt_smpl_verts_and_normals', True)
        # Optional: additional residual SMPL head (LoRA-like)
        self.use_lora_head = bool(getattr(self.cfg.MODEL, 'ADDITIONAL_LORA_SMPL_HEAD', False))
        if self.cfg.MODEL.SMPL_HEAD.TYPE == 'transformer_decoder_gendered':
            self.smpl_head = build_smpl_head_gendered()
            self.duplicate_gendered_heads = True
        else:
            self.smpl_head = build_smpl_head()
            self.duplicate_gendered_heads = False

        # FINETUNE MODE: 
        if cfg.MODEL.FINETUNE:
            # load pretrained CameraHMR weights (backbones, smpl_head)
            ckpt_path = getattr(cfg.paths, 'camerahmr_ckpt', None)
            if ckpt_path:
                self.load_pretrained(ckpt_path, subset_modules=None, strict=False, duplicate_gendered_heads=self.duplicate_gendered_heads,
                decoder_blocks = ("decpose", "decshape", "deccam", "deckp")) # Load ALL ['smpl_head']
            else:
                raise ValueError("No pretrained CameraHMR weights found. Check paths.camerahmr_ckpt.")
            
            # load cam_model weights 
            load_valid(self.cam_model, CAM_MODEL_CKPT)
        
        # Normal backbone feature extractor (for normal modality)
        self.normal_backbone = create_backbone()
        if not self.cfg.MODEL.FINETUNE:
            # load Pretrained Backbone weights
            self.normal_backbone.load_state_dict(torch.load(VITPOSE_BACKBONE, map_location='cpu', weights_only=True)['state_dict'])
        else:
            # duplicate from RGB modality
            self.normal_backbone.load_state_dict(self.backbone.state_dict())

        # Optional cross-attention normal injecter
        self.normal_injecter = None
        ni_cfg = getattr(self.cfg.MODEL, 'NORMAL_INJECTER', None)
        if ni_cfg is not None:
            # Support OmegaConf DictConfig / CfgNode / dict uniformly
            def _get(key, default=None):
                if isinstance(ni_cfg, dict):
                    return ni_cfg.get(key, default)
                return getattr(ni_cfg, key, default)

            ni_type = _get('TYPE', 'cross_attn')
            if ni_type == 'cross_attn':
                out_channels = _get('OUT_CHANNELS', 1280)
                num_heads = _get('NUM_HEADS', 8)
                dropout = _get('DROPOUT', 0.0)
                alpha = _get('WEIGHT', 1.0)
                # ViT returns (B, C=1280, H/16, W/16) by default
                self.normal_injecter = CrossAttentionNormalInjecter(in_channels=1280, out_channels=out_channels, num_heads=num_heads, dropout=dropout, alpha=alpha)
            elif ni_type == 'fully_connected':
                out_channels = _get('OUT_CHANNELS', 1280)
                hidden_channels = _get('HIDDEN_CHANNELS', None)
                dropout = _get('DROPOUT', 0.0)
                alpha = _get('WEIGHT', 1.0)
                self.normal_injecter = FullyConnectedNormalInjecter(in_channels=1280, out_channels=out_channels, hidden_channels=hidden_channels, dropout=dropout, alpha=alpha)
            elif ni_type == 'addition':
                alpha = _get('WEIGHT', 0.5)
                self.normal_injecter = AdditionNormalInjecter(alpha=alpha)

        # SMPL HEAD LORA DUPLICATE (BUT ZEROED DECODER LAYERS)
        self.smpl_head_lora = None
        if self.use_lora_head:
            log.info("Using lora head")
            # Freeze the base head
            for n, p in self.smpl_head.named_parameters():
                log.info(f"Freezing {n}")
                p.requires_grad = False

            # Create a residual head copy
            from copy import deepcopy
            self.smpl_head_lora = deepcopy(self.smpl_head)
            # Initialize residual head: encoder (transformer) random as usual; decoders zeroed
            def _init_module(m):
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif isinstance(m, torch.nn.Conv2d):
                    torch.nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
            # Zero only decoder layers of the residual head
            _init_module(self.smpl_head_lora.decpose)
            _init_module(self.smpl_head_lora.decshape)
            _init_module(self.smpl_head_lora.deccam)
            _init_module(self.smpl_head_lora.deckp)

        # TRAINING MODE: freeze backbones from training / finetune via config
        self.freeze_backbone = getattr(self.cfg.MODEL, 'FREEZE_BACKBONE', False)
        self.freeze_normal_backbone = getattr(self.cfg.MODEL, 'FREEZE_NORMAL_BACKBONE', False)
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        if self.freeze_normal_backbone:
            for param in self.normal_backbone.parameters():
                param.requires_grad = False
            self.normal_backbone.eval()

        # Loss functions
        loss_type = cfg.TRAIN.LOSS_TYPE
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type=loss_type)
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type=loss_type)
        self.keypoint_2d_loss_scaled = Keypoint2DLossScaled(loss_type=loss_type)
        self.trans_loss = TranslationLoss(loss_type=loss_type)
        # Vertices losses: keep L1/L2 and add point-to-plane
        self.vertices_loss_l1l2 = VerticesLoss(loss_type=loss_type)
        self.vertices_loss_p2p = PointToPlaneLoss()
        self.smpl_parameter_loss = ParameterLoss()
        self.smpl_normal_loss = SMPLNormalLoss()

        self.smpl = SMPL(SMPL_MODEL_DIR, gender='neutral')
        self.smpl_male = SMPL(SMPL_MODEL_DIR, gender='male')
        self.smpl_female = SMPL(SMPL_MODEL_DIR, gender='female')

        self.smpl = torch.compile(self.smpl)
        self.smpl_male = torch.compile(self.smpl_male)
        self.smpl_female = torch.compile(self.smpl_female)

        self.smpl_layer = SMPLLayer(SMPL_MODEL_DIR, gender='neutral')
        self.smpl_layer_male = SMPLLayer(SMPL_MODEL_DIR, gender='male')
        self.smpl_layer_female = SMPLLayer(SMPL_MODEL_DIR, gender='female')

        # Ground truth SMPL models
        self.smpl_gt = smplx.SMPL(SMPL_MODEL_DIR, gender='neutral').cuda()
        self.smpl_gt_male = smplx.SMPL(SMPL_MODEL_DIR, gender='male').cuda()
        self.smpl_gt_female = smplx.SMPL(SMPL_MODEL_DIR, gender='female').cuda()

        # TODO: SMPLX NOT IMPLEMENTED YET!!!
        self.smplx_gt = smplx.SMPLX(SMPLX_MODEL_DIR, gender='neutral').cuda()
        self.smplx_gt_male = smplx.SMPLX(SMPLX_MODEL_DIR, gender='male').cuda() 
        self.smplx_gt_female = smplx.SMPLX(SMPLX_MODEL_DIR, gender='female').cuda()

        self.gender_map = {0: 'male', 1: 'female', -1: 'neutral'}
        self.gendered_list = ['neutral', 'male', 'female']
        self.gendered_smpl_layers = [self.smpl_layer, self.smpl_layer_male, self.smpl_layer_female]

        self.smplx2smpl = pickle.load(open(SMPLX2SMPL, 'rb'))
        self.smplx2smpl = torch.tensor(self.smplx2smpl['matrix'][None], dtype=torch.float32)

        # Initialize ActNorm layers flag
        self.register_buffer('initialized', torch.tensor(False))

        # Disable automatic optimization for adversarial training
        self.automatic_optimization = False

        # Additional configurations
        self.J_regressor = torch.from_numpy(np.load(REGRESSOR_H36M))
        self.downsample_mat = pickle.load(open(DOWNSAMPLE_MAT, 'rb')).to_dense().cuda()

        # Store validation outputs
        self.validation_step_output = []

    def load_pretrained(self, ckpt_path: str, subset_modules: List[str] = None, strict: bool = False, decoder_blocks: List[str] = None, duplicate_gendered_heads: bool = False):
        try:
            state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            sd = state.get('state_dict', state)
            keep = {}
            for k, v in sd.items():
                if subset_modules is None:
                    allow = k.startswith("backbone.") or k.startswith("smpl_head.") or k.startswith("cam_model.")
                else:
                    allow = any(k.startswith(module) for module in subset_modules)
                if allow:
                    keep[k] = v
            if duplicate_gendered_heads:
                new_keep = {}
                for k, v in list(keep.items()):  # iterate over a snapshot
                    if k.startswith("smpl_head.") and any(k.startswith(f"smpl_head.{b}.") for b in decoder_blocks):
                        # e.g., k == "smpl_head.decpose.weight" -> base_name == "weight"
                        prefix, base_name = k.rsplit(".", 1)
                        for i in range(3):
                            new_keep[f"{prefix}.{i}.{base_name}"] = v
                        # skip copying the old single-head key to avoid unexpected keys
                    else:
                        new_keep[k] = v
                keep = new_keep
            missing, unexpected = self.load_state_dict(keep, strict=strict)
            log.info(f"Loaded pretrained from {ckpt_path}: missing={len(missing)} unexpected={len(unexpected)}")
            
        except Exception as e:
            log.warning(f"Failed to load pretrained from {ckpt_path}: {e}")

    def get_parameters(self):
        """Aggregate model parameters for optimization."""
        return list(self.smpl_head.parameters()) + list(self.backbone.parameters()) + list(self.normal_backbone.parameters())

    def configure_optimizers(self):
        """Configure optimizers with optional per-group LR and backbone freezing."""
        head_lr = getattr(self.cfg.TRAIN, 'HEAD_LR', self.cfg.TRAIN.LR)
        backbone_lr = getattr(self.cfg.TRAIN, 'BACKBONE_LR', self.cfg.TRAIN.LR)
        normal_backbone_lr = getattr(self.cfg.TRAIN, 'NORMAL_BACKBONE_LR', backbone_lr)

        param_groups = []

        head_params = []
        head_params.extend([p for p in self.smpl_head.parameters() if p.requires_grad])
        if self.smpl_head_lora is not None:
            head_params.extend([p for p in self.smpl_head_lora.parameters() if p.requires_grad])

        # Include normal_injecter (if present) with head lr by default
        if getattr(self, 'normal_injecter', None) is not None:
            head_params.extend([p for p in self.normal_injecter.parameters() if p.requires_grad])

        if len(head_params) > 0:
            param_groups.append({'params': head_params, 'lr': head_lr})

        if not getattr(self, 'freeze_backbone', False):
            bb_params = [p for p in self.backbone.parameters() if p.requires_grad]
            if len(bb_params) > 0:
                param_groups.append({'params': bb_params, 'lr': backbone_lr})

        if not getattr(self, 'freeze_normal_backbone', False):
            nbb_params = [p for p in self.normal_backbone.parameters() if p.requires_grad]
            if len(nbb_params) > 0:
                param_groups.append({'params': nbb_params, 'lr': normal_backbone_lr})

        optimizer = torch.optim.AdamW(
            params=param_groups,
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY
        )
        return optimizer

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        # Use RGB image as input
        x = batch['img']
        batch_size = x.shape[0]

        # Compute conditioning features using the backbones
        # if using ViT backbone, we need to use a different aspect ratio
        rgb_feats = self.backbone(x[:,:,:,32:-32])

        # Optional normal modality fusion
        if ('normal' in batch) and isinstance(batch['normal'], torch.Tensor):
            normal_input = batch['normal']
            normal_feats = self.normal_backbone(normal_input[:,:,:,32:-32])
            if self.normal_injecter is not None:
                conditioning_feats = self.normal_injecter(rgb_feats, normal_feats)
            else:
                # Fallback: element-wise sum
                conditioning_feats = rgb_feats + normal_feats
        else:
            conditioning_feats = rgb_feats

        # denormalize x and normal_input
        # norm_in = 255*(normal_input[0].detach().cpu().numpy()*0.5+0.5)

        # cv2.imwrite('normal_input.png', norm_in.transpose(1, 2, 0))
        # cv2.imwrite('rgb_feats.png', rgb_feats.detach().cpu().numpy().transpose(1, 2, 0))
        # cv2.imwrite('normal_feats.png', normal_feats.detach().cpu().numpy().transpose(1, 2, 0))
        # cv2.imwrite('conditioning_feats.png', conditioning_feats.detach().cpu().numpy().transpose(1, 2, 0))

        cx, cy = batch['box_center'][:, 0], batch['box_center'][:, 1]

        b = batch['box_size']
        img_h = batch['img_size'][:,0]
        img_w = batch['img_size'][:,1]
        if train:
            cam_intrinsics = batch['cam_int']
            fl_h = cam_intrinsics[:,0,0]
            vfov = (2 * torch.arctan((img_h)/(2*batch['cam_int'][:,0,0])))
            hfov = (2 * torch.arctan((img_w)/(2*batch['cam_int'][:,0,0])))
        else:
           cam_intrinsics = batch['cam_int']
           fl_h = cam_intrinsics[:,0,0]
           cam, features = self.cam_model(batch['img_full_resized'])
           vfov = cam[:, 1]
           fl_h = (img_h / (2 * torch.tan(vfov / 2)))
           cam_intrinsics[:,0,0]=fl_h
           cam_intrinsics[:,1,1]=fl_h


        # Original
        bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b],
                                dim=-1)
        bbox_info[:, :2] = bbox_info[:, :2] / cam_intrinsics[:, 0, 0].unsqueeze(-1)   # [-1, 1]
        bbox_info[:, 2] = (bbox_info[:, 2] / cam_intrinsics[:, 0, 0])  # [-1, 1]

        bbox_info = bbox_info.cuda().float()

        # Generate SMPL GT Parameters:
        if train:
            if self.enable_direct_gpu_process_gt_smpl_verts_and_normals:
                male_gender = batch['gender'] == 0
                female_gender = batch['gender'] == 1
                smpl_output_gt_male = self.smpl_male(**{k: v.float() for k,v in batch['smpl_params'].items()})
                smpl_output_gt_female = self.smpl_female(**{k: v.float() for k,v in batch['smpl_params'].items()})
                # combine smpl_output_gt_male and smpl_output_gt_female by gender
                # smpl_shape = self.smpl_gt.vertices.shape # 6890,3?

                smpl_output_gt_vertices = torch.empty(smpl_output_gt_male.vertices.shape, dtype=smpl_output_gt_male.vertices.dtype, device=smpl_output_gt_male.vertices.device)
                smpl_output_gt_vertices[male_gender] = smpl_output_gt_male.vertices[male_gender]
                smpl_output_gt_vertices[female_gender] = smpl_output_gt_female.vertices[female_gender]

                smpl_output_gt_joints = torch.empty(smpl_output_gt_male.joints.shape, dtype=smpl_output_gt_male.joints.dtype, device=smpl_output_gt_male.joints.device)
                smpl_output_gt_joints[male_gender] = smpl_output_gt_male.joints[:,:NUM_JOINTS][male_gender]
                smpl_output_gt_joints[female_gender] = smpl_output_gt_female.joints[:,:NUM_JOINTS][female_gender]

                faces_t = torch.as_tensor(self.smpl_gt.faces, dtype=torch.long, device=smpl_output_gt_vertices.device)
                smpl_output_gt_normals = compute_normals_torch(smpl_output_gt_vertices, faces_t)

                ones = torch.ones((batch_size, NUM_JOINTS, 1),device=self.device)
                batch['vertices'] = smpl_output_gt_vertices
                batch['smpl_normals'] = smpl_output_gt_normals
                batch['keypoints_3d'] = torch.cat((smpl_output_gt_joints, ones), dim=-1)
            # Calculated in DS get_item / DS init
            # else:
            #     batch['vertices'] = batch['vertices']
            #     batch['smpl_normals'] = batch['smpl_normals']
            #     batch['keypoints_3d'] = batch['keypoints_3d']
        if self.cfg.MODEL.SMPL_HEAD.TYPE == 'transformer_decoder_gendered':
            gendered_pred_smpl_params_list = self.smpl_head(rgb_feats, bbox_info=bbox_info)

            if self.use_lora_head and (self.smpl_head_lora is not None):
                gendered_pred_smpl_params_list_lora = self.smpl_head_lora(conditioning_feats, bbox_info=bbox_info)
            
            output_list = []
            # for each gender (0: neutral, 1: male, 2: female)
            for i in range(3):
                pred_smpl_params, pred_cam, decouts, pred_kp = gendered_pred_smpl_params_list[i]
                # if lora head is used
                if self.use_lora_head and (self.smpl_head_lora is not None):
                    _, _, decouts_resid, _ = gendered_pred_smpl_params_list_lora[i]
                    # Add residuals to the decoder outputs from base head
                    # Residuals are in 6D-body_pose, betas, cam, kp space before rotmat conversion
                    # Update pred_smpl_params by reconstructing from updated residuals
                    resid_body_pose6d = decouts['body_pose6d_residual'] + decouts_resid['body_pose6d_residual']
                    resid_betas = decouts['betas_residual'] + decouts_resid['betas_residual']
                    resid_cam = decouts['cam_residual'] + decouts_resid['cam_residual']
                    pred_kp = pred_kp + decouts_resid['kp_residual']
                    # Re-anchor on mean
                    batch_size = conditioning_feats.shape[0]

                    init_body_pose = self.smpl_head.init_body_pose.expand(batch_size, -1)
                    init_betas = self.smpl_head.init_betas.expand(batch_size, -1)
                    init_cam = self.smpl_head.init_cam.expand(batch_size, -1)

                    pred_body_pose6d = resid_body_pose6d + init_body_pose
                    pred_betas_vec = resid_betas + init_betas
                    pred_cam = resid_cam + init_cam
                    # Convert 6D to rotmats and build SMPL params to feed the smpl layer below
                    pred_body_pose_rotmats = aa_to_rotmat(pred_body_pose6d.view(-1, 3)).view(batch_size, 24, 3, 3)
                    pred_smpl_params = {
                        'global_orient': pred_body_pose_rotmats[:, [0]],
                        'body_pose': pred_body_pose_rotmats[:, 1:],
                        'betas': pred_betas_vec
                    }

                # Compute model vertices, joints and the projected joints
                pred_smpl_params['global_orient'] = pred_smpl_params['global_orient'].reshape(batch_size, -1, 3, 3)
                pred_smpl_params['body_pose'] = pred_smpl_params['body_pose'].reshape(batch_size, -1, 3, 3)
                pred_smpl_params['betas'] = pred_smpl_params['betas'].reshape(batch_size, -1)

                # ADDITIONALLY USE GENDER INFO TO PREDICT THE SMPL MODELS and BETTER USE FOR SMPL GT
                smpl_output = self.gendered_smpl_layers[i](**{k: v.float() for k,v in pred_smpl_params.items()}, pose2rot=False)

                pred_keypoints_3d = smpl_output.joints
                pred_vertices = smpl_output.vertices
                output = {}

                output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
                output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
                # Compute predicted SMPL vertex normals (B, V, 3)
                # faces_t = torch.as_tensor(self.smpl_gt.faces, dtype=torch.long, device=pred_vertices.device)
                # try:
                #     output['pred_vertex_normals'] = compute_normals_torch(output['pred_vertices'], faces_t)
                # except Exception:
                #     # Keep training robust if faces/normals computation fails for some reason
                #     logger.warning(f"Failed to compute predicted SMPL vertex normals for batch:{batch['imgname']}.")
                #     output['pred_vertex_normals'] = None

                # Store useful regression outputs to the output dict
                output['pred_cam'] = pred_cam
                output['pred_smpl_params'] = {k: v.clone() for k,v in pred_smpl_params.items()}

                # Compute camera translation
                device = pred_smpl_params['body_pose'].device
                dtype = pred_smpl_params['body_pose'].dtype

                cam_t = convert_to_full_img_cam(
                    pare_cam=output['pred_cam'],
                    bbox_height=batch['box_size'],
                    bbox_center=batch['box_center'],
                    img_w=img_w,
                    img_h=img_h,
                    focal_length=batch['cam_int'][:, 0, 0],
                )

                output['pred_cam_t'] = cam_t

                ## 2D Joints Projection
                joints2d = perspective_projection(
                    output['pred_keypoints_3d'],
                    rotation=torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
                    translation=cam_t,
                    cam_intrinsics=batch['cam_int'],
                )
                if self.cfg.LOSS_WEIGHTS['VERTS2D'] or self.cfg.LOSS_WEIGHTS['VERTS2D_CROP'] or self.cfg.LOSS_WEIGHTS['VERTS_2D_NORM']:
                    pred_verts_subsampled = self.downsample_mat.matmul(output['pred_vertices'])

                    pred_verts2d = perspective_projection(
                        pred_verts_subsampled,
                        rotation=torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
                        translation=cam_t,
                        cam_intrinsics=batch['cam_int'],
                    )
                    output['pred_verts2d'] = pred_verts2d
                output['pred_keypoints_2d'] = joints2d.reshape(batch_size, -1, 2)

                output_list.append(output)

        else:
            pred_smpl_params, pred_cam, decouts, pred_kp = self.smpl_head(rgb_feats, bbox_info=bbox_info)

            smpl_output = self.gendered_smpl_layers[0](**{k: v.float() for k,v in pred_smpl_params.items()}, pose2rot=False)

            pred_keypoints_3d = smpl_output.joints
            pred_vertices = smpl_output.vertices

            output = {}
            output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
            output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
            output['pred_cam'] = pred_cam
            output['pred_smpl_params'] = {k: v.clone() for k,v in pred_smpl_params.items()}
            output_list = [output]
            # import open3d as o3d
            # pc = o3d.geometry.PointCloud()
            # pc.points = o3d.utility.Vector3dVector(pred_vertices[1].detach().cpu().numpy())
            # o3d.io.write_point_cloud(f'pred_vertices.ply', pc)
            # pc = o3d.geometry.PointCloud()
            # pc.points = o3d.utility.Vector3dVector(batch['vertices'][1].detach().cpu().numpy())
            # o3d.io.write_point_cloud(f'gt_vertices.ply', pc)
        return output_list, fl_h

    def perspective_projection_vis(self, input_batch, output, max_save_img=1):
        import os
        import cv2


        translation = input_batch['translation'].detach()[:,:3]
        vertices = input_batch['vertices'].detach()
        for i in range(len(input_batch['imgname'])):
            cy, cx = input_batch['img_size'][i] // 2
            img_h, img_w = cy*2, cx*2
            imgname = input_batch['imgname'][i]
            save_filename = os.path.join('.', f'{self.global_step:08d}_{i:02d}_{os.path.basename(imgname)}')
            # focal_length_ = (img_w * img_w + img_h * img_h) ** 0.5  # Assumed fl

            focal_length_ = input_batch['cam_int'][i, 0, 0]
            focal_length = (focal_length_, focal_length_)

            rendered_img = render_image_group(
                image=cv2.imread(imgname),
                camera_translation=translation[i],
                vertices=vertices[i],
                focal_length=focal_length,
                camera_center=(cx, cy),
                camera_rotation=None,
                save_filename=save_filename,
                faces=self.smpl_gt.faces,
            )
            if i >= (max_save_img - 1):
                break

    def compute_gendered_loss(self, batch: Dict, gendered_output: List[Dict], train: bool = True) -> torch.Tensor:
        # Get annotations
        gt_keypoints_3d = batch['keypoints_3d']
        gt_smpl_params = batch['smpl_params']
        img_size = batch['img_size'].rot90().T.unsqueeze(1)
        # cv2.imwrite("img.png",batch['img_full_resized'][0].permute(1,2,0).detach().cpu().numpy()*255)
        total_loss = 0

        # apply gendered mask to the outputs
        batch_size = batch['gender'].shape[0]
        gendered_mask = torch.ones((3, batch_size), dtype=torch.bool, device=batch['gender'].device)
        # gendered_mask[0] all 1 for neutral
        gendered_mask[1] = batch['gender'] == 0 # male
        gendered_mask[2] = batch['gender'] == 1 # female

        for i, output in enumerate(gendered_output):
            gendered_name = self.gendered_list[i]
            loss = 0
            loss_mask = gendered_mask[i]
            pred_smpl_params = output['pred_smpl_params']
            pred_keypoints_2d = output['pred_keypoints_2d']
            pred_keypoints_3d = output['pred_keypoints_3d']
            pred_vertices = output['pred_vertices']

            # SMPL vertices / point losses
            loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, pelvis_id=25+14, loss_mask=loss_mask) # 331
            faces_t = torch.as_tensor(self.smpl_gt.faces, dtype=torch.long, device=pred_vertices.device)
            loss_vertices = self.vertices_loss_l1l2(pred_vertices, batch['vertices'], loss_mask=loss_mask) # 56795
            # vis pred_vertices[0]
            # import open3d as o3d
            # pc = o3d.geometry.PointCloud()
            # pc.points = o3d.utility.Vector3dVector(pred_vertices[0].detach().cpu().numpy())
            # o3d.io.write_point_cloud(f'pred_vertices_{i}.ply', pc)

            loss_vertices_pt2plane = self.vertices_loss_p2p(pred_vertices, batch['vertices'], faces_t, loss_mask=loss_mask) # 20062

            # Compute loss on SMPL parameters
            loss_smpl_params = {}
            for k, pred in pred_smpl_params.items():
                gt = gt_smpl_params[k].view(batch_size, -1)
                if 'beta' not in k:
                    gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size, -1, 3, 3)
                loss_smpl_params[k] = loss_mask * self.smpl_parameter_loss(pred.reshape(batch_size, -1), gt.reshape(batch_size, -1))

            # SMPL vertex normal cosine loss
            w_normals = self.cfg.LOSS_WEIGHTS.get('SMPL_NORMALS', 0.0)
            if w_normals and ('vertices' in batch):
                gt_vertices = batch['vertices']
                # imgnames = batch.get('imgname', None)
                loss_normals = loss_mask * self.smpl_normal_loss(pred_vertices, gt_vertices, faces_t) #, imgnames=imgnames

            loss = self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D'] * loss_keypoints_3d + \
                    sum([loss_smpl_params[k] * self.cfg.LOSS_WEIGHTS[k.upper()] for k in loss_smpl_params])+ \
                    self.cfg.LOSS_WEIGHTS['VERTICES'] * loss_vertices + \
                    self.cfg.LOSS_WEIGHTS.get('VERTICES_P2PL', 0.0) * loss_vertices_pt2plane + \
                    w_normals * loss_normals

            # crop / scaled / normed projected 2d vertices losses
            if self.cfg.LOSS_WEIGHTS['VERTS2D_CROP']:
                gt_verts2d = batch['proj_verts']
                pred_verts2d = output['pred_verts2d']
                pred_verts2d_cropped = trans_points2d_parallel(pred_verts2d, batch['_trans'])
                pred_verts2d_cropped = pred_verts2d_cropped/ self.cfg.MODEL.IMAGE_SIZE - 0.5
                gt_verts_2d_cropped = gt_verts2d.clone()
                gt_verts_2d_cropped[:,:,:2] = trans_points2d_parallel(gt_verts2d[:,:,:2], batch['_trans'])
                gt_verts_2d_cropped[:,:,:2] = gt_verts_2d_cropped[:,:,:2]/ self.cfg.MODEL.IMAGE_SIZE - 0.5

                loss_proj_vertices_cropped = self.keypoint_2d_loss(pred_verts2d_cropped, gt_verts_2d_cropped)
                loss += self.cfg.LOSS_WEIGHTS['VERTS2D_CROP'] * loss_proj_vertices_cropped

            if self.cfg.LOSS_WEIGHTS['VERTS2D']:
                gt_verts2d = batch['proj_verts'].clone()
                pred_verts2d = output['pred_verts2d'].clone()
                pred_verts2d[:, :, :2] = 2 * (pred_verts2d[:, :, :2] / img_size) - 1
                gt_verts2d[:, :, :2] = 2 * (gt_verts2d[:, :, :2] / img_size) - 1
                loss_proj_vertices = self.keypoint_2d_loss(pred_verts2d, gt_verts2d)
                loss += self.cfg.LOSS_WEIGHTS['VERTS2D'] * loss_proj_vertices

            if self.cfg.LOSS_WEIGHTS['VERTS_2D_NORM']:

                gt_verts2d = batch['proj_verts'].clone()
                pred_verts2d = output['pred_verts2d'].clone()

                pred_verts2d[:, :, :2] =  (pred_verts2d[:, :, :2] - pred_verts2d[:, [0], :2])/batch['box_size'].unsqueeze(-1).unsqueeze(-1)
                gt_verts2d[:, :, :2] =  (gt_verts2d[:, :, :2] - gt_verts2d[:, [0], :2])/batch['box_size'].unsqueeze(-1).unsqueeze(-1)
                loss_proj_vertices_norm = self.keypoint_2d_loss(pred_verts2d, gt_verts2d)
                loss += self.cfg.LOSS_WEIGHTS['VERTS_2D_NORM'] * loss_proj_vertices_norm

            # crop / scaled / normed projected 2D keypoint losses
            if self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D_CROP']:

                gt_keypoints_2d = batch['keypoints_2d'].clone()
                pred_keypoints_2d_cropped = trans_points2d_parallel(pred_keypoints_2d, batch['_trans'])
                pred_keypoints_2d_cropped = pred_keypoints_2d_cropped/ self.cfg.MODEL.IMAGE_SIZE - 0.5

                loss_keypoints_2d_cropped = self.keypoint_2d_loss(pred_keypoints_2d_cropped, gt_keypoints_2d)
                loss += self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D_CROP'] * loss_keypoints_2d_cropped

            if self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D']:
                pred_keypoints_2d_clone = pred_keypoints_2d.clone()
                pred_keypoints_2d_clone[:, :, :2] = 2 * (pred_keypoints_2d_clone[:, :, :2] / img_size) - 1
                gt_keypoints_2d = batch['orig_keypoints_2d']
                gt_keypoints_2d[:, :, :2] = 2 * (gt_keypoints_2d[:, :, :2] / img_size) - 1
                loss_keypoints_2d = self.keypoint_2d_loss_scaled(pred_keypoints_2d_clone, gt_keypoints_2d, batch['box_size'], img_size)
                loss += self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D'] * loss_keypoints_2d

            if self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D_NORM']:
                # This loss would neek full kp loss or crop kp loss to anchor the root joint
                pred_keypoints_2d_clone = pred_keypoints_2d.clone()
                pred_keypoints_2d_clone[:, :, :2] =  (pred_keypoints_2d_clone[:, :, :2] - pred_keypoints_2d_clone[:, [0], :2])/batch['box_size'].unsqueeze(-1).unsqueeze(-1)
                gt_keypoints_2d = batch['orig_keypoints_2d'].clone()
                gt_keypoints_2d[:, :, :2] =  (gt_keypoints_2d[:, :, :2] - gt_keypoints_2d[:, [0], :2])/batch['box_size'].unsqueeze(-1).unsqueeze(-1)
                loss_keypoints_2d_norm = self.keypoint_2d_loss(pred_keypoints_2d_clone, gt_keypoints_2d)
                loss += self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D_NORM'] * loss_keypoints_2d_norm

            # camera translation loss
            if self.cfg.LOSS_WEIGHTS['TRANS_LOSS']:

                gt_trans = batch['translation'][:,:3]
                pred_trans = output['pred_cam_t']
                loss_trans = self.trans_loss(pred_trans, gt_trans)
                loss += self.cfg.LOSS_WEIGHTS['TRANS_LOSS'] * loss_trans

            total_loss += loss

            losses = dict(loss=loss.detach(),
                        loss_keypoints_3d=loss_keypoints_3d.detach(),
                        loss_vertices=loss_vertices.detach(),
                        loss_vertices_pt2plane=loss_vertices_pt2plane.detach(),
                        loss_smpl_normals=loss_normals.detach(),
                        loss_kp2d_cropped=loss_keypoints_2d_cropped.detach())

            for k, v in loss_smpl_params.items():
                losses['loss_' + k] = v.detach()

            output['losses_{gendered_name}'] = losses

        return total_loss


    def forward(self, batch: Dict) -> Dict:
        return self.forward_step(batch, train=False)


    def training_step(self, joint_batch: Dict, batch_idx: int) -> Dict:

        batch = joint_batch['img']
        # mocap_batch = joint_batch['mocap']
        optimizer = self.optimizers(use_pl_optimizer=True)
        # if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
        batch_size = batch['img'].shape[0]

        gendered_output,_ = self.forward_step(batch, train=True)

        loss = self.compute_gendered_loss(batch, gendered_output, train=True)
        # Error if Nan
        if torch.isnan(loss):
            print('nan',batch['imgname'])
            for gendered_name, output in gendered_output:
                for k,v in output['losses_{gendered_name}'].items():
                    print('nan',gendered_name,k,v)

        optimizer.zero_grad()
        self.manual_backward(loss)
        # Clip gradient
        if self.cfg.TRAIN.get('GRAD_CLIP_VAL', 0) > 0:
            gn = torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.cfg.TRAIN.GRAD_CLIP_VAL, error_if_nonfinite=True)
            self.log('train/grad_norm', gn, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        optimizer.step()

        # build a single dict with all gendered losses, then log
        metric_dict = {
            f"train/loss_{name}": output[f"losses_{name}"]["loss"]
            for name in self.gendered_list  # e.g., ["neutral", "male", "female"]
        }
        self.log_dict(metric_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return gendered_output



    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx=0) -> Dict:

        batch_size = batch['img'].shape[0]
        gendered_output,_ = self.forward_step(batch, train=False)
        dataset_names = batch['dataset']

        joint_mapper_h36m = H36M_TO_J14
        J_regressor_batch_smpl = self.J_regressor[None, :].expand(batch['img'].shape[0], -1, -1).float().cuda()


        if '3dpw' in dataset_names[0]:
            # For 3dpw vertices are generated in dataset.py because gender is needed
            gt_cam_vertices = batch['vertices']
            # Get 14 predicted joints from the mesh
            gt_keypoints_3d = torch.matmul(J_regressor_batch_smpl, gt_cam_vertices)
            gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
            gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
            gt_cam_vertices = gt_cam_vertices - gt_pelvis
            # Convert predicted vertices to SMPL Fromat
            # Get 14 predicted joints from the mesh
            pred_cam_vertices = gendered_output[0]['pred_vertices']

            pred_keypoints_3d = torch.matmul(J_regressor_batch_smpl, pred_cam_vertices)
            pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
            pred_cam_vertices = pred_cam_vertices - pred_pelvis
            r_error, _ = reconstruction_error(
                pred_keypoints_3d.float().cpu().numpy(),
                gt_keypoints_3d.float().cpu().numpy(),
                reduction=None
            )

        elif 'emdb' in dataset_names[0]:
            gt_cam_vertices = batch['vertices']
            gt_keypoints_3d = torch.matmul(self.smpl.J_regressor, gt_cam_vertices)
            gt_pelvis = (gt_keypoints_3d[:, [1], :] + gt_keypoints_3d[:, [2], :]) / 2.0
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
            gt_cam_vertices = gt_cam_vertices - gt_pelvis
            pred_cam_vertices = gendered_output[0]['pred_vertices']

            pred_keypoints_3d = torch.matmul(self.smpl.J_regressor, pred_cam_vertices)
            pred_pelvis = (pred_keypoints_3d[:, [1], :] + pred_keypoints_3d[:, [2], :]) / 2.0
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
            pred_cam_vertices = pred_cam_vertices - pred_pelvis
            # Reconstuction_error (PA-MPJPE)
            r_error, _ = reconstruction_error(
                pred_keypoints_3d.float().cpu().numpy(),
                gt_keypoints_3d.float().cpu().numpy(),
                reduction=None
            )

        elif 'rich' in dataset_names[0]:
            gt_cam_vertices = batch['vertices']
            gt_keypoints_3d = torch.matmul(self.smpl.J_regressor, gt_cam_vertices)
            gt_pelvis = (gt_keypoints_3d[:, [1], :] + gt_keypoints_3d[:, [2], :]) / 2.0
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
            gt_cam_vertices = gt_cam_vertices - gt_pelvis
            pred_cam_vertices = gendered_output[0]['pred_vertices']
            pred_keypoints_3d = torch.matmul(self.smpl.J_regressor, pred_cam_vertices)
            pred_pelvis = (pred_keypoints_3d[:, [1], :] + pred_keypoints_3d[:, [2], :]) / 2.0
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
            pred_cam_vertices = pred_cam_vertices - pred_pelvis
            r_error, _ = reconstruction_error(
                pred_keypoints_3d.float().cpu().numpy(),
                gt_keypoints_3d.float().cpu().numpy(),
                reduction=None
            )

        else:
            smpl_output_gt = self.smpl_gt(**{k: v.float() for k,v in batch['smpl_params'].items()})
            male_indices = (batch['gender'] == 0)  # Assuming 0 represents males
            female_indices = ~male_indices
            male_batch = {k: v[male_indices] for k, v in batch['smpl_params'].items()}
            female_batch = {k: v[female_indices] for k, v in batch['smpl_params'].items()}

            # Create an empty tensor with the same shape as the original batch
            output_shape = (batch['gender'].shape[0], 6890, 3)  # Assuming the output shape is the same for both models
            smpl_output_gt = torch.empty(output_shape, dtype=self.smpl_gt().vertices.dtype, device=batch['gender'].device)

        # Apply the smpl_gt_male and smpl_gt_female models
            if male_indices.any():
                smpl_output_gt[male_indices] = self.smpl_gt_male(**male_batch).vertices
            if female_indices.any():
                smpl_output_gt[female_indices] = self.smpl_gt_female(**female_batch).vertices

            gt_cam_vertices = smpl_output_gt
            pred_cam_vertices = gendered_output[0]['pred_vertices']

            gt_keypoints_3d = torch.matmul(J_regressor_batch_smpl, gt_cam_vertices)
            pred_keypoints_3d = torch.matmul(J_regressor_batch_smpl, pred_cam_vertices)
            gt_pelvis = (gt_keypoints_3d[:, [1], :] + gt_keypoints_3d[:, [2], :]) / 2.0
            pred_pelvis = (pred_keypoints_3d[:, [1], :] + pred_keypoints_3d[:, [2], :]) / 2.0

            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
            pred_cam_vertices = pred_cam_vertices - pred_pelvis
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
            gt_cam_vertices = gt_cam_vertices - gt_pelvis
            r_error, _ = reconstruction_error(
                pred_keypoints_3d.float().cpu().numpy(),
                gt_keypoints_3d.float().cpu().numpy(),
                reduction=None
            )

        img_h = batch['img_size'][:,0]
        img_w = batch['img_size'][:,1]
        device = gendered_output[0]['pred_cam'].device
        cam_t = convert_to_full_img_cam(
            pare_cam=gendered_output[0]['pred_cam'],
            bbox_height=batch['box_size'],
            bbox_center=batch['box_center'],
            img_w=img_w,
            img_h=img_h,
            focal_length=batch['cam_int'][:, 0, 0],
        )

        joints2d = perspective_projection(
            gendered_output[0]['pred_keypoints_3d'],
            rotation=torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
            translation=cam_t,
            cam_intrinsics=batch['cam_int'],
        )

        if batch['keypoints_2d'].shape[1]>=17:
            pred_kp = trans_points2d_parallel(joints2d, batch['_trans'])
            pred_kp = pred_kp / self.cfg.MODEL.IMAGE_SIZE - 0.5
            gt_kp = batch['keypoints_2d']
            mask = gt_kp[:,:,2]>0
            zeros_to_insert = torch.zeros((gt_kp.shape[0], 1, 3)).cuda()
            if '3dpw' in dataset_names[0]:
                gt_kp = torch.cat((gt_kp[:, :9, :], zeros_to_insert, gt_kp[:, 9:, :]), dim=1)
                pck1, avgpck1, _ = (pck_accuracy(pred_kp[:,:18,:2],gt_kp[:,:18,:2],mask[:,:18],0.05))
                pck2, avgpck2, _ = (pck_accuracy(pred_kp[:,:18,:2],gt_kp[:,:18,:2],mask[:,:18],0.1))
            else:
                pck1, avgpck1, _ = (pck_accuracy(pred_kp[:,:18,:2],gt_kp[:,:18,:2],mask[:,:18],0.05))
                pck2, avgpck2, _ = (pck_accuracy(pred_kp[:,:18,:2],gt_kp[:,:18,:2],mask[:,:18],0.1))

        else:
            pck1 = torch.zeros(joints2d.shape)
            pck2 = torch.zeros(joints2d.shape)

        # Absolute error (MPJPE)
        error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1))
        error_verts = torch.sqrt(((pred_cam_vertices - gt_cam_vertices) ** 2).sum(dim=-1))

        # error_trans = torch.sqrt(((cam_t.unsqueeze(1) - batch['cam_trans']) ** 2).sum(dim=-1))
        # val_mrpe = error_trans.mean(-1)
        val_mpjpe = error.mean(-1)*1000
        val_pve = error_verts.mean(-1)*1000
        # Ensure tensor is on the same device for DDP reduction (NCCL requires CUDA tensors)
        val_pampjpe = torch.as_tensor(r_error.mean(-1), device=self.device) * 1000

        avgpck_005 = pck1
        avgpck_01 = pck2
        if 'coco' in dataset_names[0]:
            self.log('avgpck_0.05',avgpck_005.mean(), logger=True, sync_dist=True)
            self.log('avgpck_0.1',avgpck_01.mean(), logger=True, sync_dist=True)
        else:
            self.log('val_pve',val_pve.mean(), logger=True, sync_dist=True)
            # self.log('val_trans',val_mrpe.mean(), logger=True, sync_dist=True)
            self.log('val_mpjpe',val_mpjpe.mean(), logger=True, sync_dist=True)
            self.log('val_pampjpe',val_pampjpe.mean(), logger=True, sync_dist=True)

        self.validation_step_output.append({'val_loss': val_pve ,'val_loss_mpjpe': val_mpjpe, 'val_loss_pampjpe':val_pampjpe,  'avgpck_0.05':avgpck_005, 'avgpck_0.1':avgpck_01, 'dataloader_idx': dataloader_idx})

    def on_validation_epoch_end(self, dataloader_idx=0):
        # Flatten outputs if it's a list of lists
        outputs = self.validation_step_output
        if outputs and isinstance(outputs[0], list):
            outputs = [item for sublist in outputs for item in sublist]
        val_dataset = self.cfg.DATASETS.VAL_DATASETS.split('_')
        # Proceed with the assumption outputs is a list of dictionaries
        for dataloader_idx in range(len(val_dataset)):
            dataloader_outputs = [x for x in outputs if x.get('dataloader_idx') == dataloader_idx]
            if dataloader_outputs:  # Ensure there are outputs for this dataloader
                avg_val_loss = torch.stack([x['val_loss'] for x in dataloader_outputs]).mean()
                avg_mpjpe_loss = torch.stack([x['val_loss_mpjpe'] for x in dataloader_outputs]).mean()
                avg_pampjpe_loss = torch.stack([x['val_loss_pampjpe'] for x in dataloader_outputs]).mean()

                avg_pck_005_loss = torch.stack([x['avgpck_0.05'] for x in dataloader_outputs]).mean()
                avg_pck_01_loss = torch.stack([x['avgpck_0.1'] for x in dataloader_outputs]).mean()

                # avg_mrpe_loss = torch.stack([x['val_trans'] for x in dataloader_outputs]).mean()*1000
                logger.info('PA-MPJPE: '+str(dataloader_idx)+str(avg_pampjpe_loss))
                logger.info('MPJPE: '+str(dataloader_idx)+str(avg_mpjpe_loss))
                logger.info('PVE: '+str(dataloader_idx)+ str(avg_val_loss))
                logger.info('avgpck_0.05: '+str(dataloader_idx)+str(avg_pck_005_loss))
                logger.info('avgpck_0.1: '+str(dataloader_idx)+str(avg_pck_01_loss))
            if dataloader_idx==0:
                self.log('val_loss',avg_val_loss, logger=True, sync_dist=True)


    def test_step(self, batch: Dict, batch_idx: int, dataloader_idx=0) -> Dict:
        return self.validation_step(batch, batch_idx, dataloader_idx)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()
