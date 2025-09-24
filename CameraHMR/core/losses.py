import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.smpl_utils import compute_normals_torch

class Keypoint2DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        super(Keypoint2DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_2d: torch.Tensor, gt_keypoints_2d: torch.Tensor) -> torch.Tensor:
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        batch_size = conf.shape[0]
        loss = (conf * self.loss_fn(pred_keypoints_2d.float(), gt_keypoints_2d[:, :, :-1])).sum(dim=(1,2))
        return loss.sum()


class Keypoint2DLossScaled(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        super(Keypoint2DLossScaled, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_2d: torch.Tensor, gt_keypoints_2d: torch.Tensor, box_size, img_size) -> torch.Tensor:
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        batch_size = conf.shape[0]

        loss = (conf * self.loss_fn(pred_keypoints_2d.float(), gt_keypoints_2d[:, :, :-1]))

        loss_scale = (img_size.squeeze(1)/box_size.unsqueeze(-1)).mean(1)
        loss = (loss*loss_scale.unsqueeze(-1).unsqueeze(-1)).sum(dim=(1,2))

        return loss.sum()

class VerticesLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        super(VerticesLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_vertices: torch.Tensor, gt_vertices: torch.Tensor):
        batch_size = pred_vertices.shape[0]
        gt_vertices = gt_vertices.clone()
        loss = (self.loss_fn(pred_vertices, gt_vertices)).sum(dim=(1,2))
        return loss.sum()

class Keypoint3DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        super(Keypoint3DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_3d: torch.Tensor, gt_keypoints_3d: torch.Tensor, pelvis_id: int = 39):
        batch_size = pred_keypoints_3d.shape[0]
        gt_keypoints_3d = gt_keypoints_3d.clone()
        pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, pelvis_id, :].unsqueeze(dim=1)
        gt_keypoints_3d[:, :, :-1] = gt_keypoints_3d[:, :, :-1] - gt_keypoints_3d[:, pelvis_id, :-1].unsqueeze(dim=1)
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1]
        loss = (conf * self.loss_fn(pred_keypoints_3d.float(), gt_keypoints_3d)).sum(dim=(1,2))
        return loss.sum()

class ParameterLoss(nn.Module):

    def __init__(self):
        super(ParameterLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, pred_param: torch.Tensor, gt_param: torch.Tensor):
        batch_size = pred_param.shape[0]
        num_dims = len(pred_param.shape)
        mask_dimension = [batch_size] + [1] * (num_dims-1)
        loss_param = self.loss_fn(pred_param.float(), gt_param)
        return loss_param.sum()

class TranslationLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):

        super(TranslationLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')


    def forward(self, pred_trans: torch.Tensor, gt_trans: torch.Tensor):

        loss = self.loss_fn(pred_trans, gt_trans)
        return loss.sum()

class SMPLNormalLoss(nn.Module):

    def __init__(self, loss_type: str = 'cos'):
        super(SMPLNormalLoss, self).__init__()
        if loss_type == 'cos':
            self.loss_fn = nn.CosineEmbeddingLoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_vertices: torch.Tensor, gt_vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """
        Compute per-vertex normal loss between predicted and GT SMPL meshes.

        Args:
            pred_vertices: (B, V, 3) predicted mesh vertices.
            gt_vertices: (B, V, 3) ground-truth mesh vertices.
            faces: (F, 3) long tensor of shared topology indices.

        Returns:
            Scalar loss (summed over batch and vertices), cosine-based: 1 - cos(theta).
        """
        # Ensure tensors are float and on same device
        device = pred_vertices.device
        gt_vertices = gt_vertices.to(device=device)
        faces = faces.to(device=device, dtype=torch.long)

        # Compute area-weighted vertex normals, normalized per-vertex
        pred_normals = compute_normals_torch(pred_vertices, faces)   # (B, V, 3)
        gt_normals = compute_normals_torch(gt_vertices, faces)       # (B, V, 3)

        # Cosine similarity per vertex (orientation-agnostic via absolute value)
        cos_sim = F.cosine_similarity(pred_normals, gt_normals, dim=-1)  # (B, V)
        per_vertex_loss = 1.0 - cos_sim.abs() # TODO: orientation-agnostic, check later

        # Average over vertices, then sum over batch (match style of other losses)
        loss = per_vertex_loss.mean(dim=1).sum()
        return loss