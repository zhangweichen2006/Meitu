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

def compute_vertex_normals(verts: torch.Tensor,
                           faces: torch.Tensor,
                           eps: float = 1e-6) -> torch.Tensor:
    """
    verts: (B, V, 3)
    faces: (F, 3) long, shared across batch
    return: (B, V, 3) unit vertex normals
    """
    # Gather triangle corners
    v0 = verts[:, faces[:, 0], :]  # (B, F, 3)
    v1 = verts[:, faces[:, 1], :]
    v2 = verts[:, faces[:, 2], :]
    # Face normals (unit)
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)                # (B, F, 3)
    fn = fn / (fn.norm(dim=-1, keepdim=True) + eps)

    # Accumulate to vertices (vectorized; works well on GPU)
    B, V, _ = verts.shape
    normals = torch.zeros_like(verts)
    normals[:, faces[:, 0], :] += fn
    normals[:, faces[:, 1], :] += fn
    normals[:, faces[:, 2], :] += fn

    # Normalize vertex normals and clip components for numerical safety
    normals = normals / (normals.norm(dim=-1, keepdim=True) + eps)
    # normals = torch.clamp(normals, min=-1.0, max=1.0) # TODO: Need to check
    return normals


class PointToPlaneLoss(nn.Module):
    """
    Fast point-to-plane between *corresponding* vertices of two SMPL meshes
    (same topology). Uses GT vertex normals:
      loss_i = | (p_i - v_i) · n_i_gt |
    """
    def __init__(self, reduction: str = "mean", detach_gt: bool = True, eps: float = 1e-6):
        super().__init__()
        assert reduction in ("none", "mean", "sum")
        self.reduction = reduction
        self.detach_gt = detach_gt
        self.eps = eps

    # @torch.amp.autocast(enabled=False)  # TODO: keep normals in FP32 for stability
    def forward(self,
                pred_vertices: torch.Tensor,  # (B, V, 3), float
                gt_vertices: torch.Tensor,    # (B, V, 3), float
                faces: torch.Tensor,          # (F, 3), long
                gt_normals: torch.Tensor = None,  # optional (B, V, 3) unit normals
                mask: torch.Tensor = None,        # optional (B, V) boolean/float
                weights: torch.Tensor = None      # optional (B, V) float
                ) -> torch.Tensor:

        # Ensure FP32 for normal math
        pred = pred_vertices.float()
        gt   = gt_vertices.float()

        # Compute or use cached GT normals (no grad)
        if gt_normals is None:
            if self.detach_gt:
                gt = gt.detach()
            with torch.no_grad():
                n_gt = compute_vertex_normals(gt, faces)
        else:
            n_gt = gt_normals.float()
            # make sure unit length
            n_gt = n_gt / (n_gt.norm(dim=-1, keepdim=True) + self.eps)

        # |(p - v) · n_gt|
        diff = (pred - gt)
        dist = (diff * n_gt).sum(dim=-1).abs()   # (B, V)

        # optional masking/weighting
        if mask is not None:
            dist = dist * mask.float()
        if weights is not None:
            dist = dist * weights.float()

        if self.reduction == "mean":
            denom = (mask.float().sum() if mask is not None else dist.numel())
            loss = dist.sum() / (denom + 1e-6)
        elif self.reduction == "sum":
            loss = dist.sum()
        else:
            loss = dist  # (B, V)

        return loss

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

    def forward(self, pred_vertices: torch.Tensor, gt_vertices: torch.Tensor, faces: torch.Tensor, imgnames=None) -> torch.Tensor:
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
        if (pred_vertices is None) or (gt_vertices is None):
            if imgnames is not None:
                try:
                    print("[SMPLNormalLoss] Missing vertices. imgnames:", list(imgnames))
                except Exception:
                    print("[SMPLNormalLoss] Missing vertices. imgnames provided but not printable.")
            raise ValueError("SMPLNormalLoss: pred_vertices or gt_vertices is None")

        device = pred_vertices.device
        gt_vertices = gt_vertices.to(device=device)
        faces = faces.to(device=device, dtype=torch.long)

        # Compute area-weighted vertex normals, normalized per-vertex
        pred_normals = compute_normals_torch(pred_vertices, faces)   # (B, V, 3)
        gt_normals = compute_normals_torch(gt_vertices, faces)       # (B, V, 3)

        if (pred_normals is None) or (gt_normals is None):
            if imgnames is not None:
                try:
                    print("[SMPLNormalLoss] None normals detected. imgnames:", list(imgnames))
                except Exception:
                    print("[SMPLNormalLoss] None normals detected. imgnames provided but not printable.")
            raise ValueError("SMPLNormalLoss: pred_normals or gt_normals is None after compute_normals_torch")

        if torch.isnan(pred_normals).any() or torch.isnan(gt_normals).any():
            if imgnames is not None:
                try:
                    print("[SMPLNormalLoss] NaN normals detected. imgnames:", list(imgnames))
                except Exception:
                    print("[SMPLNormalLoss] NaN normals detected. imgnames provided but not printable.")
            # Replace NaNs with zeros to avoid crashing
            pred_normals = torch.nan_to_num(pred_normals, nan=0.0)
            gt_normals = torch.nan_to_num(gt_normals, nan=0.0)

        # Cosine similarity per vertex (orientation-agnostic via absolute value)
        cos_sim = F.cosine_similarity(pred_normals, gt_normals, dim=-1)  # (B, V)
        per_vertex_loss = 1.0 - cos_sim.abs() # TODO: orientation-agnostic, check later

        # Average over vertices, then sum over batch (match style of other losses)
        loss = per_vertex_loss.mean(dim=1).sum()
        return loss