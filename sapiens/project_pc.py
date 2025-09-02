#!/usr/bin/env python3
"""
Point Cloud Generation from Depth Maps and RGB Images
Converts depth images to colored point clouds with estimated camera parameters
"""

import os
import os.path as osp
# Prefer EGL for headless rendering (must be set before importing open3d)
os.environ.setdefault("OPEN3D_RENDERING_BACKEND", "egl")
import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm
import argparse
import glob
from PIL import Image
import hashlib
from typing import Optional
import torch
import nvtorchcam.cameras as cameras
import faiss
import plotly.io as pio
import plotly.graph_objects as go

# pio.renderers.default = "notebook_connected"

# Optional Gradio for remote-friendly web UI visualization
try:
    import gradio as gr  # type: ignore
except Exception:
    gr = None  # type: ignore

# 2d image visualization
from matplotlib import pyplot as plt

def visualize_img(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()


# Optional MeshCat for headless 3D visualization
try:
    import meshcat  # type: ignore
    from meshcat.geometry import PointCloud as MCPointCloud  # type: ignore
    from meshcat.geometry import PointsMaterial as MCPointsMaterial  # type: ignore
except Exception:
    meshcat = None  # type: ignore
    MCPointCloud = None  # type: ignore
    MCPointsMaterial = None  # type: ignore

# Reusable MeshCat visualizer instance
meshcat_vis = None

def get_meshcat_visualizer():
    global meshcat_vis
    if meshcat is None:
        return None
    if meshcat_vis is None:
        meshcat_vis = meshcat.Visualizer()  # serves on http://127.0.0.1:7000
        try:
            print(f"[MeshCat] Serving at: {meshcat_vis.url()}")
            print("Forward the port to view locally: ssh -N -L 7000:127.0.0.1:7000 <user>@<server>")
        except Exception:
            pass
    return meshcat_vis

# iphone_intrinsic_dict_portrait = {
#     "iphoneX": torch.tensor([[596.5382, 0.0, 320], [0.0, 596.5382, 240], [0.0, 0.0, 1.0]]), # 640*480
#     "iphone12": torch.tensor([[435.193, 0.0, 320], [0.0, 435.193, 240], [0.0, 0.0, 1.0]]),
# }

# 1080*1440
# K
# [[3.20512987e+03, 0.00000000e+00, 1.99443897e+03],
#  [0.00000000e+00, 3.17391061e+03, 1.41309060e+03],
#  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
# # dist
# [4.189866e-01, -3.496832e+00, -1.239132e-02, -1.649367e-03, 1.055315e+01]

# iphone_intrinsic_dict_portrait = {
#     "iphoneX": torch.tensor([[3173.91061, 0.0, 1413.09060], [0.0, 3205.12987, 1994.43897], [0.0, 0.0, 1.0]]),
# }

# init_K = torch.tensor([[1342.212, 0, 540],
#                     [0, 1342.212, 720],
#                     [0, 0, 1]])


# class TorchCameraIntrinsicsOptimizer(torch.nn.Module):
#     def __init__(self,
#                  image: np.ndarray,
#                  depth: np.ndarray,
#                  pseudo_gt_depth_pcd: o3d.geometry.PointCloud,
#                  init_intrinsic_params: torch.Tensor,
#                  cam_type: str = "OpenCVCamera"):
#         self.image = image
#         self.depth = depth
#         self.pseudo_gt_depth_pcd = pseudo_gt_depth_pcd.points
#         self.intrinsic_params = torch.nn.Parameter(init_intrinsic_params)
#         # crop usually doesn't change min(H, W) ratio
#         self.init_scale_s = min(self.image.shape[0], self.image.shape[1]) / min(init_K[0, 2], init_K[1, 2]) * 0.5
#         self.scale_params = torch.nn.Parameter(torch.tensor([self.init_scale_s]))
#         self.shift_params = torch.nn.Parameter(torch.tensor([0.0]))

#         if cam_type == "pinhole":
#             self.cam = cameras.PinholeCamera.make(torch.eye(3))
#         elif cam_type == "orthographic":
#             self.cam = cameras.OrthographicCamera.make(torch.eye(3))
#         elif cam_type == "OpenCVCamera":
#             # todo initialize opencvcamera
#             # intrinsics: (1, 4) or (1, 3, 3)
#             # ks: (1, 6)
#             # ps: (1, 2)
#             self.cam = cameras.OpenCVCamera.make(
#                 self.intrinsic_params,
#                 torch.nn.Parameter(torch.zeros(self.intrinsic_params.shape[0], 6)),
#                 torch.nn.Parameter(torch.zeros(self.intrinsic_params.shape[0], 2)),
#             )
#         else:
#             raise ValueError(f"Invalid camera type: {cam_type}")

#     # def search_initial_opencv_camera_params_models(self):


#     # def faiss_

#     def compute_reprojection_error(self, batch_img, batch_pointcloud):
#         batch_size = batch_img.shape[0]
#         batch_img = batch_img.reshape(-1, 3)
#         batch_pointcloud = batch_pointcloud.reshape(-1, 3)
#         batch_img = batch_img.to(self.device)
#         batch_pointcloud = batch_pointcloud.to(self.device)

#         batch_img_proj = self.cam.project(batch_pointcloud)
#         batch_img_proj = batch_img_proj.reshape(batch_size, -1, 2)

#         batch_img_proj = batch_img_proj.cpu().numpy()
#         batch_img = batch_img.cpu().numpy()

#         reproj_error = np.linalg.norm(batch_img_proj - batch_img, axis=1)
#         return reproj_error

#     def forward(self, batch_img, batch_pointcloud):
#         batch_size = batch_img.shape[0]


#         return loss

def read_truncated_jpg(path: str) -> np.ndarray:
    from PIL import Image, ImageFile

    # Enable loading of truncated images
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    try:
        img_pil = Image.open(path).convert('RGB')
    except (OSError, IOError) as e:
        print(f"Error loading image {path}: {e}")
        print("Skipping this image...")
        exit(1)
    return np.array(img_pil)

def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def rgb_to_xyz(rgb: torch.Tensor) -> torch.Tensor:
    rgb = srgb_to_linear(rgb)
    M = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=rgb.dtype, device=rgb.device)
    return torch.matmul(rgb, M.T)


def xyz_to_lab(xyz: torch.Tensor) -> torch.Tensor:
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x = xyz[..., 0] / Xn
    y = xyz[..., 1] / Yn
    z = xyz[..., 2] / Zn
    delta = 6.0 / 29.0

    def f(t: torch.Tensor) -> torch.Tensor:
        return torch.where(t > delta ** 3, t.pow(1.0 / 3.0), t / (3.0 * delta ** 2) + 4.0 / 29.0)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return torch.stack([L, a, b], dim=-1)


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    return xyz_to_lab(rgb_to_xyz(rgb.clamp(0.0, 1.0)))


def huber(x: torch.Tensor, delta: float = 0.02) -> torch.Tensor:
    ax = x.abs()
    return torch.where(ax < delta, 0.5 * (ax * ax) / delta, ax - 0.5 * delta)


def build_faiss_index(features_np: np.ndarray, use_gpu: bool = False, gpu_id: int = 0):
    d = int(features_np.shape[1])
    cpu_index = faiss.IndexFlatL2(d)
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
    else:
        index = cpu_index
    index.add(features_np.astype(np.float32, copy=False))
    return index


def search_faiss(index, queries_np: np.ndarray, k: int = 1):
    D, I = index.search(queries_np.astype(np.float32, copy=False), k)
    return D, I


class OpenCVIntrinsicsColorOptim(torch.nn.Module):
    def __init__(self, H, W, K0=None, ks0=None, ps0=None,
                 learn_scale=True, learn_extrinsics=False,
                 num_undistort_iters=0, z_min=1e-6, device='cuda'):
        super().__init__()
        self.H, self.W = H, W
        self.z_min = z_min
        self.num_undistort_iters = num_undistort_iters
        self.device = device

        v = torch.arange(H, device=device).float()
        u = torch.arange(W, device=device).float()
        vv, uu = torch.meshgrid(v, u, indexing='ij')
        pix = torch.stack([vv.reshape(-1), uu.reshape(-1)], -1)
        self.register_buffer('pix', pix)

        if K0 is None:
            fx0 = torch.tensor(max(W, H) * 1.0, device=device)
            fy0 = fx0.clone()
            cx0 = torch.tensor((W - 1) / 2.0, device=device)
            cy0 = torch.tensor((H - 1) / 2.0, device=device)
        else:
            K0 = torch.as_tensor(K0, dtype=torch.float32, device=device)
            fx0, fy0, cx0, cy0 = K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]

        self.log_fx = torch.nn.Parameter(torch.log(fx0.unsqueeze(0)))
        self.log_fy = torch.nn.Parameter(torch.log(fy0.unsqueeze(0)))
        self.cx = torch.nn.Parameter(cx0.unsqueeze(0))
        self.cy = torch.nn.Parameter(cy0.unsqueeze(0))

        ks0 = torch.zeros(6, device=device) if ks0 is None else torch.as_tensor(ks0, device=device, dtype=torch.float32)
        ps0 = torch.zeros(2, device=device) if ps0 is None else torch.as_tensor(ps0, device=device, dtype=torch.float32)
        self.ks = torch.nn.Parameter(ks0.clone())
        self.ps = torch.nn.Parameter(ps0.clone())

        self.learn_scale = learn_scale
        if learn_scale:
            self.log_z_scale = torch.nn.Parameter(torch.zeros(1, device=device))
            self.z_bias = torch.nn.Parameter(torch.zeros(1, device=device))

        self.learn_extrinsics = learn_extrinsics
        if learn_extrinsics:
            self.se3 = torch.nn.Parameter(torch.zeros(6, device=device))

    def _cam(self):
        fx, fy = torch.exp(self.log_fx).squeeze(), torch.exp(self.log_fy).squeeze()
        cx, cy = self.cx.squeeze(), self.cy.squeeze()
        K = torch.tensor([[fx, 0.0, cx],
                          [0.0, fy, cy],
                          [0.0, 0.0, 1.0]], device=self.device, dtype=torch.float32)
        cam = cameras.OpenCVCamera.make(
            intrinsics=K,
            ks=self.ks,
            ps=self.ps,
            z_min=self.z_min,
            num_undistort_iters=self.num_undistort_iters,
        )
        return cam

    @staticmethod
    def _apply_se3(P: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        w, v = xi[:3], xi[3:]
        theta = torch.linalg.norm(w) + 1e-12
        w_hat = w / theta
        wx, wy, wz = w_hat
        K = torch.tensor([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]], device=P.device, dtype=P.dtype)
        I = torch.eye(3, device=P.device, dtype=P.dtype)
        R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
        V = I + (1 - torch.cos(theta)) * K + ((theta - torch.sin(theta)) / (theta ** 2)) * (K @ K)
        t = V @ v
        return (P @ R.T) + t

    def backproject_and_colors(self, depth: torch.Tensor, rgb: torch.Tensor, valid_mask: Optional[torch.Tensor] = None):
        cam = self._cam()
        origin, dirs, vmask_cam = cam.pixel_to_ray(self.pix, unit_vec=False)
        depth_flat = depth.reshape(-1)

        if self.learn_scale:
            depth_flat = torch.exp(self.log_z_scale) * depth_flat + self.z_bias

        mask = vmask_cam & torch.isfinite(depth_flat) & (depth_flat > 0)
        if valid_mask is not None:
            mask = mask & valid_mask.reshape(-1).to(mask.dtype).bool()

        P = origin + dirs * depth_flat.unsqueeze(-1)
        P = P[mask]

        C = rgb.reshape(-1, 3)[mask]
        if self.learn_extrinsics:
            P = self._apply_se3(P, self.se3)
        return P, C, vmask_cam

    def current_K(self) -> np.ndarray:
        cam = self._cam()
        cam.intrinsics = cam._values['affine']
        return cam.intrinsics.detach().cpu().numpy()


def optimize_intrinsics_with_color(
    depth,
    rgb,
    Q_xyz,
    Q_rgb,
    H,
    W,
    K0=None,
    ks0=None,
    ps0=None,
    iters: int = 800,
    lr_f: float = 3e-3,
    lr_c: float = 1e-2,
    lr_d: float = 5e-3,
    lr_s: float = 1e-2,
    learn_scale: bool = True,
    learn_extrinsics: bool = False,
    num_undistort_iters: int = 0,
    z_min: float = 1e-6,
    chamfer_clip: float = 0.10,
    w_color: float = 0.2,
    alpha_lab: float = 0.01,
    subsample_pairs: int = 80000,
    use_gpu_faiss: bool = True,
    device: str = 'cuda'
):
    depth_t = torch.as_tensor(depth, dtype=torch.float32, device=device)
    rgb_t = torch.as_tensor(rgb, dtype=torch.float32, device=device)
    Q_xyz_t = torch.as_tensor(Q_xyz, dtype=torch.float32, device=device)
    Q_rgb_t = torch.as_tensor(Q_rgb, dtype=torch.float32, device=device)

    Q_lab_t = rgb_to_lab(Q_rgb_t)
    Q_feat = torch.cat([Q_xyz_t, alpha_lab * Q_lab_t], dim=-1).cpu().numpy().astype(np.float32)
    q_index = build_faiss_index(Q_feat, use_gpu=use_gpu_faiss)

    model = OpenCVIntrinsicsColorOptim(H, W, K0, ks0, ps0, learn_scale, learn_extrinsics,
                                       num_undistort_iters, z_min, device).to(device)

    params = [
        {'params': [model.log_fx, model.log_fy], 'lr': lr_f},
        {'params': [model.cx, model.cy], 'lr': lr_c},
        {'params': [model.ks, model.ps], 'lr': lr_d},
    ]
    if learn_scale:
        params += [{'params': [model.log_z_scale, model.z_bias], 'lr': lr_s}]
    if learn_extrinsics:
        params += [{'params': [model.se3], 'lr': 5e-3}]
    opt = torch.optim.Adam(params, betas=(0.9, 0.999))

    for it in range(iters):
        opt.zero_grad()

        P_pred, C_pred, mask_flat = model.backproject_and_colors(depth_t, rgb_t)
        # save depth_t as png
        save_visualize_depth(depth_t.cpu().detach().numpy(), "output/depth_t.png")
        # save_visualize_depth(rgb_t.cpu().detach().numpy(), "output/rgb_t.png")
        # save_visualize_depth(P_pred.cpu().detach().numpy(), "output/P_pred.png")
        # save_visualize_depth(C_pred.cpu().detach().numpy(), "output/C_pred.png")
        if P_pred.shape[0] == 0:
            raise RuntimeError("No valid depth pixels after masking; check your depth units and z_min.")

        P_lab = rgb_to_lab(C_pred)

        P_feat = torch.cat([P_pred, alpha_lab * P_lab], dim=-1).detach().cpu().numpy().astype(np.float32)
        D, I = search_faiss(q_index, P_feat, k=1)
        I = torch.as_tensor(I[:, 0], device=device, dtype=torch.long)

        Q_nn = Q_xyz_t[I]
        Q_lab_nn = Q_lab_t[I]

        if P_pred.shape[0] > subsample_pairs:
            sel = torch.randperm(P_pred.shape[0], device=device)[:subsample_pairs]
            P_pred_s = P_pred[sel]
            Q_nn_s = Q_nn[sel]
            P_lab_s = P_lab[sel]
            Q_lab_nn_s = Q_lab_nn[sel]
        else:
            P_pred_s, Q_nn_s = P_pred, Q_nn
            P_lab_s, Q_lab_nn_s = P_lab, Q_lab_nn

        geo_res = (P_pred_s - Q_nn_s)
        geo_d = geo_res.norm(p=2, dim=-1).clamp(max=chamfer_clip)
        loss_geo = huber(geo_d, delta=0.02).mean()

        loss_col = (P_lab_s - Q_lab_nn_s).abs().mean()

        fx, fy = torch.exp(model.log_fx), torch.exp(model.log_fy)
        cx_cy_reg = (((model.cx - (W - 1) / 2) / W) ** 2 + ((model.cy - (H - 1) / 2) / H) ** 2)
        fx_fy_reg = (torch.log(fx / fy)).pow(2)
        focal_reg = huber(torch.log(fx / (0.2 * W))) + huber(torch.log((4.0 * W) / fx))

        loss = loss_geo + w_color * loss_col + 0.01 * cx_cy_reg + 0.01 * fx_fy_reg + 0.001 * focal_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if it % 50 == 0 or it == iters - 1:
            s = torch.exp(model.log_z_scale).item() if model.learn_scale else 1.0
            b = model.z_bias.item() if model.learn_scale else 0.0
            print(f"[{it:04d}] loss={loss.item():.5f} | geo={loss_geo.item():.5f} col={loss_col.item():.5f} "
                  f"| fx={torch.exp(model.log_fx).item():.1f} fy={torch.exp(model.log_fy).item():.1f} "
                  f"cx={model.cx.item():.1f} cy={model.cy.item():.1f} | scale={s:.6f} bias={b:.6f}")

    out = {
        "K": model.current_K(),
        "ks": model.ks.detach().cpu().numpy(),
        "ps": model.ps.detach().cpu().numpy(),
    }
    if model.learn_scale:
        out["scale"] = float(torch.exp(model.log_z_scale).detach().cpu())
        out["bias"] = float(model.z_bias.detach().cpu())
    if model.learn_extrinsics:
        out["se3"] = model.se3.detach().cpu().numpy()
    return out



# vertical field-of-view
def estimate_camera_intrinsics(width, height, fov_vertical_degrees: float = 65.0, fov_horizontal_degrees: float = 0.0):
    """
    Estimate camera intrinsic matrix K from image size and provided FOVs.

    - If both horizontal and vertical FOVs are provided (> 0), compute both fx and fy.
    - If only one FOV is provided (> 0), assume square pixels and set the other focal equal.
    - If neither is provided (<= 0), fall back to 65 degrees and square pixels.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        fov_vertical_degrees: Assumed vertical field-of-view in degrees (0 to ignore).
        fov_horizontal_degrees: Assumed horizontal field-of-view in degrees (0 to ignore).

    Returns:
        3x3 intrinsic matrix K.
    """
    fx = None
    fy = None

    # Compute from provided horizontal FOV
    if float(fov_horizontal_degrees) > 0.0:
        fov_horizontal = np.radians(float(fov_horizontal_degrees))
        fx = width / (2.0 * np.tan(fov_horizontal * 0.5))

    # Compute from provided vertical FOV
    if float(fov_vertical_degrees) > 0.0:
        fov_vertical = np.radians(float(fov_vertical_degrees))
        fy = height / (2.0 * np.tan(fov_vertical * 0.5))

    # Assume square pixels to fill missing focal(s)
    if fx is None and fy is not None:
        fx = fy
    if fy is None and fx is not None:
        fy = fx

    # Fallback if neither FOV provided
    if fx is None and fy is None:
        fallback_fov = np.radians(65.0)
        fx = width / (2.0 * np.tan(fallback_fov * 0.5))
        fy = fx

    # Principal point at image center (cx along width, cy along height)
    cx = width * 0.5
    cy = height * 0.5

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    return K

def save_visualize_depth(depth_image: np.ndarray,
                         save_path: Optional[str],
                         mask: Optional[np.ndarray] = None) -> Optional[str]:
    """
    Create a colored visualization of a depth map and save it to disk.

    - Ignores NaN/Inf values when computing min/max
    - Normalizes valid depths to 0..255
    - Uses OpenCV's INFERNO colormap for better contrast

    Args:
        depth_image: numpy array of shape (H, W), raw depth values (float or int)
        save_path: optional explicit path to save PNG. If None, saves under
                   ./output/depth_vis_debug/depth_vis_<hash>.png

    Returns:
        The file path where the visualization was saved, or None if saving failed.
    """
    try:
        if depth_image is None or depth_image.size == 0:
            return None

        # Prepare output directory if not provided
        if save_path is None:
            out_dir = os.path.join(os.getcwd(), 'output', 'depth_vis_debug')
            os.makedirs(out_dir, exist_ok=True)
            # Hash on shape and a few stats to create a stable-ish name per map
            stats = np.array([
                depth_image.shape[0], depth_image.shape[1],
                float(np.nanmin(depth_image)) if np.isfinite(depth_image).any() else 0.0,
                float(np.nanmax(depth_image)) if np.isfinite(depth_image).any() else 0.0,
            ], dtype=np.float32).tobytes()
            short_hash = hashlib.md5(stats).hexdigest()[:8]
            save_path = os.path.join(out_dir, f'depth_vis_{short_hash}.png')

        # Choose mask: if provided, use it; else treat finite values as foreground
        if mask is None:
            mask = np.isfinite(depth_image)

        if not np.any(mask):
            vis = np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.uint8)
            cv2.imwrite(save_path, vis)
            return save_path

        # Match demo_depth_vis.py visualization exactly
        # 1) Compute min/max over foreground only
        depth_foreground = depth_image[mask]
        dmin = float(np.min(depth_foreground))
        dmax = float(np.max(depth_foreground))
        if dmax <= dmin:
            dmax = dmin + 1e-6

        # 2) Invert-normalize foreground to 0..255
        depth_norm_fg = 1.0 - ((depth_foreground - dmin) / (dmax - dmin))
        depth_norm_fg = (depth_norm_fg * 255.0).clip(0, 255).astype(np.uint8)

        # 3) Color only the foreground using INFERNO; background set to 100 gray
        h, w = depth_image.shape[:2]
        vis = np.full((h, w, 3), 100, dtype=np.uint8)
        colored_fg = cv2.applyColorMap(depth_norm_fg, cv2.COLORMAP_INFERNO)
        colored_fg = colored_fg.reshape(-1, 3)
        vis[mask] = colored_fg

        # Write file (BGR image)
        cv2.imwrite(save_path, vis)
        return save_path
    except Exception:
        return None

def save_pc(pcd):
    o3d.io.write_point_cloud("output/pc.ply", pcd)

def _pcd_to_plotly_figure(pcd: o3d.geometry.PointCloud, max_points: int = 200000) -> "go.Figure":
    # Accept either Open3D point cloud or path string
    if isinstance(pcd, str) or isinstance(pcd, os.PathLike):
        try:
            if os.path.exists(pcd):
                pcd = o3d.io.read_point_cloud(pcd)
        except Exception:
            pcd = None
    if not isinstance(pcd, o3d.geometry.PointCloud):
        fig = go.Figure()
        fig.update_layout(title="Invalid point cloud input", scene_aspectmode="data")
        return fig

    pts = np.asarray(pcd.points)
    if pts.size == 0:
        fig = go.Figure()
        fig.update_layout(title="Empty point cloud", scene_aspectmode="data")
        return fig

    cols = np.asarray(pcd.colors) if len(pcd.colors) > 0 else None

    num_points = pts.shape[0]
    if num_points > max_points:
        sel = np.random.choice(num_points, size=max_points, replace=False)
        pts = pts[sel]
        if cols is not None and cols.shape[0] == num_points:
            cols = cols[sel]

    if cols is not None and cols.shape[0] == pts.shape[0]:
        cols_255 = (np.clip(cols, 0.0, 1.0) * 255.0).astype(np.uint8)
        marker_color = [f"rgb({r},{g},{b})" for r, g, b in cols_255]
    else:
        marker_color = "#1f77b4"

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=marker_color,
                    opacity=1.0
                ),
            )
        ]
    )
    fig.update_layout(
        title="Point Cloud",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    return fig

def visualize_point_cloud(pcd: o3d.geometry.PointCloud,
                          server_port: int = 7860,
                          server_name: str = "0.0.0.0",
                          share: bool = False):
    # Accept either an Open3D point cloud object or a filesystem path to a PLY/PCD file
    if isinstance(pcd, str) or isinstance(pcd, os.PathLike):
        try:
            if os.path.exists(pcd):
                pcd = o3d.io.read_point_cloud(pcd)
        except Exception as e:
            print(f"[Gradio] Failed to load point cloud from path '{pcd}': {e}")
            return None
    if not isinstance(pcd, o3d.geometry.PointCloud):
        print("[Gradio] visualize_point_cloud expects an Open3D PointCloud or a valid file path.")
        return None

    if gr is None:
        print("[Gradio] Not installed. Run: pip install gradio plotly. Saving PLY to output/pc.ply instead.")
        try:
            os.makedirs("output", exist_ok=True)
            o3d.io.write_point_cloud("output/pc.ply", pcd)
        except Exception:
            pass
        return None

    fig = _pcd_to_plotly_figure(pcd)

    with gr.Blocks() as demo:
        gr.Markdown("### Point Cloud Viewer")
        gr.Markdown("Interact with the 3D scatter using mouse drag and scroll.")
        gr.Plot(value=fig, label="3D Point Cloud", height=640)

    print(f"[Gradio] Serving at http://{server_name}:{server_port}")
    print("Forward the port to your local machine:")
    print(f"ssh -N -L {server_port}:127.0.0.1:{server_port} <user>@<server>")
    demo.launch(server_name=server_name, server_port=server_port, share=share, inbrowser=False, show_error=True)
    return None

def depth_to_point_cloud(depth_image,
                        rgb_image,
                        camera_intrinsics,
                        depth_scale: float = 1.0,
                        scale_s: float = 1.0,
                        shift_t: float = 0.0):
    """
    Convert depth image to colored point cloud

    Args:
        depth_image: Depth map (H x W)
        rgb_image: RGB image (H x W x 3)
        camera_intrinsics: 3x3 camera intrinsic matrix
        depth_scale: Unit conversion for depth values. If your abs depth is in
            meters already, use 1.0. If your abs depth is in millimeters, use 1000.0.
        scale_s: Scale to convert relative depth to absolute: abs = rel * s + t
        shift_t: Shift to convert relative depth to absolute: abs = rel * s + t

    Returns:
        Open3D point cloud object
    """
    height, width = depth_image.shape

    # Create coordinate grids
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    # Convert relative depth to absolute depth via scale and shift, then to meters
    rel_depth = depth_image.astype(np.float32)
    abs_depth = rel_depth * float(scale_s) + float(shift_t)
    depth_m = abs_depth / float(depth_scale)

    # Remove invalid depth values and background pixels
    # 1. Depth mask: remove where depth is invalid (NaN, inf, or unrealistic values)
    depth_mask = (~np.isnan(depth_m)) & (~np.isinf(depth_m))

    # 2. RGB mask: remove where RGB is black/background (all channels are 0)
    rgb_mask = np.any(rgb_image > 0, axis=2)  # True where any RGB channel > 0

    # Combine masks: valid depth AND non-black RGB
    valid_mask = depth_mask & rgb_mask
    mask_flat = ~valid_mask
    # save_visualize_depth(valid_mask)

    # Extract valid pixels
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    depth_valid = depth_m[valid_mask]

    # Get camera parameters
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

    # Convert to 3D coordinates
    x = (u_valid - cx) * depth_valid / fx
    y = (v_valid - cy) * depth_valid / fy
    z = depth_valid

    # Stack coordinates
    points_3d = np.stack([x, y, z], axis=1)

    # Get corresponding colors
    colors = rgb_image[valid_mask] / 255.0  # Normalize to [0, 1]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Optional interactive visualization during conversion (off by default to avoid blocking)
    if os.environ.get("PC_VISUALIZE_ON_CONVERT", "0") == "1":
        visualize_point_cloud(pcd)

    return pcd, mask_flat

def cam_depth_to_point_cloud(depth_image,
                             rgb_image,
                             depth_scale: float = 1.0,
                             scale_s: float = 1.0,
                             shift_t: float = 0.0,
                             fov_vertical_degrees: float = 0.0,
                             fov_horizontal_degrees: float = 0.0,
                             base_name: str = "",
                             output_dir: str = ""):

    height, width = depth_image.shape
    camera_intrinsics = estimate_camera_intrinsics(
        width,
        height,
        fov_vertical_degrees=fov_vertical_degrees,
        fov_horizontal_degrees=fov_horizontal_degrees,
    )

    # Convert to point cloud
    pcd, mask_flat = depth_to_point_cloud(depth_image,
                            rgb_image,
                            camera_intrinsics,
                            depth_scale=depth_scale,
                            scale_s=scale_s,
                            shift_t=shift_t)

    pseudo_gt_pcd_file = f"/home/cevin/Meitu/Pi3/output/test_data_img/{base_name.split('.')[0]}.ply"
    pseudo_gt_pcd = o3d.io.read_point_cloud(pseudo_gt_pcd_file)
    # Save point cloud
    output_pseudo_gt_file = os.path.join(output_dir, os.path.splitext(base_name)[0] + f"_pseudo_gt_fov_{fov_vertical_degrees:.1f}_{fov_horizontal_degrees:.1f}.ply")
    o3d.io.write_point_cloud(output_pseudo_gt_file, pseudo_gt_pcd)



def process_depth_to_pointcloud(depth_dir,
                               rgb_dir,
                               output_dir,
                               depth_scale: float = 1.0,
                               render_png: bool = True,
                               scale_s: float = 1.0,
                               shift_t: float = 0.0,
                               fov_horizontal_degrees_list: float = [],
                               fov_vertical_degrees_list: float = [32.5, 45.0, 52.5, 60.0, 65.0, 70.0]):
    """
    Process all depth images in directory to point clouds

    Args:
        depth_dir: Directory containing depth images (will look for .npy files in parent dir)
        rgb_dir: Directory containing RGB images
        output_dir: Output directory for point clouds
        depth_scale: Scale factor for depth values (1.0 for raw depth data)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Look for .npy files in the parent directory (where raw depth data is stored)
    parent_dir = os.path.dirname(depth_dir)
    depth_files = glob.glob(os.path.join(parent_dir, "*.npy"))
    depth_files = sorted(depth_files)

    print(f"Found {len(depth_files)} depth images")

    for depth_file in tqdm(depth_files, desc="Converting to point clouds"):
        try:
            # Get corresponding RGB file
            base_name = os.path.basename(depth_file)
            rgb_file = os.path.join(rgb_dir, base_name).replace(".npy", ".jpg")

            # Try different extensions if exact match not found
            if not os.path.exists(rgb_file):
                name_without_ext = os.path.splitext(base_name)[0]
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    rgb_candidate = os.path.join(rgb_dir, name_without_ext + ext)
                    if os.path.exists(rgb_candidate):
                        rgb_file = rgb_candidate
                        break

            if not os.path.exists(rgb_file):
                print(f"Warning: RGB file not found for {base_name}, skipping")
                continue

            # Provide png_path for rendering helper that derives output names
            global png_path
            png_path = rgb_file

            # Load raw depth data from .npy file - skip files with legacy dtype issues
            try:
                # Simple approach: try to load normally first
                depth_image = np.load(depth_file)
            except (AttributeError, ValueError) as e:
                # Handle legacy files referencing removed numpy attributes (e.g., numpy.core.multiarray.number)
                if 'numpy.core.multiarray' in str(e) and 'has no attribute' in str(e):
                    try:
                        import re
                        m = re.search(r"attribute '([^']+)'", str(e))
                        missing_attr = m.group(1) if m else None
                        if missing_attr:
                            # Best-effort monkey-patch to satisfy old pickle references
                            if not hasattr(np.core.multiarray, missing_attr) and hasattr(np, missing_attr):
                                setattr(np.core.multiarray, missing_attr, getattr(np, missing_attr))
                        # Retry with pickle enabled for legacy encodings
                        depth_image = np.load(depth_file, allow_pickle=True)
                    except Exception:
                        print(f"Skipping legacy numpy file {depth_file}: incompatible dtype")
                        continue
                else:
                    print(f"Error loading depth file {depth_file}: {e}")
                    continue
            except Exception as e:
                # Final fallback: try with pickle enabled
                try:
                    depth_image = np.load(depth_file, allow_pickle=True)
                except Exception:
                    print(f"Error loading depth file {depth_file}: {e}")
                    continue

            if depth_image is None or depth_image.size == 0:
                print(f"Warning: Could not load depth data {depth_file}")
                continue

            # Save a depth visualization matching demo depth vis (use RGB-based mask)
            # This will save under depth_vis_debug with a stable hashed name
            # Actual saving of the final PNG happens later; this is for quick inspection
            # (non-fatal if it fails)
            try:
                if 'rgb_image' in locals() and rgb_image is not None:
                    depth_vis_mask = np.any(rgb_image > 0, axis=2)
                else:
                    depth_vis_mask = np.isfinite(depth_image)
                output_dir_depth_vis = output_dir.replace("/"+output_dir.split("/")[-1], "/depth_vis")
                os.makedirs(output_dir_depth_vis, exist_ok=True)
                save_path = os.path.join(output_dir_depth_vis, os.path.splitext(base_name)[0]+".png")
                save_visualize_depth(depth_image, save_path, depth_vis_mask)
            except Exception:
                pass

            # Load RGB image
            rgb_image = read_truncated_jpg(rgb_file)
            if rgb_image is None:
                print(f"Warning: Could not load RGB image {rgb_file}")
                continue

            # Resize RGB to match depth if needed
            if rgb_image.shape[:2] != depth_image.shape:
                rgb_image = cv2.resize(rgb_image, (depth_image.shape[1], depth_image.shape[0]))

            # Estimate camera intrinsics
            if len(fov_vertical_degrees_list) != 0:
                for fov_vertical_degrees in fov_vertical_degrees_list:
                    cam_depth_to_point_cloud(depth_image,
                                            rgb_image,
                                            depth_scale=depth_scale,
                                            scale_s=scale_s,
                                            shift_t=shift_t,
                                            fov_vertical_degrees=fov_vertical_degrees,
                                            fov_horizontal_degrees=0,
                                            base_name=base_name,
                                            output_dir=output_dir)

            if len(fov_horizontal_degrees_list) != 0:
                for fov_horizontal_degrees in fov_horizontal_degrees_list:
                    cam_depth_to_point_cloud(depth_image,
                                            rgb_image,
                                            depth_scale=depth_scale,
                                            scale_s=scale_s,
                                            shift_t=shift_t,
                                            fov_vertical_degrees=0,
                                            fov_horizontal_degrees=fov_horizontal_degrees,
                                            base_name=base_name,
                                            output_dir=output_dir)


            depth_image[mask_flat] = 0

            opt = optimize_camera_intrinsics_on_pseudo_gt_pcd(depth_image, rgb_image, camera_intrinsics)

            pcd_opt = optimized_depth2pointcloud_projection(rgb_image, depth_image, opt)

            # Save point cloud
            output_file = os.path.join(output_dir, os.path.splitext(base_name)[0] + ".ply")
            o3d.io.write_point_cloud(output_file, pcd)

            output_opt_file = os.path.join(output_dir, os.path.splitext(base_name)[0] + "_opt.ply")
            o3d.io.write_point_cloud(output_opt_file, pcd_opt)

            print(f"Saved point cloud: {output_file} ({len(pcd.points)} points)")

            # Render point cloud to PNG if requested
            if render_png:
                png_file = os.path.join(output_dir, os.path.splitext(base_name)[0] + "_render.png")
                render_point_cloud_png(output_file, png_file, camera_intrinsics, width, height)

                png_opt_file = os.path.join(output_dir, os.path.splitext(base_name)[0] + "_opt_render.png")
                render_point_cloud_png(output_opt_file, png_opt_file, camera_intrinsics, width, height)

                # # Also render with provided world2cam-based helper
                # try:
                #     world2cam = torch.eye(4)
                #     pts_world = torch.as_tensor(np.asarray(pcd.points), dtype=torch.float32)
                #     cols = torch.as_tensor(np.asarray(pcd.colors), dtype=torch.float32)
                #     render_with_world2cam_suffix(world2cam, "render", "pcd", pts_world, cols, height, width)

                #     pts_world_opt = torch.as_tensor(np.asarray(pcd_opt.points), dtype=torch.float32)
                #     cols_opt = torch.as_tensor(np.asarray(pcd_opt.colors), dtype=torch.float32)
                #     render_with_world2cam_suffix(world2cam, "render", "opt", pts_world_opt, cols_opt, height, width)
                # except Exception as e:
                #     print(f"[WARN] render_with_world2cam_suffix failed: {e}")

        except Exception as e:
            print(f"Error processing {depth_file}: {e}")
            continue


def render_point_cloud_png(ply_file, output_png, camera_intrinsics, original_width, original_height):
    """
    Render a point cloud to PNG. Prefer Open3D OffscreenRenderer (EGL). Fallback to manual projection.
    """
    try:
        # Load point cloud
        pcd = o3d.io.read_point_cloud(ply_file)
        # Try EGL offscreen renderer first
        try:
            w, h = int(original_width), int(original_height)
            renderer = o3d.visualization.rendering.OffscreenRenderer(w, h)
            scene = renderer.scene
            scene.set_background([1.0, 1.0, 1.0, 0.0])

            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            mat.point_size = 2.0

            scene.add_geometry("pcd", pcd, mat)

            # Camera from intrinsics: approximate by FOV
            fx, fy = float(camera_intrinsics[0, 0]), float(camera_intrinsics[1, 1])
            cx, cy = float(camera_intrinsics[0, 2]), float(camera_intrinsics[1, 2])
            fov_y = np.degrees(2.0 * np.arctan2(0.5 * h, fy)) if fy > 0 else 60.0
            scene.camera.set_projection(fov_y, w / h, 0.01, 100.0,
                                        o3d.visualization.rendering.Camera.FovType.Vertical)

            # Look at the cloud center from a distance along +Z
            bbox = pcd.get_axis_aligned_bounding_box()
            center = bbox.get_center().tolist()
            extent = np.linalg.norm(np.array(bbox.get_extent()))
            eye = [center[0], center[1], center[2] + max(1.0, 1.5 * extent)]
            up = [0.0, 1.0, 0.0]
            scene.camera.look_at(center, eye, up)

            img = renderer.render_to_image()
            o3d.io.write_image(output_png, img)
            print(f"Saved rendered PNG (EGL): {output_png}")
            return True
        except Exception:
            pass

        # Fallback: manual projection
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        print(f"Rendering {len(points)} points to PNG by projection (fallback)...")
        if len(points) == 0 or len(colors) == 0:
            print("No points or colors to render!")
            return False
        H, W = original_height, original_width
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
        z = points[:, 2]
        front_mask = z > 1e-6
        if not front_mask.any():
            print("No points in front of camera!")
            return False
        pts_cam = points[front_mask]
        cols = colors[front_mask]
        z = z[front_mask]
        u = fx * (pts_cam[:, 0] / z) + cx
        v = fy * (pts_cam[:, 1] / z) + cy
        x = np.round(u).astype(np.int64)
        y = np.round(v).astype(np.int64)
        in_bounds = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        if not in_bounds.any():
            print("No projected points inside image bounds!")
            return False
        x = x[in_bounds]; y = y[in_bounds]; z = z[in_bounds]; cols = cols[in_bounds]
        lin_idx = (y * W + x).astype(np.int64)
        order = np.lexsort((z, lin_idx))
        lin_sorted = lin_idx[order]
        _, first_pos = np.unique(lin_sorted, return_index=True)
        sel = order[first_pos]
        img_out = np.zeros((H, W, 3), dtype=np.uint8)
        final_x = lin_idx[sel] % W
        final_y = lin_idx[sel] // W
        final_cols = (cols[sel] * 255.0).clip(0, 255).astype(np.uint8)
        img_out[final_y, final_x] = final_cols
        Image.fromarray(img_out).save(output_png)
        print(f"Saved rendered PNG (fallback): {output_png}")
        return True

    except Exception as e:
        print(f"Error rendering point cloud to PNG: {e}")
        return False

def render_with_world2cam_suffix(world2cam_mat: torch.Tensor, out_folder: str, suffix: str, pts_world: torch.Tensor, cols: torch.Tensor, H: int, W: int):
    if pts_world.numel() == 0:
        return
    ones2 = torch.ones((pts_world.shape[0], 1), device=pts_world.device, dtype=pts_world.dtype)
    pts_world_h2 = torch.cat([pts_world, ones2], dim=1)
    pts_cam_h2 = (world2cam_mat @ pts_world_h2.T).T
    pts_cam2 = pts_cam_h2[:, :3]
    z2 = pts_cam2[:, 2]
    front2 = z2 > 1e-6
    if not front2.any():
        print(f"[DEBUG] No points in front of camera for {out_folder} view")
        return
    pts_cam2 = pts_cam2[front2]
    cols2 = cols[front2]
    z2 = z2[front2]
    fov_scale = 1.0
    fx2 = float(max(H, W)) * fov_scale
    fy2 = float(max(H, W)) * fov_scale
    cx2 = float(W) / 2.0
    cy2 = float(H) / 2.0
    u2 = fx2 * (pts_cam2[:, 0] / z2) + cx2
    v2 = fy2 * (pts_cam2[:, 1] / z2) + cy2
    x2 = torch.round(u2).to(torch.int64)
    y2 = torch.round(v2).to(torch.int64)
    in_bounds2 = (x2 >= 0) & (x2 < W) & (y2 >= 0) & (y2 < H)
    if not in_bounds2.any():
        print(f"[DEBUG] No projected points in bounds for {out_folder} view")
        return
    x2 = x2[in_bounds2]
    y2 = y2[in_bounds2]
    z2 = z2[in_bounds2]
    cols2 = cols2[in_bounds2]
    lin_idx2 = (y2 * W + x2).detach().cpu().numpy().astype(np.int64)
    z_np2 = z2.detach().cpu().numpy()
    cols_np2 = (cols2.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    order2 = np.lexsort((z_np2, lin_idx2))
    lin_sorted2 = lin_idx2[order2]
    first_pos2 = np.unique(lin_sorted2, return_index=True)[1]
    sel2 = order2[first_pos2]
    img_out2 = np.zeros((H, W, 3), dtype=np.uint8)
    xs2 = lin_idx2[sel2] % W
    ys2 = lin_idx2[sel2] // W
    img_out2[ys2, xs2] = cols_np2[sel2]
    png_out_folder = osp.join(png_path.rsplit('/', 1)[0], out_folder)
    if not osp.exists(png_out_folder):
        os.makedirs(png_out_folder)
    out_path = osp.join(png_out_folder, png_path.rsplit('/', 1)[1].split('.')[0] + f'_{suffix}.png')
    Image.fromarray(img_out2).save(out_path)
    print(f"Saved rendered image to: {out_path}")
    return out_path

# def visualize_point_cloud(pcd):
#     """
#     Visualize a point cloud file
#     """
#     # Estimate normals for better visualization
#     pcd.estimate_normals()

#     # Visualize
#     o3d.visualization.draw_geometries([pcd],
#                                     window_name="Point Cloud Viewer",
#                                     width=1024,
#                                     height=768)

def optimize_camera_intrinsics_on_pseudo_gt_pcd(depth_image: np.ndarray,
                                                rgb_image: np.ndarray,
                                                init_K: np.ndarray,
                                                learn_scale: bool = True,
                                                learn_extrinsics: bool = False,
                                                iters: int = 300,
                                                chamfer_clip: float = 0.10,
                                                w_color: float = 0.2,
                                                alpha_lab: float = 0.01,
                                                subsample_pairs: int = 60000,
                                                use_gpu_faiss: bool = False,
                                                num_undistort_iters: int = 0,
                                                device: Optional[str] = None) -> dict:
    H, W = depth_image.shape
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pcd_init, mask_flat = depth_to_point_cloud(depth_image, rgb_image, init_K)
    depth_image[mask_flat] = 0
    # convert depth nan to 0
    depth_image[np.isnan(depth_image)] = 0
    Q_xyz = np.asarray(pcd_init.points)
    Q_rgb = np.asarray(pcd_init.colors).astype(np.float32)

    rgb_float = (rgb_image.astype(np.float32) / 255.0) if rgb_image.dtype != np.float32 or rgb_image.max() > 1.0 else rgb_image

    opt_result = optimize_intrinsics_with_color(
        depth=depth_image,
        rgb=rgb_float,
        Q_xyz=Q_xyz,
        Q_rgb=Q_rgb,
        H=H,
        W=W,
        K0=init_K,
        ks0=None,
        ps0=None,
        iters=iters,
        learn_scale=learn_scale,
        learn_extrinsics=learn_extrinsics,
        num_undistort_iters=num_undistort_iters,
        chamfer_clip=chamfer_clip,
        w_color=w_color,
        alpha_lab=alpha_lab,
        subsample_pairs=subsample_pairs,
        use_gpu_faiss=use_gpu_faiss,
        device=device,
    )

    return opt_result

def optimized_depth2pointcloud_projection(rgb_image: np.ndarray, depth_image: np.ndarray, opt_result: dict, num_undistort_iters: int = 0):
    H, W = depth_image.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    rgb = torch.as_tensor((rgb_image.astype(np.float32) / 255.0), dtype=torch.float32, device=device)
    depth = torch.as_tensor(depth_image.astype(np.float32), dtype=torch.float32, device=device)

    K = torch.as_tensor(opt_result.get('K'), dtype=torch.float32, device=device)
    ks = torch.as_tensor(opt_result.get('ks', np.zeros(6, dtype=np.float32)), dtype=torch.float32, device=device)
    ps = torch.as_tensor(opt_result.get('ps', np.zeros(2, dtype=np.float32)), dtype=torch.float32, device=device)
    scale = float(opt_result.get('scale', 1.0))
    bias = float(opt_result.get('bias', 0.0))

    cam = cameras.OpenCVCamera.make(
        intrinsics=K,
        ks=ks,
        ps=ps,
        z_min=1e-6,
        num_undistort_iters=num_undistort_iters,
    )

    v = torch.arange(H, device=device).float()
    u = torch.arange(W, device=device).float()
    vv, uu = torch.meshgrid(v, u, indexing='ij')
    pix = torch.stack([vv.reshape(-1), uu.reshape(-1)], -1)

    origin, dirs, vmask = cam.pixel_to_ray(pix, unit_vec=False)
    depth_flat = (scale * depth + bias).reshape(-1)

    mask = vmask & torch.isfinite(depth_flat) & (depth_flat > 0)
    P = origin + dirs * depth_flat.unsqueeze(-1)
    P = P[mask]
    C = rgb.reshape(-1, 3)[mask]

    points_3d = P.detach().cpu().numpy()
    colors = C.detach().cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def main():
    parser = argparse.ArgumentParser(description="Convert depth images to colored point clouds")
    parser.add_argument("--depth_dir",
                       default="/home/cevin/Meitu/sapiens/output/depth_normal_5col_ALL49/depth_ALL",
                       help="Directory containing depth images")
    parser.add_argument("--rgb_dir",
                       default="/home/cevin/Meitu/data/test_data_img/all",
                       help="Directory containing RGB images")
    parser.add_argument("--output_dir",
                       default="/home/cevin/Meitu/sapiens/output/point_clouds",
                       help="Output directory for point clouds")
    parser.add_argument("--depth_scale", type=float, default=1.0,
                       help="Unit conversion for absolute depth. If abs depth is in meters use 1.0; if in mm use 1000.0")
    parser.add_argument("--scale_s", type=float, default=1.0,
                       help="Scale s for converting relative depth to absolute: abs = rel * s + t")
    parser.add_argument("--shift_t", type=float, default=0.0,
                       help="Shift t for converting relative depth to absolute: abs = rel * s + t")
    parser.add_argument("--fov_horizontal_degrees_list", type=float, default=[32.5, 45.0, 52.5, 60.0, 65.0, 70.0],
                       help="Assumed horizontal FOV (degrees) to estimate intrinsics if unknown")
    parser.add_argument("--fov_vertical_degrees_list", type=float, default=[],
                       help="Assumed vertical FOV (degrees); if 0, ignored. If both provided, use both.")
    parser.add_argument("--render_png", action="store_true",
                       help="Render PNG images of point clouds")
    parser.add_argument("--visualize", type=str, default=None,
                       help="Visualize a specific PLY file")
    parser.add_argument("--gradio_port", type=int, default=7860,
                       help="Port to run the Gradio server on")
    parser.add_argument("--gradio_server_name", type=str, default="0.0.0.0",
                       help="Server name (bind address) for Gradio, e.g., 0.0.0.0")
    parser.add_argument("--gradio_share", action="store_true",
                       help="Enable Gradio public share link (may upload metadata externally)")

    args = parser.parse_args()

    if args.visualize:
        pcd_input = args.visualize
        if isinstance(pcd_input, str) and os.path.exists(pcd_input):
            try:
                pcd_input = o3d.io.read_point_cloud(pcd_input)
            except Exception:
                pass
        visualize_point_cloud(
            pcd_input,
            server_port=args.gradio_port,
            server_name=args.gradio_server_name,
            share=args.gradio_share,
        )
    else:
        print(f"Processing depth images from: {args.depth_dir}")
        print(f"Using RGB images from: {args.rgb_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Depth scale: {args.depth_scale}")

        process_depth_to_pointcloud(
            args.depth_dir,
            args.rgb_dir,
            args.output_dir,
            depth_scale=args.depth_scale,
            render_png=args.render_png,
            scale_s=args.scale_s,
            shift_t=args.shift_t,
            fov_horizontal_degrees_list=args.fov_horizontal_degrees_list,
            fov_vertical_degrees_list=args.fov_vertical_degrees_list,
        )
        print("Point cloud generation completed!")

if __name__ == "__main__":
    main()
