import cv2
import os
import json
import torch
import subprocess
import shutil
from core.utils.torch_compat import torch as _torch_compat  # registers safe globals on import
from core.utils.numpy_compat import ensure_numpy_legacy_aliases
ensure_numpy_legacy_aliases()
import smplx
import trimesh
import numpy as np
from imageio import v2 as imageio
from glob import glob
from torchvision.transforms import Normalize
from detectron2.config import LazyConfig
from core.utils.utils_detectron2 import DefaultPredictor_Lazy

from core.camerahmr_model import CameraHMR
from core.constants import CHECKPOINT_PATH, CAM_MODEL_CKPT, SMPL_MODEL_PATH, DETECTRON_CKPT, DETECTRON_CFG
from core.datasets.dataset import Dataset
from core.utils.renderer_pyrd import Renderer
from core.utils import recursive_to
from core.utils.geometry import batch_rot2aa
from core.cam_model.fl_net import FLNet
from core.constants import IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD, NUM_BETAS, DENSEKP_CKPT
from core.utils.geometry import rotmat_to_aa  # or batch_rot2aa
from core.utils.geometry import aa_to_rotmat
from types import SimpleNamespace

# 假设来自 CameraHMR 的输出：
# global_orient: (B, 1, 3, 3)
# body_pose:     (B, 23, 3, 3)
def smpl_rotmat_to_axis_angle(global_orient_mat, body_pose_mat):
    B = global_orient_mat.shape[0]
    go_aa = rotmat_to_aa(global_orient_mat.squeeze(1))                  # (B, 3)
    body_aa = rotmat_to_aa(body_pose_mat.reshape(-1, 3, 3)).reshape(B, 23, 3)  # (B, 23, 3)
    # 拼成 24*3 的轴角（SMPL）
    pose_aa = torch.cat([go_aa[:, None, :], body_aa], dim=1)            # (B, 24, 3)
    pose_aa_flat = pose_aa.reshape(B, 24 * 3)                           # (B, 72)
    return go_aa, body_aa, pose_aa, pose_aa_flat

def resize_image(img, target_size):
    height, width = img.shape[:2]
    aspect_ratio = width / height

    # Calculate the new size while maintaining the aspect ratio
    if aspect_ratio > 1:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_width = int(target_size * aspect_ratio)
        new_height = target_size

    # Resize the image using OpenCV
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a new blank image with the target size
    final_img = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255

    # Paste the resized image onto the blank image, centering it
    start_x = (target_size - new_width) // 2
    start_y = (target_size - new_height) // 2
    final_img[start_y:start_y + new_height, start_x:start_x + new_width] = resized_img

    return aspect_ratio, final_img

class HumanMeshEstimator:
    def __init__(self, smpl_model_path=SMPL_MODEL_PATH, threshold=0.25, mesh_opacity=0.3, same_mesh_color=False, save_smpl_obj=False, use_smplify=False, export_init_npz=None, model_path=None):
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.model_path = model_path
        self.model = self.init_model()
        self.detector = self.init_detector(threshold)
        self.cam_model = self.init_cam_model()
        self.smpl_model = smplx.SMPLLayer(model_path=smpl_model_path, num_betas=NUM_BETAS).to(self.device)
        self.normalize_img = Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
        self.mesh_opacity = mesh_opacity
        self.same_mesh_color = same_mesh_color
        self.save_smpl_obj = save_smpl_obj
        self.use_smplify = use_smplify
        self.smplify = None
        self.smplify_args = None
        self.densekp_model = None
        self.export_init_npz = export_init_npz
        self._init_records = [] if export_init_npz else None

        if self.use_smplify or self.export_init_npz:
            try:
                # Lazy import to avoid heavy deps if not requested
                from CamSMPLify.cam_smplify import SMPLify
                from CamSMPLify.constants import LOSS_CUT, LOW_THRESHOLD, HIGH_THRESHOLD
                from core.densekp_trainer import DenseKP
                # Initialize SMPLify
                self.smplify = SMPLify(vis=False, verbose=False, device=self.device)
                self.smplify_args = SimpleNamespace(
                    loss_cut=LOSS_CUT,
                    high_threshold=HIGH_THRESHOLD,
                    low_threshold=LOW_THRESHOLD,
                    vis_int=100,
                )
                # Load DenseKP for dense 2D keypoints required by SMPLify / exporter
                self.densekp_model = DenseKP.load_from_checkpoint(DENSEKP_CKPT, strict=False).to(self.device)
                self.densekp_model.eval()
            except Exception as e:
                print(f"Failed to initialize CamSMPLify / DenseKP; continuing without refinement: {e}")
                self.use_smplify = False

    def init_cam_model(self):
        model = FLNet()
        checkpoint = torch.load(CAM_MODEL_CKPT, weights_only=True)['state_dict']
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    def init_model(self):
        print("Test model_path:", self.model_path)
        model = CameraHMR.load_from_checkpoint(self.model_path if self.model_path else CHECKPOINT_PATH, strict=False)
        model = model.to(self.device)
        model.eval()
        return model

    def init_detector(self, threshold):

        detectron2_cfg = LazyConfig.load(str(DETECTRON_CFG))
        detectron2_cfg.train.init_checkpoint = DETECTRON_CKPT
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = threshold
        detector = DefaultPredictor_Lazy(detectron2_cfg)
        return detector


    def convert_to_full_img_cam(self, pare_cam, bbox_height, bbox_center, img_w, img_h, focal_length):
        s, tx, ty = pare_cam[:, 0], pare_cam[:, 1], pare_cam[:, 2]
        tz = 2. * focal_length / (bbox_height * s)
        cx = 2. * (bbox_center[:, 0] - (img_w / 2.)) / (s * bbox_height)
        cy = 2. * (bbox_center[:, 1] - (img_h / 2.)) / (s * bbox_height)
        cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
        return cam_t

    def get_output_mesh(self, params, pred_cam, batch):
        smpl_output = self.smpl_model(**{k: v.float() for k, v in params.items()})
        pred_keypoints_3d = smpl_output.joints
        pred_vertices = smpl_output.vertices
        img_h, img_w = batch['img_size'][0]
        cam_trans = self.convert_to_full_img_cam(
            pare_cam=pred_cam,
            bbox_height=batch['box_size'],
            bbox_center=batch['box_center'],
            img_w=img_w,
            img_h=img_h,
            focal_length=batch['cam_int'][:, 0, 0]
        )
        return pred_vertices, pred_keypoints_3d, cam_trans

    def get_cam_intrinsics(self, img):
        img_h, img_w, c = img.shape
        aspect_ratio, img_full_resized = resize_image(img, IMAGE_SIZE)
        img_full_resized = np.transpose(img_full_resized.astype('float32'),
                            (2, 0, 1))/255.0
        img_full_resized = self.normalize_img(torch.from_numpy(img_full_resized).float())

        estimated_fov, _ = self.cam_model(img_full_resized.unsqueeze(0))
        vfov = estimated_fov[0, 1]
        fl_h = (img_h / (2 * torch.tan(vfov / 2))).item()
        # fl_h = (img_w * img_w + img_h * img_h) ** 0.5
        cam_int = np.array([[fl_h, 0, img_w/2], [0, fl_h, img_h / 2], [0, 0, 1]]).astype(np.float32)
        return cam_int


    def remove_pelvis_rotation(self, smpl):
        """We don't trust the body orientation coming out of bedlam_cliff, so we're just going to zero it out."""
        smpl.body_pose[0][0][:] = np.zeros(3)


    def _process_and_render(self, img_cv2, dataset_imglabel, overlay_fname, mesh_fname):
        """Shared pipeline used by both image and video frames.

        Returns (out_smpl_params, output_cam_trans, focal_length_, img_h, img_w, batch)
        or (None, None, None, h, w, None) if no detections.
        """
        # Detect humans
        det_out = self.detector(img_cv2)
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        if boxes.shape[0] == 0:
            # No detections: write original image and exit
            cv2.imwrite(overlay_fname, img_cv2)
            h, w = img_cv2.shape[:2]
            return None, None, None, h, w, None, None
        bbox_scale = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
        bbox_center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0

        # Intrinsics and dataset
        cam_int = self.get_cam_intrinsics(img_cv2)
        dataset = Dataset(img_cv2, bbox_center, bbox_scale, cam_int, False, dataset_imglabel)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=10)

        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            img_h, img_w = batch['img_size'][0]
            with torch.no_grad():
                out_smpl_params, out_cam, focal_length_ = self.model(batch)

            output_vertices, output_joints, output_cam_trans = self.get_output_mesh(out_smpl_params, out_cam, batch)

            # DenseKP (unified) – compute once for this batch if available
            dense_kp_batch = None
            if self.densekp_model is not None:
                try:
                    with torch.no_grad():
                        dk_out = self.densekp_model(batch)
                        dense_kp_batch = dk_out['pred_keypoints']  # (B, K, 3)
                except Exception as e:
                    print(f"DenseKP inference failed: {e}")

            # Optional CamSMPLify refinement
            if self.use_smplify and self.smplify is not None and dense_kp_batch is not None:
                try:
                    go_aa, body_aa, pose_aa, pose_aa_flat = smpl_rotmat_to_axis_angle(
                        out_smpl_params['global_orient'], out_smpl_params['body_pose']
                    )

                    B = pose_aa_flat.shape[0]
                    refined_vertices = output_vertices.clone()
                    refined_cam_t = output_cam_trans.clone()

                    for bi in range(B):
                        try:
                            init_pose_flat = pose_aa_flat[bi].detach().cpu().numpy()[None, :]
                            init_betas = out_smpl_params['betas'][bi].detach().cpu().numpy()[None, :]
                            cam_t_init = refined_cam_t[bi].detach().cpu().numpy()
                            center = batch['box_center'][bi].detach().cpu().numpy()
                            scale = (batch['box_size'][bi].detach().cpu().item() / 200.0)
                            cam_int_np = batch['cam_int'][bi].detach().cpu().numpy()
                            imgname = batch['imgname'][bi] if isinstance(batch['imgname'], (list, tuple)) else batch['imgname']
                            dense_kp = dense_kp_batch[bi].detach().cpu().numpy()

                            result = self.smplify(self.smplify_args, init_pose_flat, init_betas, cam_t_init,
                                                   center, scale, cam_int_np, imgname,
                                                   joints_2d_=None, dense_kp=dense_kp, ind=bi)
                            if isinstance(result, dict) and result:
                                go_aa_ref = result['global_orient'].detach().view(1, 1, 3)
                                body_aa_ref = result['pose'].detach().view(1, 23, 3)
                                go_mat_ref = aa_to_rotmat(go_aa_ref.view(1, 3)).view(1, 1, 3, 3)
                                body_mat_ref = aa_to_rotmat(body_aa_ref.view(23, 3)).view(1, 23, 3, 3)
                                betas_ref = result['betas'].detach().view(1, -1)

                                params_refined = {
                                    'global_orient': go_mat_ref.to(self.device).float(),
                                    'body_pose': body_mat_ref.to(self.device).float(),
                                    'betas': betas_ref.to(self.device).float(),
                                }
                                smpl_out_ref = self.smpl_model(**params_refined)
                                refined_vertices[bi] = smpl_out_ref.vertices[0]
                                refined_cam_t[bi] = result['camera_translation'].detach().to(self.device)
                        except Exception as e:
                            print(f"CamSMPLify refinement failed for person {bi}: {e}")

                    output_vertices = refined_vertices
                    output_cam_trans = refined_cam_t
                except Exception as e:
                    print(f"CamSMPLify pipeline failed; rendering initial predictions. Error: {e}")

            if self.save_smpl_obj:
                try:
                    mesh = trimesh.Trimesh(output_vertices[0].cpu().numpy(), self.smpl_model.faces, process=False)
                    mesh.export(mesh_fname)
                except Exception as e:
                    print(f"Failed to export OBJ {mesh_fname}: {e}")

            # Render overlay
            if isinstance(batch['cam_int'], torch.Tensor):
                f_val = float(batch['cam_int'][0, 0, 0].detach().cpu().item())
            else:
                f_val = float(focal_length_[0]) if isinstance(focal_length_, torch.Tensor) else float(focal_length_)
            focal_length = (f_val, f_val)
            # FIX BUG IN CAMERA TRANS RENDERER
            # pred_vertices_array = (output_vertices + output_cam_trans.unsqueeze
            # (1)).detach().cpu().numpy()
            # Restore: bake translation in predicted vertices
            pred_vertices_array = (output_vertices + output_cam_trans.unsqueeze(1)).detach().cpu().numpy()
            renderer = Renderer(focal_length=focal_length[0], img_w=img_w, img_h=img_h, faces=self.smpl_model.faces, same_mesh_color=self.same_mesh_color, mesh_opacity=self.mesh_opacity)
            bg_img_rgb = cv2.cvtColor(img_cv2.copy(), cv2.COLOR_BGR2RGB)
            front_view_rgb = renderer.render_front_view(pred_vertices_array, bg_img_rgb=bg_img_rgb)
            final_img_bgr = cv2.cvtColor(front_view_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(overlay_fname, final_img_bgr)
            renderer.delete()

            return out_smpl_params, output_cam_trans, focal_length_, int(img_h), int(img_w), batch, dense_kp_batch

        # Should not reach here (dataloader yields at least one batch if boxes exist)
        h, w = img_cv2.shape[:2]
        return None, None, None, h, w, None, None
    def process_image(self, img_path, output_img_folder, output_cam_folder, i, no_id=True):
        img_cv2 = cv2.imread(str(img_path))
        print(img_path)
        if img_cv2 is None:
            try:
                # Fallback for formats not handled by OpenCV (e.g., GIF/WEBP)
                img = imageio.imread(str(img_path))
                if img.ndim == 2:
                    img = np.stack([img, img, img], axis=-1)
                if img.shape[2] == 4:
                    img = img[:, :, :3]
                # imageio returns RGB; convert to BGR for downstream detector which expects BGR
                img_cv2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"Skipping unreadable image: {img_path} ({e})")
                return

        fname, img_ext = os.path.splitext(os.path.basename(img_path))
        if no_id:
            suffix=''
        else:
            suffix=f'_{i:06d}'
        overlay_fname = os.path.join(output_img_folder, f'{os.path.basename(fname)}{suffix}{img_ext}')
        mesh_fname = os.path.join(output_img_folder, f'{os.path.basename(fname)}{suffix}.obj')
        if output_cam_folder:
            os.makedirs(output_cam_folder, exist_ok=True)

        out_smpl_params, output_cam_trans, focal_length_, img_h, img_w, batch, dense_kp_batch = self._process_and_render(
            img_cv2=img_cv2,
            dataset_imglabel=img_path,
            overlay_fname=overlay_fname,
            mesh_fname=mesh_fname,
        )
        # === Collect init params for offline CamSMPLify (.npz) ===
        if self.export_init_npz is not None and out_smpl_params is not None and output_cam_trans is not None:
            try:
                go_aa, body_aa, pose_aa, pose_aa_flat = smpl_rotmat_to_axis_angle(out_smpl_params['global_orient'], out_smpl_params['body_pose'])
                # DenseKP if available, else zeros
                if self.densekp_model is not None and batch is not None:
                    with torch.no_grad():
                        dk_out = self.densekp_model(batch)
                        dense_kp_batch = dk_out['pred_keypoints']  # (B, K, 3)
                else:
                    dense_kp_batch = None

                B = pose_aa_flat.shape[0]
                for bi in range(B):
                    record = {
                        'pose': pose_aa_flat[bi].detach().cpu().numpy().reshape(24, 3),
                        'shape': out_smpl_params['betas'][bi].detach().cpu().numpy(),
                        'cam_int': batch['cam_int'][bi].detach().cpu().numpy() if batch is not None else None,
                        'cam_t': output_cam_trans[bi].detach().cpu().numpy(),
                        'center': batch['box_center'][bi].detach().cpu().numpy() if batch is not None else None,
                        'scale': float(batch['box_size'][bi].detach().cpu().item()) if batch is not None else None,
                        'imgname': (batch['imgname'][bi] if isinstance(batch['imgname'], (list, tuple)) else batch['imgname']) if batch is not None else img_path,
                    }
                    if dense_kp_batch is not None:
                        record['dense_kp'] = dense_kp_batch[bi].detach().cpu().numpy()
                    else:
                        record['dense_kp'] = np.zeros((0, 3), dtype=np.float32)
                    self._init_records.append(record)
            except Exception as e:
                print(f"Exporter (init .npz) collection failed: {e}")

            # Rendering is handled inside _process_and_render

            # === Export SMPL-X-like params and camera for IDOL reconstruct (image mode) ===
            if output_cam_folder:
                try:
                    # Save camera JSON
                    def tensor_to_list(t, max_len=None):
                        if t is None:
                            return None
                        arr = t.detach().float().cpu().numpy()#.reshape(-1)
                        if max_len is not None:
                            arr = arr[:max_len]
                        return arr.tolist()

                    # Convert to JSON-compatible axis-angle and keep first 21 body joints -> 21*3
                    go_aa, body_aa, pose_aa, pose_aa_flat = smpl_rotmat_to_axis_angle(
                        out_smpl_params['global_orient'], out_smpl_params['body_pose']
                    )
                    # body_aa corresponds to SMPL joints 1..23 → take first 21 joints directly
                    body_aa_21 = body_aa[:, :21, :]

                    global_orient = tensor_to_list(go_aa.squeeze(0)) or [0.0, 0.0, 0.0]
                    body_pose = tensor_to_list(body_aa_21.squeeze(0)) or [[0.0, 0.0, 0.0] for _ in range(21)] #21*3
                    betas_raw = tensor_to_list(out_smpl_params.get('betas').squeeze(0)) or [0.0] * 10
                    betas_save = (betas_raw + [0.0] * 10)[:10]

                    cam_t = tensor_to_list(output_cam_trans[0].squeeze(0)) if isinstance(output_cam_trans, torch.Tensor) else [0.0, 0.0, 2.0]
                    camera_R = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                    camera_t = cam_t[:3] if cam_t is not None else [0.0, 0.0, 2.0]
                    # Ensure Python float (CPU) for JSON
                    if isinstance(focal_length_, torch.Tensor):
                        f_est = float(focal_length_[0].detach().cpu().item())
                    else:
                        f_est = float(focal_length_)
                    camera = {
                        "R": camera_R,
                        "t": camera_t,
                        "focal": [f_est, f_est],
                        "princpt": [img_w / 2.0, img_h / 2.0],
                    }

                    zeros_hand = [0.0] * 45
                    zeros3 = [0.0, 0.0, 0.0]
                    zeros_expr10 = [0.0] * 10

                    idol_json = {
                        "camera": camera,
                        "root_pose": global_orient,
                        "body_pose": body_pose,
                        "betas_save": betas_save,
                        "nlf_smplx_betas": betas_save,
                        "lhand_pose": zeros_hand,
                        "rhand_pose": zeros_hand,
                        "jaw_pose": zeros3,
                        "leye_pose": zeros3,
                        "reye_pose": zeros3,
                        "expr": zeros_expr10,
                        "trans": [0.0, 0.0, 0.0],
                    }

                    id_json_path = os.path.join(output_cam_folder, f"{os.path.basename(fname)}{suffix}.json")
                    with open(id_json_path, 'w') as f:
                        json.dump(idol_json, f)

                except Exception as e:
                    print(f"Failed to export Camera info for IDOL inputs {fname}: {e}")


    def run_on_images(self, image_folder, out_folder, out_cam_folder):
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']
        images_list = [image for ext in image_extensions for image in glob(os.path.join(image_folder, ext))]
        for ind, img_path in enumerate(images_list):
            self.process_image(img_path, out_folder, out_cam_folder, ind)
        # Save collected init .npz
        if self.export_init_npz and self._init_records:
            try:
                import numpy as _np
                out_dir = os.path.dirname(self.export_init_npz)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                out = {k: [] for k in ['pose','shape','cam_int','cam_t','center','scale','imgname','dense_kp']}
                for rec in self._init_records:
                    for k in out.keys():
                        if k in rec:
                            out[k].append(rec[k])
                # Stack appropriately
                out_np = {}
                for k, v in out.items():
                    if len(v) == 0:
                        continue
                    if k == 'imgname':
                        out_np[k] = _np.array(v)
                    else:
                        out_np[k] = _np.array(v, dtype=_np.float32)
                _np.savez(self.export_init_npz, **out_np)
                print(f"Saved init params to {self.export_init_npz}")
            except Exception as e:
                print(f"Failed to save init .npz: {e}")

    def process_frame(self, frame_bgr, output_img_folder, frame_index):
        img_cv2 = frame_bgr
        fname = f"frame_{frame_index:06d}"
        img_ext = ".png"
        overlay_fname = os.path.join(output_img_folder, f'{fname}{img_ext}')
        mesh_fname = os.path.join(output_img_folder, f'{fname}.obj')

        out_smpl_params, output_cam_trans, focal_length_, img_h, img_w, batch, dense_kp_batch = self._process_and_render(
            img_cv2=img_cv2,
            dataset_imglabel=f"frame_{frame_index}",
            overlay_fname=overlay_fname,
            mesh_fname=mesh_fname,
        )
        # === Export SMPL-X-like params and camera for IDOL reconstruct ===
        try:
            idol_root = "/home/cevin/Meitu/IDOL"
            idol_img_dir = os.path.join(idol_root, "test_data_img", "all")
            os.makedirs(idol_img_dir, exist_ok=True)

            # Save the raw frame for IDOL input (even-sized JPEG)
            id_img_path = os.path.join(idol_img_dir, f"{fname}.jpg")
            h, w = img_cv2.shape[:2]
            pad_h = h % 2
            pad_w = w % 2
            if pad_h != 0 or pad_w != 0:
                img_to_save = cv2.copyMakeBorder(img_cv2, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
            else:
                img_to_save = img_cv2
            cv2.imwrite(id_img_path, img_to_save)

            # Build SMPL/SMPL-X parameter JSON expected by IDOL's load_smplify_json
            def tensor_to_list(t, max_len=None):
                if t is None:
                    return None
                arr = t.detach().float().cpu().numpy().reshape(-1)
                if max_len is not None:
                    arr = arr[:max_len]
                return arr.tolist()

            go_aa, body_aa, pose_aa, pose_aa_flat = smpl_rotmat_to_axis_angle(
                out_smpl_params['global_orient'], out_smpl_params['body_pose']
            )
            body_aa_21 = body_aa[:, :21, :]
            global_orient = tensor_to_list(go_aa) or [0.0, 0.0, 0.0]
            body_pose = tensor_to_list(body_aa_21.reshape(-1)) or [0.0] * 63
            betas_raw = tensor_to_list(out_smpl_params.get('betas')) or [0.0] * 10
            betas_save = (betas_raw + [0.0] * 10)[:10]

            cam_t = tensor_to_list(output_cam_trans[0]) if isinstance(output_cam_trans, torch.Tensor) else [0.0, 0.0, 2.0]
            camera_R = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            camera_t = cam_t[:3] if cam_t is not None else [0.0, 0.0, 2.0]
            f_est = float(focal_length_[0]) if isinstance(focal_length_, torch.Tensor) else float(focal_length_)
            camera = {
                "R": camera_R,
                "t": camera_t,
                "focal": [f_est, f_est],
                "princpt": [img_w / 2.0, img_h / 2.0],
            }

            zeros_hand = [0.0] * 45
            zeros3 = [0.0, 0.0, 0.0]
            zeros_expr10 = [0.0] * 10

            idol_json = {
                "camera": camera,
                "root_pose": global_orient,
                "body_pose": body_pose,
                "betas_save": betas_save,
                "lhand_pose": zeros_hand,
                "rhand_pose": zeros_hand,
                "jaw_pose": zeros3,
                "leye_pose": zeros3,
                "reye_pose": zeros3,
                "expr": zeros_expr10,
                "trans": [0.0, 0.0, 0.0],
            }

            id_json_path = os.path.join(idol_img_dir, f"{fname}.json")
            with open(id_json_path, 'w') as f:
                json.dump(idol_json, f)

        except Exception as e:
            print(f"Failed to export IDOL inputs for {fname}: {e}")

    def _export_video_from_frames(self, frames_dir, video_out_path, fps):
        pattern = os.path.join(frames_dir, 'frame_%06d.png')
        os.makedirs(os.path.dirname(video_out_path), exist_ok=True)
        ffmpeg_path = shutil.which('ffmpeg')
        fps_value = int(round(fps)) if fps and fps > 0 else 30
        if ffmpeg_path:
            cmd = [
                ffmpeg_path,
                '-y',
                '-framerate', str(fps_value),
                '-start_number', '0',
                '-i', pattern,
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                video_out_path
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return
            except Exception as e:
                print(f"ffmpeg failed ({e}), falling back to OpenCV VideoWriter...")

        # Fallback: OpenCV VideoWriter
        frame_paths = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.startswith('frame_') and f.endswith('.png')])
        if not frame_paths:
            print(f"No frames found in {frames_dir} to make a video.")
            return
        first = cv2.imread(frame_paths[0])
        if first is None:
            print(f"Failed to read first frame {frame_paths[0]}")
            return
        height, width = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_out_path, fourcc, float(fps_value), (width, height))
        for fp in frame_paths:
            img = cv2.imread(fp)
            if img is None:
                continue
            writer.write(img)
        writer.release()

    def run_on_video(self, video_path, out_folder_root=None, export_video=True):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        if out_folder_root is None or len(str(out_folder_root).strip()) == 0:
            out_folder = os.path.join(os.path.dirname(video_path), video_basename)
        else:
            out_folder = os.path.join(out_folder_root, video_basename)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_index = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                self.process_frame(frame, out_folder, frame_index)
                frame_index += 1
        finally:
            cap.release()
        if export_video:
            video_out_path = os.path.join(out_folder, f"{video_basename}_overlay.mp4")
            self._export_video_from_frames(out_folder, video_out_path, fps)
