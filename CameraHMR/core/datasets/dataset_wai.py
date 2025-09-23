import os
import cv2
import copy
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.pylogger import get_pylogger
from ..constants import (
    FLIP_KEYPOINT_PERMUTATION,
    NUM_JOINTS,
    NUM_BETAS,
    NUM_PARAMS_SMPL,
    NUM_PARAMS_SMPLX,
    SMPL_MODEL_DIR,
    SMPLX_MODEL_DIR,
)
from .utils import expand_to_aspect_ratio, get_example, resize_image
from torchvision.transforms import Normalize


log = get_pylogger(__name__)


def _safe_get_cam_intrinsics(scene_meta: dict, frame: dict) -> np.ndarray:
    """Build pinhole intrinsics matrix from scene/frame meta."""
    fx = float(frame.get('fl_x', scene_meta.get('fl_x', 0.0)))
    fy = float(frame.get('fl_y', scene_meta.get('fl_y', 0.0)))
    cx = float(frame.get('cx', scene_meta.get('cx', 0.0)))
    cy = float(frame.get('cy', scene_meta.get('cy', 0.0)))
    if fx == 0.0 or fy == 0.0:
        # Fallback to square focal using image size if missing
        w = int(frame.get('w', scene_meta.get('w', 0)))
        h = int(frame.get('h', scene_meta.get('h', 0)))
        fx = fy = float(max(w, h))
        cx = w / 2.0
        cy = h / 2.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return K


def _iter_wai_frames(root_dir: str, dataset_name: str, scene_list: list[str] | None):
    """Yield tuples of (scene_name, frame_dict) for all frames under a WAI dataset."""
    dataset_root = os.path.join(root_dir, dataset_name)
    if scene_list is None:
        # discover scenes by presence of scene_meta.json
        try:
            scene_list = [d for d in os.listdir(dataset_root)
                          if os.path.isfile(os.path.join(dataset_root, d, 'scene_meta.json'))]
        except FileNotFoundError:
            scene_list = []

    for scene_name in scene_list:
        scene_dir = os.path.join(dataset_root, scene_name)
        meta_path = os.path.join(scene_dir, 'scene_meta.json')
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path, 'r') as f:
                scene_meta = json.load(f)
        except Exception as e:
            log.warning(f"Skipping scene {scene_name}: failed to read meta {e}")
            continue
        frames = scene_meta.get('frames', [])
        frame_names_map = scene_meta.get('frame_names', {})
        # invert if needed (name->index mapping often present)
        for frame in frames:
            # ensure width/height exist
            if 'w' not in frame:
                frame['w'] = scene_meta.get('w', None)
            if 'h' not in frame:
                frame['h'] = scene_meta.get('h', None)
            # ensure file path present (prefer explicit 'image', else 'file_path')
            image_rel = frame.get('image', frame.get('file_path', None))
            if image_rel is None:
                # try to reconstruct from frame_name convention
                fname = str(frame.get('frame_name', ''))
                image_rel = os.path.join('images', f"{fname}.png")
                frame['image'] = image_rel
            # ensure frame_name available
            if 'frame_name' not in frame:
                # try reverse lookup by file path
                inv = {v: k for k, v in frame_names_map.items()} if frame_names_map else {}
                frame['frame_name'] = inv.get(image_rel, image_rel)
            # pack scene_meta reference for later intrinsics fallback
            frame['_scene_meta'] = scene_meta
            yield scene_name, frame


class DatasetWAI(Dataset):
    """
    WAI-format dataset adapter for CameraHMR.

    Usage: select datasets in config as e.g. 'wai:eth3d' and set
    DATASETS.WAI_ROOT to the directory containing WAI datasets (with subfolder 'eth3d').
    Optionally set DATASETS.WAI_METADATA_DIR to a directory that contains aggregated
    scene lists like '<split>/<dataset_name>_scene_list_<split>.npy'.
    """

    def __init__(self, cfg, dataset: str, version: str = 'test', is_train: bool = False):
        super(DatasetWAI, self).__init__()

        # Parse dataset identifier: expect 'wai:<dataset_name>' or raw dataset name
        if dataset.startswith('wai:'):
            wai_name = dataset.split(':', 1)[1]
        else:
            wai_name = dataset

        self.cfg = cfg
        self.wai_dataset_name = wai_name
        self.version = 'train' if version == 'train' else 'test'
        self.is_train = is_train

        # Image normalization and cropping params
        self.IMG_SIZE = cfg.MODEL.IMAGE_SIZE
        self.BBOX_SHAPE = cfg.MODEL.get('BBOX_SHAPE', None)
        self.MEAN = 255. * np.array(cfg.MODEL.IMAGE_MEAN)
        self.STD = 255. * np.array(cfg.MODEL.IMAGE_STD)
        self.normalize_img = Normalize(mean=cfg.MODEL.IMAGE_MEAN,
                                       std=cfg.MODEL.IMAGE_STD)
        self.use_skimage_antialias = cfg.DATASETS.get('USE_SKIMAGE_ANTIALIAS', False)
        self.border_mode = {
            'constant': cv2.BORDER_CONSTANT,
            'replicate': cv2.BORDER_REPLICATE,
        }[cfg.DATASETS.get('BORDER_MODE', 'constant')]

        # Resolve WAI root and optional metadata dir
        self.wai_root = cfg.DATASETS.get('WAI_ROOT', os.environ.get('WAI_ROOT', None))
        self.wai_metadata_dir = cfg.DATASETS.get('WAI_METADATA_DIR', os.environ.get('WAI_METADATA_DIR', None))
        if self.wai_root is None:
            raise ValueError("WAI_ROOT not provided. Set cfg.DATASETS.WAI_ROOT or env WAI_ROOT")

        # Load scene list if available for split, else discover
        scene_list = None
        if self.wai_metadata_dir is not None:
            split_dir = os.path.join(self.wai_metadata_dir, self.version)
            npy_path = os.path.join(split_dir, f"{self.wai_dataset_name}_scene_list_{self.version}.npy")
            if os.path.isfile(npy_path):
                try:
                    scene_list = np.load(npy_path, allow_pickle=True).tolist()
                except Exception as e:
                    log.warning(f"Failed to load scene list from {npy_path}: {e}")

        # Aggregate per-frame entries
        img_relpaths = []
        centers = []
        scales = []
        cam_ints = []
        cam_exts = []
        widths = []
        heights = []

        for scene_name, frame in _iter_wai_frames(self.wai_root, self.wai_dataset_name, scene_list):
            scene_dir = os.path.join(self.wai_root, self.wai_dataset_name, scene_name)
            image_rel = frame.get('image', frame.get('file_path'))
            if image_rel is None:
                continue
            rel_path = os.path.join(scene_name, image_rel)

            # width/height and bbox defaults
            w = int(frame.get('w') or 0)
            h = int(frame.get('h') or 0)
            if (w == 0 or h == 0):
                # try to read the image to get size (slow)
                img_full_path = os.path.join(self.wai_root, self.wai_dataset_name, rel_path)
                if os.path.isfile(img_full_path):
                    im = cv2.imread(img_full_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                    if isinstance(im, np.ndarray):
                        h, w = im.shape[:2]
            if w == 0 or h == 0:
                # skip if cannot determine size
                log.warning(f"Skipping frame without size: {rel_path}")
                continue

            cx = w / 2.0
            cy = h / 2.0
            scale = max(w, h) / 200.0

            # intrinsics/extrinsics
            scene_meta = frame.get('_scene_meta', {})
            K = _safe_get_cam_intrinsics(scene_meta, frame)
            # cam2world -> world2cam for ext
            T_c2w = np.array(frame.get('transform_matrix', np.eye(4))).reshape(4, 4)
            try:
                T_w2c = np.linalg.inv(T_c2w)
            except np.linalg.LinAlgError:
                T_w2c = np.eye(4)

            img_relpaths.append(rel_path)
            centers.append(np.array([cx, cy], dtype=np.float32))
            scales.append(scale)
            cam_ints.append(K.astype(np.float32))
            cam_exts.append(T_w2c.astype(np.float32))
            widths.append(w)
            heights.append(h)

        # Convert to numpy arrays and store
        self.img_dir = os.path.join(self.wai_root, self.wai_dataset_name)
        self.imgname = np.array(img_relpaths)
        self.center = np.array(centers, dtype=np.float32)
        self.scale = np.array(scales, dtype=np.float32)
        self.cam_int = np.array(cam_ints, dtype=np.float32)
        self.cam_ext = np.array(cam_exts, dtype=np.float32)

        # Default annotations (not available in WAI): SMPL pose/betas, keypoints, gender
        self.pose = np.zeros((len(self.imgname), NUM_PARAMS_SMPL * 3), dtype=np.float32)
        self.betas = np.zeros((len(self.imgname), NUM_BETAS), dtype=np.float32)
        self.keypoints = np.zeros((len(self.imgname), NUM_JOINTS, 3), dtype=np.float32)
        self.gender = -1 * np.ones(len(self.imgname), dtype=np.int32)

        # Filter out entries where the corresponding image file doesn't exist
        img_paths = [os.path.join(self.img_dir, str(p)) for p in self.imgname]
        valid_paths = np.array([os.path.isfile(p) for p in img_paths])
        if not valid_paths.all():
            num_missing = int((~valid_paths).sum())
            log.warning(f"WAI {self.wai_dataset_name}: {num_missing} missing images. Skipping those samples.")
            # Apply mask to all per-sample arrays
            self.imgname = self.imgname[valid_paths]
            self.scale = self.scale[valid_paths]
            self.center = self.center[valid_paths]
            self.pose = self.pose[valid_paths]
            self.betas = self.betas[valid_paths]
            self.cam_int = self.cam_int[valid_paths]
            self.cam_ext = self.cam_ext[valid_paths]
            self.keypoints = self.keypoints[valid_paths]

        self.length = int(self.scale.shape[0])
        log.info(f"Loaded WAI dataset '{self.wai_dataset_name}' split={self.version}, num samples {self.length}")

    def __len__(self):
        return int(len(self.imgname))

    def __getitem__(self, index: int):
        item = {}

        scale = float(self.scale[index])
        center = self.center[index].copy()
        keypoints_2d = self.keypoints[index].copy()
        orig_keypoints_2d = self.keypoints[index].copy()

        center_x = float(center[0])
        center_y = float(center[1])
        bbox_size = expand_to_aspect_ratio(scale * 200, target_aspect_ratio=self.BBOX_SHAPE).max()
        if bbox_size < 1:
            bbox_size = max(1.0, scale * 200)

        augm_config = copy.deepcopy(self.cfg.DATASETS.CONFIG)
        imgname = os.path.join(self.img_dir, self.imgname[index])

        # Full-res resized image for camera head
        cv_img = cv2.imread(imgname, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        cv_img = cv_img[:, :, ::-1]
        _, img_full_resized = resize_image(cv_img, 256)
        img_full_resized = np.transpose(img_full_resized.astype('float32'), (2, 0, 1)) / 255.0
        item['img_full_resized'] = self.normalize_img(torch.from_numpy(img_full_resized).float())

        # SMPL params (zeros by default for WAI)
        item['pose'] = self.pose[index]
        item['betas'] = self.betas[index]
        smpl_params = {
            'global_orient': self.pose[index][:3].astype(np.float32),
            'body_pose': self.pose[index][3:].astype(np.float32),
            'betas': self.betas[index].astype(np.float32),
        }
        item['smpl_params'] = smpl_params

        # Translation from extrinsics if needed (not used by default losses)
        item['translation'] = self.cam_ext[index][:, 3]

        img_patch_rgba, img_patch_cv, keypoints_2d, img_size, cx, cy, bbox_w, bbox_h, trans, scale_aug = get_example(
            imgname,
            center_x,
            center_y,
            bbox_size,
            bbox_size,
            keypoints_2d,
            FLIP_KEYPOINT_PERMUTATION,
            self.IMG_SIZE,
            self.IMG_SIZE,
            self.MEAN,
            self.STD,
            self.is_train,
            augm_config,
            is_bgr=True,
            return_trans=True,
            use_skimage_antialias=self.use_skimage_antialias,
            border_mode=self.border_mode,
            dataset=self.wai_dataset_name,
        )

        # Camera intrinsics
        item['cam_int'] = np.array(self.cam_int[index]).astype(np.float32)

        new_center = np.array([cx, cy])
        img_patch = img_patch_rgba[:3, :, :]
        item['img'] = img_patch
        item['img_disp'] = img_patch_cv
        item['keypoints_2d'] = keypoints_2d.astype(np.float32)
        item['orig_keypoints_2d'] = orig_keypoints_2d
        item['box_center'] = new_center
        item['box_size'] = bbox_w * scale_aug
        item['img_size'] = 1.0 * img_size.copy()
        item['_scale'] = scale
        item['_trans'] = trans
        item['imgname'] = imgname
        item['dataset'] = f"wai-{self.wai_dataset_name}"
        item['gender'] = -1

        return item


