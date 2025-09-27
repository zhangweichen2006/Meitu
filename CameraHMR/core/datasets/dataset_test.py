
import os
import cv2
import torch
import copy
from ..utils.numpy_compat import ensure_numpy_legacy_aliases
ensure_numpy_legacy_aliases()
import numpy as np
from torch.utils.data import Dataset
from ..utils.pylogger import get_pylogger
from ..configs import DATASET_FOLDERS, DATASET_FILES, SAPIENS_TRAINING_PROCESS_NORMAL_VERSION, SAPIENS_TEST_PROCESS_NORMAL_VERSION, SAPIENS_TRAINING_PROCESS_NORMAL_VERSION2, SAPIENS_TEST_PROCESS_NORMAL_VERSION2, SAPIENS_TRAINING_IMGMATCH_NORMAL_VERSION, SAPIENS_TEST_IMGMATCH_NORMAL_VERSION, NORMAL_PREPROCESS
from .utils import expand_to_aspect_ratio, get_example, resize_image, revert_or_regen_sapiens_normals
from torchvision.transforms import Normalize
from ..constants import FLIP_KEYPOINT_PERMUTATION, NUM_JOINTS

log = get_pylogger(__name__)


class DatasetTest(Dataset):
    def __init__(self, cfg, dataset, is_train=False, version='test', device='cuda'):
        super(DatasetTest, self).__init__()

        self.dataset = dataset
        self.is_train = is_train
        self.version = version
        self.cfg = cfg
        self.device = device
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

        self.check_file_completeness_and_filter = cfg.DATASETS.get('check_file_completeness_and_filter', False)

        self.img_dir = DATASET_FOLDERS[dataset]
        # if self.data not exists
        if not os.path.exists(DATASET_FILES[dataset]):
            self.data = {}
            self.imgname = []
            for root, dirs, files in os.walk(self.img_dir):
                for file in files:
                    if file.endswith('.jpg') or file.endswith('.png'):
                        self.imgname.append(os.path.join(root, file))
            self.imgname = np.array(self.imgname).tolist()

        else:
            self.data = np.load(DATASET_FILES[dataset], allow_pickle=True)
            self.imgname = self.data['imgname']

        # Resolve normal paths (test uses processed and imgmatch variants)
        self.sapiens_normal_version = SAPIENS_TRAINING_PROCESS_NORMAL_VERSION if "training-images" in self.img_dir else SAPIENS_TEST_PROCESS_NORMAL_VERSION
        self.sapiens_normal_version2 = SAPIENS_TRAINING_PROCESS_NORMAL_VERSION2 if "training-images" in self.img_dir else SAPIENS_TEST_PROCESS_NORMAL_VERSION2
        self.sapiens_normal_imgmatch_version = SAPIENS_TRAINING_IMGMATCH_NORMAL_VERSION if "training-images" in self.img_dir else SAPIENS_TEST_IMGMATCH_NORMAL_VERSION
        self.replace_src_folder = "training-images" if "training-images" in self.img_dir else "test-images"

        # Build image paths and optionally filter missing files
        self.img_paths = [os.path.join(self.img_dir, str(p)) for p in self.imgname]
        if self.check_file_completeness_and_filter:
            valid_paths = np.array([os.path.isfile(p) for p in self.img_paths])
            if not valid_paths.all():
                num_missing = int((~valid_paths).sum())
                log.warning(f"{self.dataset}: {num_missing} missing images. Skipping those samples.")
                self.imgname = np.array(self.imgname)[valid_paths].tolist()
                self.data = {k: v[valid_paths] for k, v in self.data.items()}
                self.img_paths = np.array(self.img_paths)[valid_paths].tolist()

        # Normal modality configuration
        if 'sapiens_normals_folder' in self.data and 'normal_swapHW' in self.data and 'normal_preprocess' in self.data:
            sapiens_normals_folder, sapiens_normals_folder2 = self.data['sapiens_normals_folder']
            self.normal_swapHW = self.data['normal_swapHW']
            self.normal_preprocess = self.data['normal_preprocess']
        else:
            self.normal_preprocess = NORMAL_PREPROCESS.get(self.dataset, {}).get('preprocess', 'resize')
            self.normal_swapHW = NORMAL_PREPROCESS.get(self.dataset, {}).get('swapHW', False)

        # Precompute potential normal file paths (processed and imgmatch)
        self.sapiens_normals_path = [self.img_to_normals_path(i, self.replace_src_folder, self.sapiens_normal_version) for i in self.img_paths]
        self.sapiens_normals_path2 = [self.img_to_normals_path(i, self.replace_src_folder, self.sapiens_normal_version2) for i in self.img_paths]
        self.sapiens_normals_path_imgmatch = [self.img_to_normals_path(i, self.replace_src_folder, self.sapiens_normal_imgmatch_version) for i in self.img_paths]

        # Ensure normals exist; if missing, try revert or regenerate using shared utils
        normals_exist_mask = np.array([os.path.isfile(p) for p in self.sapiens_normals_path_imgmatch]) if len(self.sapiens_normals_path_imgmatch) > 0 else np.zeros(len(self.img_paths), dtype=bool)
        if not normals_exist_mask.all():
            num_missing = int((~normals_exist_mask).sum())
            log.warning(f"{self.dataset}: {num_missing} missing normals detected. Attempting revert/regeneration...")
            img_to_fix = np.array(self.img_paths)[~normals_exist_mask].tolist()
            normals_imgmatch_to_fix = np.array(self.sapiens_normals_path_imgmatch)[~normals_exist_mask].tolist()
            # best-effort: call revert/regenerate
            try:
                revert_or_regen_sapiens_normals(
                    img_to_fix,
                    sapiens_normals_path_to_regenerate=[],
                    sapiens_normals_path_to_regenerate_imgmatch=normals_imgmatch_to_fix,
                    sapiens_normals_path=self.sapiens_normals_path,
                    sapiens_normals_path2=self.sapiens_normals_path2,
                    replace_src_folder=self.replace_src_folder,
                    sapiens_normal_version=self.sapiens_normal_version,
                    sapiens_normal_version2=self.sapiens_normal_version2,
                    normal_swapHW=self.normal_swapHW,
                    normal_preprocess=self.normal_preprocess,
                    log=log,
                    dataset=self.dataset,
                    cfg=self.cfg,
                    device=getattr(self, 'device', 'cuda'),
                )
            except Exception as e:
                log.warning(f"{self.dataset}: revert/regenerate failed with error: {e}")
            # re-check after attempt
            normals_exist_mask = np.array([os.path.isfile(p) for p in self.sapiens_normals_path_imgmatch]) if len(self.sapiens_normals_path_imgmatch) > 0 else np.zeros(len(self.img_paths), dtype=bool)
        self.enable_normals = bool(normals_exist_mask.all())
        if not self.enable_normals:
            missing = int((~normals_exist_mask).sum())
            if missing > 0:
                log.warning(f"{self.dataset}: normals still missing for {missing} samples. Proceeding without normals.")

        # Scalar/meta fields
        if 'scale' in self.data:
            self.scale = self.data['scale']
        if 'center' in self.data:
            self.center = self.data['center']
        if 'gtkps' in self.data:
            self.keypoints = self.data['gtkps'][:, :NUM_JOINTS]
        elif 'part' in self.data:
            self.keypoints = self.data['part']
        elif 'body_keypoints_2d' in self.data:
            self.keypoints = self.data['body_keypoints_2d']
        else:
            self.keypoints = np.zeros((len(self.imgname), NUM_JOINTS, 3), dtype=np.float32)
        if self.keypoints.shape[2] < 3:
            ones_array = np.ones((self.keypoints.shape[0], self.keypoints.shape[1], 1))
            self.keypoints = np.concatenate((self.keypoints, ones_array), axis=2)

        if 'cam_int' in self.data:
            self.cam_int = self.data['cam_int']
        else:
            self.cam_int = np.zeros((len(self.imgname), 3, 3), dtype=np.float32)

        try:
            gender = self.data['gender']
            self.gender = np.array([0 if str(g) == 'm' or str(g) == 'male' else 1 for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1 * np.ones(len(self.imgname)).astype(np.int32)

        self.length = self.gender.shape[0]
        log.info(f'Loaded {self.dataset} dataset, num samples {self.length}')

    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()
        keypoints_2d = self.keypoints[index].copy()
        orig_keypoints_2d = self.keypoints[index].copy()
        center_x = center[0]
        center_y = center[1]
        bbox_size = expand_to_aspect_ratio(scale * 200, target_aspect_ratio=self.BBOX_SHAPE).max()
        if bbox_size < 1:
            breakpoint()

        augm_config = copy.deepcopy(self.cfg.DATASETS.CONFIG)
        imgname = os.path.join(self.img_dir, self.imgname[index])
        cv_img = cv2.imread(imgname, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        cv_img = cv_img[:, :, ::-1]
        aspect_ratio, img_full_resized = resize_image(cv_img, 256)
        img_full_resized = np.transpose(img_full_resized.astype('float32'), (2, 0, 1)) / 255.0
        item['img_full_resized'] = self.normalize_img(torch.from_numpy(img_full_resized).float())

        # Prepare normal path if enabled
        normal_path_arg = None
        if self.enable_normals:
            normal_path_arg = self.sapiens_normals_path_imgmatch[index]

        img_patch_rgba, \
        img_patch_cv, \
        keypoints_2d, \
        img_size, cx, cy, bbox_w, bbox_h, trans, scale_aug, normal_patch = get_example(
            imgname,
            center_x, center_y,
            bbox_size, bbox_size,
            keypoints_2d,
            FLIP_KEYPOINT_PERMUTATION,
            self.IMG_SIZE, self.IMG_SIZE,
            self.MEAN, self.STD, self.is_train, augm_config,
            is_bgr=True, return_trans=True,
            use_skimage_antialias=self.use_skimage_antialias,
            border_mode=self.border_mode,
            dataset=self.dataset,
            normal_path=normal_path_arg,
            normal_swapHW=self.normal_swapHW,
            normal_preprocess=self.normal_preprocess
        )

        img_w = img_size[1]
        img_h = img_size[0]
        if self.cam_int.size and self.cam_int.ndim == 3 and self.cam_int.shape[1:] == (3, 3):
            item['cam_int'] = np.array(self.cam_int[index]).astype(np.float32)
        else:
            fl = 5000
            item['cam_int'] = np.array([[fl, 0, img_w / 2.], [0, fl, img_h / 2.], [0, 0, 1]]).astype(np.float32)

        new_center = np.array([cx, cy])
        img_patch = img_patch_rgba[:3, :, :]
        item['img'] = img_patch
        item['img_disp'] = img_patch_cv
        if self.enable_normals and normal_patch is not None:
            item['normal'] = normal_patch
        item['keypoints_2d'] = keypoints_2d.astype(np.float32)
        item['orig_keypoints_2d'] = orig_keypoints_2d
        item['box_center'] = new_center
        item['box_size'] = bbox_w * scale_aug
        item['img_size'] = 1.0 * img_size.copy()
        item['_scale'] = scale
        item['_trans'] = trans
        item['imgname'] = imgname
        item['dataset'] = self.dataset
        item['gender'] = self.gender[index]
        return item

    def __len__(self):
        return int(len(self.imgname))

    # Helpers to map an image path to normal file paths
    def img_to_normals_path(self, img_path, replace_src, replace_tgt):
        base1 = os.path.normpath(img_path.replace(replace_src, replace_tgt))
        root1, _ = os.path.splitext(base1)
        return root1 + '.npy'

