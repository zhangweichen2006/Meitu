
import os
import cv2
import torch
import copy
import numpy as np
from torch.utils.data import Dataset
from ..utils.pylogger import get_pylogger
from ..configs import DATASET_FOLDERS, DATASET_FILES
from ..constants import FLIP_KEYPOINT_PERMUTATION, NUM_JOINTS, NUM_BETAS, NUM_PARAMS_SMPL
from .utils import expand_to_aspect_ratio, get_example, resize_image
from torchvision.transforms import Normalize
log = get_pylogger(__name__)


class DatasetTrain(Dataset):
    def __init__(self, cfg, dataset, is_train=True):
        super(DatasetTrain, self).__init__()

        self.dataset = dataset
        self.is_train = is_train
        self.cfg = cfg
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

        self.check_file_completeness = cfg.DATASETS.get('CHECK_FILE_COMPLETENESS', False)

        self.img_dir = DATASET_FOLDERS[dataset]
        self.data = np.load(DATASET_FILES[is_train][dataset], allow_pickle=True)
        self.imgname = self.data['imgname']
        self.scale = self.data['scale']
        self.center = self.data['center']
        if 'pose_cam' in self.data:
            self.body_pose = self.data['pose_cam'][:, :NUM_PARAMS_SMPL*3].astype(np.float) #Change 24
        elif 'pose' in self.data:
            self.body_pose = self.data['pose'][:, :NUM_PARAMS_SMPL*3].astype(np.float) #Change 24
        else:
            self.body_pose = np.zeros((len(self.imgname), NUM_PARAMS_SMPL*3), dtype=np.float32)

        if self.body_pose.shape[1] == NUM_PARAMS_SMPL:
            self.body_pose = self.body_pose.reshape(-1,NUM_PARAMS_SMPL*3)
        self.betas = self.data['shape'].astype(np.float)[:,:NUM_BETAS]
        self.cam_int = self.data['cam_int']
        self.keypoints = self.data['gtkps'][:,:NUM_JOINTS]
        self.length = self.scale.shape[0]
        if 'cam_ext' in self.data:
            self.cam_ext = self.data['cam_ext']
        else:
            self.cam_ext = np.zeros((self.imgname.shape[0], 4, 4))

        #Only for BEDLAM and AGORA
        if 'trans_cam' in self.data:
            self.trans_cam = self.data['trans_cam']
        else:
            self.trans_cam = np.zeros((self.imgname.shape[0],3))
        log.info(f'Loaded {self.dataset} dataset, num samples {self.length}')

        # Filter out entries where the corresponding image file doesn't exist
        img_paths = [os.path.join(self.img_dir, str(p)) for p in self.imgname]
        valid_paths = np.array([os.path.isfile(p) for p in img_paths])
        if not valid_paths.all() and self.check_file_completeness:
            num_missing = int((~valid_paths).sum())
            log.warning(f"{self.dataset}: {num_missing} missing images. Skipping those samples.")
            # Apply mask to all per-sample arrays
            self.imgname = self.imgname[valid_paths]
            self.scale = self.scale[valid_paths]
            self.center = self.center[valid_paths]
            if isinstance(self.body_pose, np.ndarray) and len(self.body_pose) == self.length:
                self.body_pose = self.body_pose[valid_paths]
            if isinstance(self.betas, np.ndarray) and len(self.betas) == self.length:
                self.betas = self.betas[valid_paths]
            if isinstance(self.cam_int, np.ndarray) and len(self.cam_int) == self.length:
                self.cam_int = self.cam_int[valid_paths]
            if isinstance(self.keypoints, np.ndarray) and len(self.keypoints) == self.length:
                self.keypoints = self.keypoints[valid_paths]
            if isinstance(self.cam_ext, np.ndarray) and len(self.cam_ext) == self.length:
                self.cam_ext = self.cam_ext[valid_paths]
            if isinstance(self.trans_cam, np.ndarray) and len(self.trans_cam) == self.length:
                self.trans_cam = self.trans_cam[valid_paths]
            self.length = self.scale.shape[0]

    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()
        keypoints_2d = self.keypoints[index].copy()
        orig_keypoints_2d = self.keypoints[index].copy()
        center_x = center[0]
        center_y = center[1]
        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=self.BBOX_SHAPE).max()
        if bbox_size < 1:
            #Todo raise proper error
            breakpoint()

        augm_config = copy.deepcopy(self.cfg.DATASETS.CONFIG)
        imgname = os.path.join(self.img_dir, self.imgname[index])
        cv_img = cv2.imread(imgname, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        cv_img = cv_img[:, :, ::-1]
        aspect_ratio, img_full_resized = resize_image(cv_img, 256)
        img_full_resized = np.transpose(img_full_resized.astype('float32'),
                        (2, 0, 1))/255.0
        item['img_full_resized'] = self.normalize_img(torch.from_numpy(img_full_resized).float())

        item['pose'] = self.body_pose[index]
        item['betas'] = self.betas[index]

        smpl_params = {'global_orient': self.body_pose[index][:3].astype(np.float32),
                    'body_pose': self.body_pose[index][3:].astype(np.float32),
                    'betas': self.betas[index].astype(np.float32)
                    }
        item['smpl_params'] = smpl_params
        item['translation'] = self.cam_ext[index][:, 3]
        if 'trans_cam' in self.data.files:
            item['translation'][:3] += self.trans_cam[index]
        img_patch_rgba = None
        img_patch_cv = None
        img_patch_rgba, \
        img_patch_cv,\
        keypoints_2d, \
        img_size, cx, cy, bbox_w, bbox_h, trans, scale_aug = get_example(imgname,
                                      center_x, center_y,
                                      bbox_size, bbox_size,
                                      keypoints_2d,
                                      FLIP_KEYPOINT_PERMUTATION,
                                      self.IMG_SIZE, self.IMG_SIZE,
                                      self.MEAN, self.STD, self.is_train, augm_config,
                                      is_bgr=True, return_trans=True,
                                      use_skimage_antialias=self.use_skimage_antialias,
                                      border_mode=self.border_mode,
                                      dataset=self.dataset
                                      )
        new_center = np.array([cx, cy])
        img_patch = img_patch_rgba[:3,:,:]
        item['cam_int'] = np.array(self.cam_int[index]).astype(np.float32)
        item['img_disp'] = img_patch_cv
        item['img'] = img_patch
        item['keypoints_2d'] = keypoints_2d.astype(np.float32)
        item['orig_keypoints_2d'] = orig_keypoints_2d
        item['box_center'] = new_center
        item['box_size'] = bbox_w * scale_aug
        item['img_size'] = 1.0 * img_size.copy()
        item['_scale'] = scale
        item['_trans'] = trans
        item['imgname'] = imgname
        item['dataset'] = self.dataset
        return item

    def __len__(self):
        return int(len(self.imgname))

