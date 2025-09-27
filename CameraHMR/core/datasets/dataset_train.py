
import os
import cv2
import torch
import copy
import numpy as np
from torch.utils.data import Dataset
from ..utils.pylogger import get_pylogger
from ..configs import DATASET_FOLDERS, DATASET_FILES,SAPIENS_TRAINING_PROCESS_NORMAL_VERSION, SAPIENS_TEST_PROCESS_NORMAL_VERSION, SAPIENS_TRAINING_PROCESS_NORMAL_VERSION2, SAPIENS_TEST_PROCESS_NORMAL_VERSION2, SAPIENS_TRAINING_IMGMATCH_NORMAL_VERSION, SAPIENS_TEST_IMGMATCH_NORMAL_VERSION, NORMAL_PREPROCESS
from ..constants import FLIP_KEYPOINT_PERMUTATION, NUM_JOINTS, NUM_BETAS, NUM_PARAMS_SMPL
from .utils import expand_to_aspect_ratio, get_example, resize_image
from torchvision.transforms import Normalize
# from tools.vis import vis_img,vis_pc
from SapiensLite.demo.adhoc_image_dataset import AdhocImageDataset
from SapiensLite.demo.vis_normal import get_preprocess_args, inference_model, load_model
from SapiensLite.demo.revert_utils import revert_npy
from tqdm import tqdm
import torch.nn.functional as F

log = get_pylogger(__name__)


class DatasetTrain(Dataset):
    def __init__(self, cfg, dataset, is_train=True, version='train'):
        super(DatasetTrain, self).__init__()

        self.dataset = dataset
        self.is_train = is_train
        self.version = version
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

        self.check_file_completeness_and_filter = cfg.DATASETS.get('check_file_completeness_and_filter', False)

        self.img_dir = DATASET_FOLDERS[dataset]
        self.data = np.load(DATASET_FILES['traintest'][dataset], allow_pickle=True)

        self.imgname = self.data['imgname']

        self.sapiens_normal_version = SAPIENS_TRAINING_PROCESS_NORMAL_VERSION if "training-images" in self.img_dir else SAPIENS_TEST_PROCESS_NORMAL_VERSION
        self.sapiens_normal_version2 = SAPIENS_TRAINING_PROCESS_NORMAL_VERSION2 if "training-images" in self.img_dir else SAPIENS_TEST_PROCESS_NORMAL_VERSION2
        self.sapiens_normal_imgmatch_version = SAPIENS_TRAINING_IMGMATCH_NORMAL_VERSION if "training-images" in self.img_dir else SAPIENS_TEST_IMGMATCH_NORMAL_VERSION
        self.replace_src_folder = "training-images" if "training-images" in self.img_dir else "test-images"
        
        # Filter out entries where the corresponding image file doesn't exist
        self.img_paths = [os.path.join(self.img_dir, str(p)) for p in self.imgname]
        if self.check_file_completeness_and_filter:
            valid_paths = np.array([os.path.isfile(p) for p in self.img_paths])
            if not valid_paths.all():
                num_missing = int((~valid_paths).sum())
                log.warning(f"{self.dataset}: {num_missing} missing images. Skipping those samples.")
                # Apply mask to all arrays
                self.imgname = np.array(self.imgname)[valid_paths].tolist()
                self.data = {k: v[valid_paths] for k, v in self.data.items()}
                self.img_paths = np.array(self.img_paths)[valid_paths].tolist()

        # check sapiens normal files path
        self.sapiens_normals_path = [self.img_to_normals_path(i, self.replace_src_folder, self.sapiens_normal_version) for i in self.img_paths]
        self.sapiens_normals_path2 = [self.img_to_normals_path(i, self.replace_src_folder, self.sapiens_normal_version2) for i in self.img_paths]
        self.sapiens_normals_path_imgmatch = [self.img_to_normals_path(i, self.replace_src_folder, self.sapiens_normal_imgmatch_version) for i in self.img_paths]

        # indicate existance of sapiens normal files in label file
        if 'sapiens_normals_folder' in self.data and 'normal_swapHW' in self.data and 'normal_preprocess' in self.data:
            sapiens_normals_folder, sapiens_normals_folder2 = self.data['sapiens_normals_folder']

            self.normal_swapHW = self.data['normal_swapHW']
            self.normal_preprocess = self.data['normal_preprocess']
            if self.check_file_completeness_and_filter:
                valid_paths_sapiens_normals_imgmatch = np.array([os.path.isfile(p) for p in self.sapiens_normals_path_imgmatch])
                if not valid_paths_sapiens_normals_imgmatch.all():
                    num_missing = int((~valid_paths_sapiens_normals_imgmatch).sum())
                    log.warning(f"{self.dataset}: {num_missing} missing sapiens pixel normals. Regenerating those samples...")
                    # use normal path 1 only and if normal not exist in both folders, regenerate the normal, if failed, just skip computing normal loss.
                    sapiens_normals_path_to_regenerate = []
                    sapiens_normals_path_to_regenerate_imgmatch = np.array(self.sapiens_normals_path_imgmatch)[~valid_paths_sapiens_normals_imgmatch].tolist()
                    imgname_to_regenerate = np.array(self.img_paths)[~valid_paths_sapiens_normals_imgmatch].tolist()
                    # revert first if not exist regenerate the normal
                    self.revert_or_regen_sapiens_normals(imgname_to_regenerate,sapiens_normals_path_to_regenerate, sapiens_normals_path_to_regenerate_imgmatch)
        else:
            # process and check folder
            self.normal_preprocess = NORMAL_PREPROCESS['traintest'][self.dataset]['preprocess']
            self.normal_swapHW = NORMAL_PREPROCESS['traintest'][self.dataset]['swapHW']

            # normal folder path
            sapiens_normals_folder = self.img_dir.replace(self.replace_src_folder, self.sapiens_normal_version) if self.is_train else self.img_dir.replace(self.replace_src_folder, self.sapiens_normal_version)
            sapiens_normals_folder2 = self.img_dir.replace(self.replace_src_folder, self.sapiens_normal_version2) if self.is_train else self.img_dir.replace(self.replace_src_folder, self.sapiens_normal_version2)

            # Only raise if BOTH candidate folders are missing
            if (not os.path.exists(sapiens_normals_folder)) and (not os.path.exists(sapiens_normals_folder2)):
                log.error(f'Sapiens normal folder does not exist: {sapiens_normals_folder}')

                log.info(f'Processing sapiens pixel normals ...')
                sapiens_ckpt = cfg.paths.get('sapiens_normal_ckpt', os.environ.get('SAPIENS_NORMAL_CKPT', None))
                self.infer_batch_size = cfg.pretrained_models.sapiens.get('infer_batch_size', 4)
                # sapiens_normal_model = SapiensNormalWrapper() # SLOW...
                # TODO: port script
                self.regen_sapiens_normals(self.img_paths, self.sapiens_normals_path, self.sapiens_normals_path_imgmatch)

            # revert normal files first if not exist regenerate the normal
            if self.check_file_completeness_and_filter:
                self.revert_or_regen_sapiens_normals(self.img_paths, self.sapiens_normals_path, self.sapiens_normals_path_imgmatch)

            # save smpl_normals to dataset
            self.data['sapiens_normals_folder'] = (sapiens_normals_folder, sapiens_normals_folder2)
            self.data['normal_swapHW'] = self.normal_swapHW
            self.data['normal_preprocess'] = self.normal_preprocess
            np.savez(DATASET_FILES['traintest'][dataset], **self.data)


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

        try:
            gender = self.data['gender']
            self.gender = np.array([0 if str(g) == 'm' or str(g)=='male'
                                    else 1 for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1 * np.ones(len(self.imgname)).astype(np.int32)

    # Helper to map an image path to the best available normals file path
    def _map_to_normals_path(self, img_path, replace_src, replace_tgt, replace_tgt2):
        # Prefer first folder, fall back to second; try .npz then .npy
        base1 = os.path.normpath(img_path.replace(replace_src, replace_tgt))
        root1, _ = os.path.splitext(base1)
        cand1_npy = root1 + '.npy'
        if os.path.isfile(cand1_npy):
            return cand1_npy
        # Try second folcand1_npyder if provided
        base2 = os.path.normpath(img_path.replace(replace_src, replace_tgt2))
        root2, _ = os.path.splitext(base2)
        cand2_npy = root2 + '.npy'
        if os.path.isfile(cand2_npy):
            # print(f"copying {cand2_npy} to {cand1_npy} to use later...")
            # os.system(f"cp {cand2_npy} {cand1_npy} &")
            return cand2_npy
        # Default to first candidate with .npz extension
        return None

    def img_to_normals_path(self, img_path, replace_src, replace_tgt):
        base1 = os.path.normpath(img_path.replace(replace_src, replace_tgt))
        root1, _ = os.path.splitext(base1)
        return root1 + '.npy'

    def revert_or_regen_sapiens_normals(self, imgname_to_regenerate, sapiens_normals_path_to_regenerate, sapiens_normals_path_to_regenerate_imgmatch):
        valid_paths_sapiens_normals = []
        # use normal path 2 and copy to normal path 1
        for idx, i in enumerate(imgname_to_regenerate):
            if not os.path.isfile(sapiens_normals_path_to_regenerate_imgmatch[idx]):
                map_normal_path = self._map_to_normals_path(i, self.replace_src_folder, self.sapiens_normal_version, self.sapiens_normal_version2)
                if map_normal_path:
                    # revert imgmatch normal from processed normal
                    log.info(f"Reverting imgmatch normal from {map_normal_path}")
                    rev_normal_imgmatch = revert_npy(map_normal_path, imgname_to_regenerate[idx], swapHW=self.normal_swapHW, mode=self.normal_preprocess)
                    os.makedirs(os.path.dirname(sapiens_normals_path_to_regenerate_imgmatch[idx]), exist_ok=True)
                    np.save(sapiens_normals_path_to_regenerate_imgmatch[idx], rev_normal_imgmatch)
                    valid_paths_sapiens_normals.append(True)
                    cv2.imwrite(sapiens_normals_path_to_regenerate_imgmatch[idx].replace('.npy', '.png'), 255*(rev_normal_imgmatch*0.5+0.5))
                    os.system(f"mkdir -p {os.path.dirname(self.sapiens_normals_path2[idx])}")
                    os.system(f"mv {self.sapiens_normals_path[idx]} {self.sapiens_normals_path2[idx]} &")
                else:
                    valid_paths_sapiens_normals.append(False)
            else:
                valid_paths_sapiens_normals.append(True)
                # move normal path 1 (vepfs local) to normal path 2 (tos)
                log.info(f"IMGMatch normal exists. moving {self.sapiens_normals_path[idx]} to {self.sapiens_normals_path2[idx]}")
                os.system(f"mkdir -p {os.path.dirname(self.sapiens_normals_path2[idx])}")
                os.system(f"mv {self.sapiens_normals_path[idx]} {self.sapiens_normals_path2[idx]} &")
                os.system(f"mv {self.sapiens_normals_path[idx].replace('.npy', '.png')} {self.sapiens_normals_path2[idx].replace('.npy', '.png')} &")
        valid_paths_sapiens_normals = np.array(valid_paths_sapiens_normals)

        if not valid_paths_sapiens_normals.all():
            num_missing = int((~valid_paths_sapiens_normals).sum())
            log.warning(f"{self.dataset}: {num_missing} missing sapiens pixel normals. Regenerating those samples...")
            # use normal path 1 only and if normal not exist in both folders, regenerate the normal, if failed, just skip computing normal loss.
            sapiens_normals_path_to_regenerate = [] # np.array(self.sapiens_normals_path)[~valid_paths_sapiens_normals].tolist() Not necessary for now.
            sapiens_normals_path_to_regenerate_imgmatch = np.array(sapiens_normals_path_to_regenerate_imgmatch)[~valid_paths_sapiens_normals].tolist()
            imgname_to_regenerate = np.array(self.img_paths)[~valid_paths_sapiens_normals].tolist()
            self.regen_sapiens_normals(imgname_to_regenerate, sapiens_normals_path_to_regenerate, sapiens_normals_path_to_regenerate_imgmatch)
        else:
            log.info(f"{self.dataset}: All sapiens pixel normals exist.")

    def regen_sapiens_normals(self, imgname_to_regenerate, sapiens_normals_path_to_regenerate=None, sapiens_normals_path_to_regenerate_imgmatch=None):
        log.info(f'Regenerating sapiens normals ...')
        cropping, resize, zoom_to_3Dpt = get_preprocess_args(self.normal_preprocess)
        dataset = AdhocImageDataset(imgname_to_regenerate, (1024, 768), mean=[123.5, 116.5, 103.5], std=[58.5, 57.0, 57.5], cropping=cropping, resize=resize, zoom_to_3Dpt=zoom_to_3Dpt, out_names=sapiens_normals_path_to_regenerate, out_imgmatch_names=sapiens_normals_path_to_regenerate_imgmatch, swapHW=self.normal_swapHW)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1
        )
        SAPIENS_NORMAL_CKPT = self.cfg.paths.get('sapiens_normal_ckpt', os.environ.get('SAPIENS_NORMAL_CKPT', None))
        USE_TORCHSCRIPT = "_torchscript" in SAPIENS_NORMAL_CKPT
        exp_model = load_model(SAPIENS_NORMAL_CKPT, USE_TORCHSCRIPT)
        if not USE_TORCHSCRIPT:
            dtype = torch.bfloat16
            exp_model.to(dtype)
            exp_model = torch.compile(exp_model, mode="max-autotune", fullgraph=True)
        else:
            dtype = torch.float32  # TorchScript models use float32
            exp_model = exp_model.to(self.device)
        for batch_idx, (batch_image_name, batch_out_name, batch_out_imgmatch_name, batch_orig_imgs, batch_imgs) in tqdm(
            enumerate(dataloader), total=len(dataloader)
        ):
            result = inference_model(exp_model, batch_imgs, dtype=torch.float32)
            if batch_out_name:
                os.makedirs(os.path.dirname(batch_out_name[0]), exist_ok=True)
                np.save(batch_out_name[0], result[0])
            normal_result = revert_npy(result[0], batch_orig_imgs[0], swapHW=self.normal_swapHW, mode=self.normal_preprocess)
            os.makedirs(os.path.dirname(batch_out_imgmatch_name[0]), exist_ok=True)
            np.save(batch_out_imgmatch_name[0], normal_result)
            if batch_idx == 0:
                cv2.imwrite(batch_out_imgmatch_name[0].replace('.npy', '.png'), 255*(normal_result*0.5+0.5))

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
        img_size, cx, cy, bbox_w, bbox_h, trans, scale_aug, normal_patch = get_example(imgname,
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
                                      normal_path=(self.sapiens_normals_path_imgmatch[index] if hasattr(self, 'sapiens_normals_path') and len(self.sapiens_normals_path_imgmatch) > 0 else None),
                                      normal_swapHW=self.normal_swapHW,
                                      normal_preprocess=self.normal_preprocess
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
        item['gender'] = self.gender[index]
        return item

    def __len__(self):
        return int(len(self.imgname))

