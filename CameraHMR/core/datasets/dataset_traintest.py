
import os
import cv2
import torch
import copy
from ..utils.numpy_compat import ensure_numpy_legacy_aliases
ensure_numpy_legacy_aliases()
import smplx
import pickle
import numpy as np
from torch.utils.data import Dataset
from ..utils.pylogger import get_pylogger
from ..configs import DATASET_FOLDERS, DATASET_FILES
from .utils import expand_to_aspect_ratio, get_example, resize_image
from torchvision.transforms import Normalize
from ..constants import FLIP_KEYPOINT_PERMUTATION, NUM_JOINTS, NUM_BETAS, NUM_PARAMS_SMPL, NUM_PARAMS_SMPLX, SMPLX2SMPL, SMPLX_MODEL_DIR, SMPL_MODEL_DIR
from ..utils.camera_ray_utils import calc_plucker_embeds
from ..utils.smpl_utils import compute_normals_torch
# from tools.vis import vis_img,vis_pc
from ..sapiens_normal_model import SapiensNormalWrapper
import ast

log = get_pylogger(__name__)


class DatasetTrainTest(Dataset):
    def __init__(self, cfg, dataset, version='traintest', is_train=False, mean=None, std=None, cropsize=None):
        super(DatasetTrainTest, self).__init__()

        self.dataset = dataset
        self.version = version
        self.cfg = cfg
        self.is_train = is_train

        self.IMG_SIZE = cfg.MODEL.IMAGE_SIZE
        self.BBOX_SHAPE = cfg.MODEL.get('BBOX_SHAPE', None)
        if mean is not None:
            self.MEAN = np.array(mean)
        else:
            self.MEAN = 255. * np.array(cfg.MODEL.IMAGE_MEAN)
        if std is not None:
            self.STD = np.array(std)
        else:
            self.STD = 255. * np.array(cfg.MODEL.IMAGE_STD)
        self.normalize_img = Normalize(mean=cfg.MODEL.IMAGE_MEAN,
                                    std=cfg.MODEL.IMAGE_STD)
        self.use_skimage_antialias = cfg.DATASETS.get('USE_SKIMAGE_ANTIALIAS', False)
        self.border_mode = {
            'constant': cv2.BORDER_CONSTANT,
            'replicate': cv2.BORDER_REPLICATE,
        }[cfg.DATASETS.get('BORDER_MODE', 'constant')]

        self.img_dir = DATASET_FOLDERS[dataset]
        self.data = np.load(DATASET_FILES[version][dataset], allow_pickle=True)
        self.imgname = self.data['imgname']

        self.img_paths = [os.path.join(self.img_dir, str(p)) for p in self.imgname]
        self.valid_paths = np.array([os.path.isfile(p) for p in self.img_paths])

        # Filter out entries where the corresponding image file doesn't exist
        if not self.valid_paths.all():
            num_missing = int((~self.valid_paths).sum())
            log.warning(f"{self.dataset}: {num_missing} missing images. Skipping those samples.")
            # Apply mask to all per-sample arrays
            self.imgname = self.imgname[self.valid_paths]
            self.data = {k: v[self.valid_paths] for k, v in self.data.items()}
            self.img_paths = np.array(self.img_paths)[self.valid_paths].tolist()

        self.scale = self.data['scale']
        self.center = self.data['center']
        if ('coco' in self.dataset or 'lsp' in self.dataset):
            self.scale = self.scale/200

        if 'pose_cam' in self.data:
            if 'smplx' in self.dataset:
                self.pose = self.data['pose_cam'][:, :NUM_PARAMS_SMPLX*3].astype(np.float)
            else:
                self.pose = self.data['pose_cam'][:, :NUM_PARAMS_SMPL*3].astype(np.float)
        else:
            self.pose = np.zeros((len(self.imgname), 24*3), dtype=np.float32)

        if 'shape' in self.data:
            self.betas = self.data['shape'].astype(np.float)[:,:NUM_BETAS]
        else:
            self.betas = np.zeros((len(self.imgname), 10), dtype=np.float32)

        if 'part' in self.data:
            self.keypoints = self.data['part']
        elif 'gtkps' in self.data:
            self.keypoints = self.data['gtkps'][:,:NUM_JOINTS]#Todo later: change it to a variable
        elif 'body_keypoints_2d' in self.data:
            self.keypoints = self.data['body_keypoints_2d']
        else:
            self.keypoints = np.zeros((len(self.imgname), NUM_JOINTS, 3))

        if self.keypoints.shape[2]<3:
            ones_array = np.ones((self.keypoints.shape[0],self.keypoints.shape[1],1))
            self.keypoints = np.concatenate((self.keypoints, ones_array), axis=2)

        if 'cam_int' in self.data:
            self.cam_int = self.data['cam_int']
        else:
            self.cam_int = np.zeros((len(self.imgname),3,3), dtype=np.float32)

        try:
            gender = self.data['gender']
            self.gender = np.array([0 if str(g) == 'm' or str(g)=='male'
                                    else 1 for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1 * np.ones(len(self.imgname)).astype(np.int32)

        self.smpl_gt_male = smplx.SMPL(SMPL_MODEL_DIR,
                                gender='male')
        self.smpl_gt_female = smplx.SMPL(SMPL_MODEL_DIR,
                                    gender='female')
        self.smpl_gt_neutral = smplx.SMPL(SMPL_MODEL_DIR,
                                    gender='neutral')

        self.smplx_gt_male = smplx.SMPLX(SMPLX_MODEL_DIR,
                                gender='male')
        self.smplx_gt_female = smplx.SMPLX(SMPLX_MODEL_DIR,
                                    gender='female')
        self.smplx_gt_neutral = smplx.SMPLX(SMPLX_MODEL_DIR,
                                    gender='neutral')
        self.smplx2smpl = pickle.load(open(SMPLX2SMPL, 'rb'))
        self.smplx2smpl = torch.tensor(self.smplx2smpl['matrix'][None],
                                        dtype=torch.float32)

        if 'cam_ext' in self.data:
            self.cam_ext = self.data['cam_ext']
        else:
            self.cam_ext = np.zeros((self.imgname.shape[0], 4, 4))

        #Only for BEDLAM and AGORA
        if 'trans_cam' in self.data:
            self.trans_cam = self.data['trans_cam']
        else:
            self.trans_cam = np.zeros((self.imgname.shape[0],3))

        if 'smpl_normals' in self.data:
            self.smpl_normals = self.data['smpl_normals']
        else:
            log.info(f'Processing smpl normals ...')
            smpl_normals_arr = []
            # batch compute smpl_normals (batch size = 16)
            for i in range(0, len(self.imgname)):
                smpl_output = self.smpl_gt_neutral.forward(
                    global_orient=torch.from_numpy(self.pose[i][:3]).float().unsqueeze(0),
                    body_pose=torch.from_numpy(self.pose[i][3:]).float().unsqueeze(0),
                    betas=torch.from_numpy(self.betas[i]).float().unsqueeze(0)
                )
                verts = smpl_output.vertices.squeeze(0)
                faces_t = torch.as_tensor(self.smpl_gt_neutral.faces, dtype=torch.long, device=verts.device)
                vertex_normals = compute_normals_torch(verts, faces_t)
                smpl_normals_arr.append(vertex_normals.detach().cpu().numpy())
            self.smpl_normals = np.array(smpl_normals_arr)
            # save smpl_normals to dataset
            data_arrays = {k: self.data[k] for k in self.data.files}
            data_arrays['smpl_normals'] = self.smpl_normals
            np.savez(DATASET_FILES[self.version][dataset], **data_arrays)

        if 'sapiens_pixel_normals_path' in self.data:
            self.sapiens_pixel_normals_path = self.data['sapiens_pixel_normals_path']
        else:
            log.info(f'Processing sapiens pixel normals ...')
            sapiens_ckpt = cfg.paths.get('sapiens_normal_ckpt', os.environ.get('SAPIENS_NORMAL_CKPT', None))
            self.infer_batch_size = cfg.pretrained_models.sapiens.get('infer_batch_size', 4)
            sapiens_normal_model = SapiensNormalWrapper(
                                        checkpoint_path=sapiens_ckpt,
                                        use_torchscript=cfg.pretrained_models.sapiens.use_torchscript,
                                        fp16=cfg.pretrained_models.sapiens.fp16,
                                        input_size_hw=ast.literal_eval(cfg.pretrained_models.sapiens.input_crop_size_hw),
                                        mean=cfg.pretrained_models.sapiens.mean,
                                        std=cfg.pretrained_models.sapiens.std,
                                        compile_model=cfg.pretrained_models.sapiens.compile_model,
                                    )

            sapiens_pixel_normals = []
            for i in range(0, len(self.img_paths), self.infer_batch_size):
                end_idx = min(i+self.infer_batch_size, len(self.imgname))
                normal_res = sapiens_normal_model.infer_paths(self.img_paths[i:end_idx])
                sapiens_pixel_normals.append(normal_res)
                # normal 2 rgb and save
                # save sapiens_pixel_normals to image_path_normal
                sapiens_normal_folder = self.img_paths[0].replace('training-images', 'traintest-labels').rsplit('/',1)[0]+"_sapiens_normals"

                for n, normal in enumerate(normal_res):
                    normal_rgb = np.array(normal*0.5+0.5)*255.0
                    normal_rgb = np.moveaxis(np.array(normal_rgb), 0, -1).astype(np.uint8)
                    normal_rgb = cv2.cvtColor(normal_rgb, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(f'{sapiens_normal_folder}/normal_rgb_{n}.png', normal_rgb)

            self.sapiens_pixel_normals = sapiens_pixel_normals
            self.data['sapiens_pixel_normals'] = self.sapiens_pixel_normals

        if 'pi3_pc' in self.data:
            self.pi3_pc = self.data['pi3_pc']
        else:
            self.pi3_pc = []

        self.length = self.scale.shape[0]
        log.info(f'Loaded {self.dataset} dataset, num samples {self.length}')

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

        item['pose'] = self.pose[index]
        item['betas'] = self.betas[index]

        if 'smplx' in self.dataset:
            smpl_params = {'global_orient': self.pose[index][:3].astype(np.float32),
                        'body_pose': self.pose[index][3:66].astype(np.float32),
                        'betas': self.betas[index].astype(np.float32)
                        }
            item['smpl_params'] = smpl_params
        else:
            smpl_params = {'global_orient': self.pose[index][:3].astype(np.float32),
                        'body_pose': self.pose[index][3:].astype(np.float32),
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

        # TODO: Calculate plucker_embeds

        # COMENTED OUT VAL CODE
        # img_w = img_size[1]
        # img_h = img_size[0]
        # fl = 5000 # This will be updated in forward_step of camerahmr_trainer
        # item['cam_int'] = np.array([[fl, 0, img_w/2.], [0, fl, img_h / 2.], [0, 0, 1]]).astype(np.float32)
        item['cam_int'] = np.array(self.cam_int[index]).astype(np.float32)


        new_center = np.array([cx, cy])
        img_patch = img_patch_rgba[:3,:,:]
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
        item['dataset'] = self.dataset
        item['gender'] = self.gender[index]

        item['pose'] = self.pose[index]
        item['betas'] = self.betas[index]

        if 'smplx' in self.dataset:
            if self.gender[index] == 1:
                model = self.smplx_gt_female
            elif self.gender[index] == 0:
                model = self.smplx_gt_male
            else:
                model = self.smplx_gt_neutral

            gt_smpl_out = model(
                        global_orient=torch.from_numpy(item['smpl_params']['global_orient']).unsqueeze(0),
                        body_pose=torch.from_numpy(item['smpl_params']['body_pose']).unsqueeze(0),
                        betas=torch.from_numpy(item['smpl_params']['betas']).unsqueeze(0))
            gt_vertices = gt_smpl_out.vertices.detach()
            gt_vertices = torch.matmul(self.smplx2smpl, gt_vertices)
            item['keypoints_3d'] = torch.matmul(self.smpl_gt_neutral.J_regressor, gt_vertices[0])
            item['vertices'] = gt_vertices[0].float()
            # gt_smpl_normals = gt_smpl_out.normals.detach()
        else:
            if self.gender[index] == 1:
                model = self.smpl_gt_female
            elif self.gender[index] == 0:
                model = self.smpl_gt_male
            else:
                model = self.smpl_gt_neutral
            gt_smpl_out = model(
                        global_orient=torch.from_numpy(item['smpl_params']['global_orient']).unsqueeze(0),
                        body_pose=torch.from_numpy(item['smpl_params']['body_pose']).unsqueeze(0),
                        betas=torch.from_numpy(item['smpl_params']['betas']).unsqueeze(0))

            gt_vertices = gt_smpl_out.vertices.detach()
            item['keypoints_3d'] = torch.matmul(model.J_regressor, gt_vertices[0])
            item['vertices'] = gt_vertices[0].float()

        if 'smpl_normals' in self.data.files:
            item['smpl_normals'] = self.smpl_normals[index]
        else:
            item['smpl_normals'] = np.zeros((1, 6890, 3))

        if 'sapiens_pixel_normals_path' in self.data.files:
            item['sapiens_pixel_normals_path'] = self.sapiens_pixel_normals_path[index]
        else:
            item['sapiens_pixel_normals_path'] = np.zeros((1, 1, 1))

        return item

    def __len__(self):
        return int(len(self.imgname))

