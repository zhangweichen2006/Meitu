import numpy as np
import random
import cv2
import os
import argparse
import torch
import pyrender
import trimesh
import smplx

from tqdm import tqdm

# for visualizing and checking purpose, no need jaw, eye pose
smpl_shape = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3), 'body_pose': (-1, 69)}
smplx_shape = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3), 
        'body_pose': (-1, 21, 3), 'left_hand_pose': (-1, 15, 3), 'right_hand_pose': (-1, 15, 3)}

def get_cam_params(param, idx):

    '''
        Read camera parameters from humandata
        Input:
        Output:
    '''

    R, T = None, None
    
    # read cam params
    try:
        focal_length = param['meta'].item()['focal_length'][idx]
        camera_center = param['meta'].item()['principal_point'][idx]
    except TypeError:
        focal_length = param['meta'].item()['focal_length']
        camera_center = param['meta'].item()['princpt']
    try:
        R = param['meta'].item()['R'][idx]
        T = param['meta'].item()['T'][idx]
    except KeyError:
        R = None
        T = None
    except IndexError:
        R = None
        T = None

    focal_length = np.asarray(focal_length).reshape(-1)
    camera_center = np.asarray(camera_center).reshape(-1)

    if len(focal_length)==1:
        focal_length = [focal_length, focal_length]
    if len(camera_center)==1:
        camera_center = [camera_center, camera_center]

    return focal_length, camera_center, R, T


def render_pose(img, body_model_param, body_model, camera, return_mask=False,
                 R=None, T=None):
    
    # the inverse is same
    pyrender2opencv = np.array([[1.0, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])
    
    output = body_model(**body_model_param, return_verts=True)
    faces = body_model.faces
    
    vertices = output['vertices'].detach().cpu().numpy().squeeze()
    
    # adjust vertices beased on R and T
    if R is not None:
        joints = output['joints'].detach().cpu().numpy().squeeze()
        root_joints = joints[0]
        verts_T = np.dot(np.array(R), root_joints) - root_joints  + np.array(T)
        vertices = vertices + verts_T

    # render material
    base_color = (1.0, 193/255, 193/255, 1.0)
    material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.3,
            alphaMode='OPAQUE',
            baseColorFactor=base_color)
    
    # transfer to trimesh
    body_trimesh = trimesh.Trimesh(vertices, faces, process=False)
    body_mesh = pyrender.Mesh.from_trimesh(body_trimesh, material=material)

    # prepare camera and light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    cam_pose = pyrender2opencv @ np.eye(4)
    
    # build scene
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                    ambient_light=(0.3, 0.3, 0.3))
    scene.add(camera, pose=cam_pose)
    scene.add(light, pose=cam_pose)
    scene.add(body_mesh, 'mesh')

    # render scene
    # os.environ["PYOPENGL_PLATFORM"] = "osmesa" # include this line if use in vscode
    r = pyrender.OffscreenRenderer(viewport_width=img.shape[1],
                                    viewport_height=img.shape[0],
                                    point_size=1.0)
    
    # 
    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    # depth = r.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
    # normal, _ = r.render(scene, flags=pyrender.RenderFlags.FACE_NORMALS)
    
    color = color.astype(np.float32) / 255.0
    # depth = np.asarray(depth, dtype=np.float32)
    # normal = np.asarray(normal, dtype=np.float32)

    # set transparency in [0.0, 1.0]
    alpha = 0.8 
    valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
    valid_mask = valid_mask * alpha
    
    img = img / 255
    output_img = (color[:, :, :] * valid_mask + (1 - valid_mask) * img)

    img = (output_img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if return_mask:
        return img, valid_mask, (color * 255).astype(np.uint8)

    return img


def visualize_humandata(args):
    
    '''
    '''

    # TODO: load from args.path
    param = dict(np.load(args.hd_path, allow_pickle=True))

    # check for annot and type
    has_smplx, has_smpl, has_gender = False, False, False
    if 'smpl' in param.keys():
        has_smpl = True
    elif 'smplx' in param.keys():
        has_smplx = True
    if 'meta' in param.keys():
        if 'gender' in param['meta'].item().keys():
            has_gender = True
    assert has_smpl or has_smplx, 'No body model annotation found in the dataset'

    # load params
    if has_smpl:
        body_model_param_smpl = param['smpl'].item()
    if has_smplx:
        body_model_param_smplx = param['smplx'].item()  

    # read smplx only if has both smpl and smplx
    if has_smpl and has_smplx:
        has_smpl = False

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    flat_hand_mean = args.flat_hand_mean
    if 'misc' in param.keys():
        if 'flat_hand_mean' in param['misc'].item().keys():
            flat_hand_mean = param['misc'].item()['flat_hand_mean']


    # build smpl model TODO: args for model path
    gendered_smpl = {}
    for gender in ['male', 'female', 'neutral']:
        kwargs_smpl = dict(
            gender=gender,
            num_betas=10,
            use_face_contour=True,
            use_pca=False,
            batch_size=1)
        gendered_smpl[gender] = smplx.create(
            args.body_model_path, 'smpl', 
            **kwargs_smpl).to(device)
    
    # build smplx model TODO: model path
    gendered_smplx = {}
    for gender in ['male', 'female', 'neutral']:
        kwargs_smplx = dict(
            gender=gender,
            num_betas=10,
            use_face_contour=True,
            flat_hand_mean=flat_hand_mean,
            use_pca=False,
            batch_size=1)
        gendered_smplx[gender] = smplx.create(
            args.body_model_path, 'smplx', 
            **kwargs_smplx).to(device)   
    
    # for idx in idx_list:
    sample_size = args.render_num
    if sample_size > len(param['image_path']):
        idxs = range(len(param['image_path']))
    else:
        idxs = random.sample(range(len(param['image_path'])), sample_size)

    for idx in tqdm(sorted(idxs), desc=f'Processing npz {os.path.basename(args.hd_path)}, sample size: {sample_size}',
                    position=0, leave=False):

        # Load image
        image_p = param['image_path'][idx]
        image_path = os.path.join(args.image_folder, image_p) 

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # ---------------------- render single pose ------------------------
        # read cam params
        focal_length, camera_center, R, T = get_cam_params(param, idx)

        # read gender
        if has_gender:
            try:
                gender = param['meta'].item()['gender'][idx]
            except IndexError: 
                gender = 'neutral'
        else:
            gender = 'neutral'
 
        # prepare for mesh projection
        camera = pyrender.camera.IntrinsicsCamera(
            fx=focal_length[0], fy=focal_length[1],
            cx=camera_center[0], cy=camera_center[1])

        if has_smpl:
            intersect_key = list(set(body_model_param_smpl.keys()) & set(smpl_shape.keys()))
            body_model_param_tensor = {key: torch.tensor(
                    np.array(body_model_param_smpl[key][idx:idx+1]).reshape(smpl_shape[key]),
                            device=device, dtype=torch.float32)
                            for key in intersect_key
                            if len(body_model_param_smpl[key][idx:idx+1]) > 0}
        
            rendered_image = render_pose(img=image, 
                                    body_model_param=body_model_param_tensor, 
                                    body_model=gendered_smpl[gender],
                                    camera=camera,
                                    R=R, T=T)             
        if has_smplx:
            intersect_key = list(set(body_model_param_smplx.keys()) & set(smplx_shape.keys()))
            body_model_param_tensor = {key: torch.tensor(
                    np.array(body_model_param_smplx[key][idx:idx+1]).reshape(smplx_shape[key]),
                            device=device, dtype=torch.float32)
                            for key in intersect_key
                            if len(body_model_param_smplx[key][idx:idx+1]) > 0}

            rendered_image = render_pose(img=image, 
                                        body_model_param=body_model_param_tensor, 
                                        body_model=gendered_smplx[gender],
                                        camera=camera,
                                        R=R, T=T)

        # ---------------------- render results ----------------------
        os.makedirs(args.output_folder, exist_ok=True)
        
        # save image
        out_image_path = os.path.join(args.output_folder, 
                                    f'{os.path.basename(args.hd_path)[:-4]}_{idx}.png')
        # print(f'Saving image to {out_image_path}')
        cv2.imwrite(out_image_path, rendered_image)
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()    
    # path args
    parser.add_argument('--hd_path', type=str, required=False,
                        help='path to humandata npz file',
                        default='/mnt/d/test_area/hd_sample_SMPLestX/hd_10sample.npz')
    parser.add_argument('--image_folder', type=str, required=False, 
                        help='path to the image base folder',
                        default='/mnt/d/test_area/hd_sample_SMPLestX')
    parser.add_argument('--output_folder', type=str, required=False,
                        help='path to folder that writes the rendered image',
                        default='/mnt/d/test_area/hd_sample_SMPLestX/output')
    # TODO: add default bm path
    parser.add_argument('--body_model_path', type=str, required=False,
                        help='path to smpl/smplx models folder, if you follow repo file structure, \
                            no need to specify',
                        default='/home/weichen/wc_workspace/models/human_model')
    
    # render args
    parser.add_argument('--flat_hand_mean', type=bool, required=False,
                        help='use flat hand mean for smplx, will try to load from humandata["misc"] \
                            if not find, will use value from args',
                        default=False)
    parser.add_argument('--render_num', type=int, required=False,
                        help='Randomly senect how many instances to render',
                        default='10')
    
    args = parser.parse_args()

    visualize_humandata(args)
    