import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import argparse
import torch
from tqdm import tqdm
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange
from lib.utils.infer_util import *
from lib.utils.train_util import instantiate_from_config
import torchvision
import json
###############################################################################
# Arguments.
###############################################################################

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file.', required=False)
    parser.add_argument('--input_path', type=str, help='Path to input image or directory.', required=False)
    parser.add_argument('--resume_path', type=str, help='Path to saved ckpt.', required=False)
    parser.add_argument('--output_path', type=str, default='outputs/', help='Output directory.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling.')
    parser.add_argument('--distance', type=float, default=1.5, help='Render distance.')
    parser.add_argument('--no_rembg', action='store_true', help='Do not remove input background.')
    parser.add_argument('--render_mode', type=str, default='novel_pose',
                    choices=['novel_pose', 'reconstruct', 'novel_pose_A'],
                    help='Rendering mode: novel_pose (animation), reconstruct (reconstruction), or novel_pose_A (360-degree view with A-pose)')

    return parser.parse_args()

###############################################################################
# Stage 0: Configuration.
###############################################################################

device = torch.device('cuda')



def process_data_on_gpu(args, model, gpu_id, img_paths_list, smplx_ref_path_list, smplx_path_driven_list):
    torch.cuda.set_device(gpu_id)
    model = model.cuda()
    image_plist = []


    render_mode =  args.render_mode


    cam_idx = 0 # 12 # fixed cameras and changes pose for novel poses
    num_imgs = 60
    if_load_betas = True


    if_use_video_cam = False  # If the SMPLX sequence provides camera parameters, this can be set to true.
    if_uniform_coordinates = True # Normalize the SMPL-X sequence for the purpose of driving.


    for input_path, smplx_ref_path, smplx_path in tqdm(zip(img_paths_list, smplx_ref_path_list, smplx_path_driven_list), total = len(img_paths_list)):
        print(f"Processing: {input_path}")

        args.input_path = input_path
        args.input_path_smpl = smplx_ref_path

        # get a name for results
        name = args.input_path.split("/")[-1].split(".")[0] #+ smplx_path.split("/")[-1].split(".")[0]

        ###############################################################################
        # Stage 1: Parameters loading
        ###############################################################################

        ''' # Stage 1.1: SMPLX loading (Beta)'''
        smplx_betas = torch.zeros(10).to(device)
        if args.input_path_smpl is not None and os.path.exists(args.input_path_smpl):
            smplx_json = json.load(open(args.input_path_smpl))
            if "shapes" in smplx_json.keys():
                smplx_betas = torch.as_tensor(smplx_json['shapes'][:10], dtype=torch.float32, device=device)
            elif 'betas' in smplx_json.keys():
                smplx_betas = torch.as_tensor(smplx_json['betas'][:10], dtype=torch.float32, device=device)
            elif 'betas_save' in smplx_json.keys():
                smplx_betas = torch.as_tensor(smplx_json['betas_save'][:10], dtype=torch.float32, device=device)
        smpl_params = torch.zeros(1, 189).to(device)
        if if_load_betas and smplx_betas is not None:
            smpl_params[:, 70:80] = smplx_betas.to(device)

        ''' # Stage 1.2: SMPLX loading (Pose)'''
        # animation
        if render_mode in ['novel_pose'] :

            if smplx_path.endswith(".npy"):
                smpl_params = load_smplx_from_npy(smplx_path)
                smpl_params[:, 70:80] = smplx_betas.to(device)

                # ========= Note: If the video camera is not used, center everything at the origin ========
                if if_uniform_coordinates:
                    print("''' Ending --- Adjusting root orientation angles '''")

                    # Extract root orientation and translation from SMPL parameters
                    root_orient = smpl_params[:, 4:7]
                    trans = smpl_params[:, 1:4]

                    # Reset the first frame's rotation and adjust translations
                    new_root_orient, new_trans = reset_first_frame_rotation(root_orient, trans)

                    # Update the root orientation and translation in the SMPL parameters
                    smpl_params[:, 4:7] = new_root_orient
                    smpl_params[:, 1:4] = new_trans.squeeze()  # Apply the new translation


            elif smplx_path.endswith(".json"):
                ''' for motion-x input '''
                smpl_params = load_smplx_from_json(smplx_path)
                smpl_params[:, 70:80] = smplx_betas.to(device)
                if_use_video_cam = True


        elif render_mode in ['reconstruct']:
            have_reconstruct_cam = False
            if smplx_ref_path is not None and os.path.exists(smplx_ref_path):
                RT_rec, intri_rec, smpl_rec = load_smplify_json(smplx_ref_path)
                have_reconstruct_cam = True
            H_rec, W_rec = get_image_dimensions(input_path)
            '''Apply root rotation for a full 360-degree view of the object'''
            if_add_root_rotate = True
            if have_reconstruct_cam and if_add_root_rotate:
                smpl_params = add_root_rotate_to_smplx(smpl_rec, num_imgs)
                print(" '''ending ---  invert the root angles'''")
            elif not have_reconstruct_cam:
                # Fallback without SMPLify JSON: use default SMPL params and default cameras
                default_params = model.get_default_smplx_params().squeeze().to(device)
                smpl_params = add_root_rotate_to_smplx(default_params.clone(), num_imgs)
                print("Reconstruct fallback: using default SMPL params and default cameras")
            else:
                smpl_params = smpl_params.to(device)
                num_imgs = 1

        elif render_mode in ['novel_pose_A']:
            smpl_params = model.get_default_smplx_params().squeeze()
            smpl_params = smpl_params.to(device)
            smpl_params = add_root_rotate_to_smplx(smpl_params.clone(), num_imgs)
            smpl_params[:, 70:80] = smplx_betas.to(device)

        else:
            raise NotImplementedError(f"Render mode '{render_mode}' is not supported.")

        '''# Stage 1.3: Image loading '''
        image = load_image(args.input_path, args.output_folders['ref'])
        H,W = 896,640
        image_bs = image.unsqueeze(0).to(device)
        num_imgs = 180

        ''' # Stage 1.4 Camera loading'''
        if not if_use_video_cam:
            # prepare cameras
            K, cam_list = prepare_camera(resolution_x=H, resolution_y=W, num_views=num_imgs, stides=1)
            cameras = construct_camera(K, cam_list)

            if render_mode == 'novel_pose': # if poses are changed, cameras will be fixed
                intrics = torch.Tensor([K[0,0],K[1,1], 256, 256]).reshape(-1)
                model.decoder.renderer.image_size = [512, 512]

                assert cameras.shape[-1] == 20
                cameras[:, :4] = intrics
                cameras = cameras[cam_idx:cam_idx+1]
                num_imgs = smpl_params.shape[0]
                cameras = cameras.repeat(num_imgs, 1)
                cameras = cameras[:, None, :] # length of the pose sequences
                print("modify the render images's resolution into 512x512 ")

            elif render_mode in ['reconstruct']: # using reference smplify's smplx and camera when available
                if 'intri_rec' in locals() and 'RT_rec' in locals():
                    cameras = torch.concat([intri_rec.reshape(-1,4), RT_rec.reshape(-1, 16)], dim=1)
                    model.decoder.renderer.image_size = [W_rec, H_rec]; print(f"modify the render images's resolution into {H_rec}x{W_rec}")
                    cameras = cameras.reshape(1,1,20).expand(num_imgs,1,-1)
                    cameras = cameras.cuda()
                else:
                    # Fallback: use default prepared cameras similar to novel_pose_A
                    model.decoder.renderer.image_size = [W, H]
                    cameras = cameras[0].reshape(1,1,20).expand(num_imgs,1,-1)

            elif render_mode == 'novel_pose_A':
                model.decoder.renderer.image_size = [W, H]
                cameras = cameras[0].reshape(1,1,20).expand(num_imgs,1,-1)

        elif if_use_video_cam: # for the animation with motion-x
            cameras = construct_camera_from_motionx(smplx_path)
            H, W = 2*cameras[0, 0, [3]].int().item(), 2*cameras[0,0, [2]].int().item()
            model.decoder.renderer.image_size = [W, H]; print(f"modify the render images's resolution into {H}x{W}")
            # model.decoder.renderer =

        ###############################################################################
        # Stage 2: Reconstruction.
        ###############################################################################

        sample = image_bs[[0]] # N, 3, H, W,
        # if if_use_dataset:
        #     sample = rearrange(sample, 'b h w c -> b c h w') # N, 3, H, W,

        image_path_idx = os.path.join(args.output_folders['ref'], f'{name}_ref.jpg')
        torchvision.utils.save_image( sample[0], image_path_idx)

        with torch.no_grad():
            # get latents
            code = model.forward_image_to_uv(sample, is_training=False)

        with torch.no_grad():
            output_list = []
            num_imgs_batch = 5
            total_frames = min(smpl_params.shape[0],300)
            res_uv = None
            for i in tqdm(range(0, total_frames, num_imgs_batch)):
                if i+num_imgs_batch > total_frames:
                    num_imgs_batch = total_frames - i
                code_bt = code.expand(num_imgs_batch, -1, -1, -1)
                # cameras_bt = cameras.expand(num_imgs_batch, -1, -1)
                cameras_bt = cameras[i:i+num_imgs_batch]

                if render_mode in ['reconstruct', 'novel_pose_A'] and res_uv is not None:
                    pass
                else:
                    res_uv = model.decoder._decode_feature(code_bt) # Decouple UV attributes
                    res_points = model.decoder._sample_feature(res_uv) # Sampling
                # Animate
                res_def_points = model.decoder.deform_pcd(res_points, smpl_params[i:i+num_imgs_batch].to(code_bt.dtype), zeros_hands_off=True, value=0.02)
                output = model.decoder.forward_render(res_def_points, cameras_bt.to(code_bt.dtype), num_imgs=1)
                image = output["image"][:, 0].cpu().to(torch.float32)

                print("output shape ", output["image"][:, 0].shape)
                output_list.append(image) # [:, 0] stands to get the all scenes (poses)
                del output

            output = torch.concatenate(output_list, 0)
            frames = rearrange(output, "b h w c -> b c h w")#.cpu().numpy()

            video_path_idx = os.path.join(args.output_folders['video'], f'{name}.mp4')

            save_video(
                frames[:,:4,...].to(torch.float32),
                video_path_idx,
            )
            image_plist.append(frames)
            print("saving into ", video_path_idx)
    return image_plist

def setup_directories(base_path, config_name):
    """Create output directories for results"""
    dirs = {
        'image': os.path.join(base_path, config_name, 'images'),
        'video': os.path.join(base_path, config_name),
        'ref': os.path.join(base_path, config_name)
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs

def main():
    """Main execution function"""
    # Parse arguments and set random seed
    args = parse_args()

    args.config = "configs/idol_v0.yaml"
    args.resume_path = "work_dirs/ckpt/model.ckpt"

    config = OmegaConf.load(args.config)
    config_name = os.path.basename(args.config).replace('.yaml', '')
    model_config = config.model

    resume_path =  args.resume_path
    # Initialize model
    model = instantiate_from_config(model_config)
    model.encoder = model.encoder.to(torch.bfloat16) ; print("moving encoder to bf16")
    model = model.__class__.load_from_checkpoint(resume_path, **config.model.params)
    model = model.to(device)
    model = model.eval()

    # Setup input paths: iterate all images in test_data_img/all
    import glob
    test_dir = args.input_path
    img_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    img_paths_list = []
    for ext in img_exts:
        img_paths_list.extend(sorted(glob.glob(os.path.join(test_dir, ext))))
    # Fallback to demo image if no images found
    if len(img_paths_list) == 0:
        img_paths_list = ['work_dirs/demo_data/4.jpg']
    # For each image, try to find a same-stem JSON next to it; else None
    smplx_ref_path_list = []
    for p in img_paths_list:
        stem = os.path.splitext(p)[0]
        candidate_json = stem + '.json'
        json_folder = os.path.dirname(candidate_json).replace("all", "out_smpl")
        smplx_ref_path_list.append(candidate_json if os.path.exists(candidate_json) else os.path.join(json_folder, os.path.basename(p).replace(".jpg", ".json")))
    # Use the provided driving sequence for all images (novel_pose path)
    smplx_path_driven_list = ['work_dirs/demo_data/Ways_to_Catch_360_clip1.json' for _ in img_paths_list]
    # smplx_path_driven_list = ['work_dirs/demo_data/finedance-5-144.npy.npy' for _ in img_paths_list]

    # Setup output directories
    args.output_path = args.output_path
    args.render_mode = 'reconstruct' # 'novel_pose_A' #'reconstruct' #'novel_pose'

    # make output directories
    args.output_folders = setup_directories(args.output_path, config_name)

    # Process data
    image_plist = process_data_on_gpu(
        args,
        model, 0,
        img_paths_list,
        smplx_ref_path_list,
        smplx_path_driven_list
    )

    return image_plist

if __name__ == "__main__":
    main()


