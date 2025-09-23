import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import argparse
from mesh_estimator import HumanMeshEstimator


def make_parser():
    parser = argparse.ArgumentParser(description='CameraHMR Regressor')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image_folder", type=str,
        help="Path to input image folder.")
    input_group.add_argument("--video", type=str,
        help="Path to input video file.")
    parser.add_argument("--output_folder", type=str,
        help="Path to folder output folder.")
    parser.add_argument("--output_cam", type=str,
        help="Path to folder output camera folder.")
    parser.add_argument("--export_init_npz", type=str,
        help="Path to write init params (.npz) for CamSMPLify optimize.py")
    parser.add_argument("--opacity", type=float, default=0.5,
        help="Overlay opacity for meshes in [0,1]; lower = more transparent.")
    parser.add_argument("--same_mesh_color", action="store_true",
        help="Use same color for all people instead of per-person hues.")
    parser.add_argument("--save_smpl_obj", action="store_true",
        help="Only output the image with the mesh overlayed on top.")
    parser.add_argument("--use_smplify", action="store_true",
        help="Enable CamSMPLify refinement using 2D dense keypoints and intrinsics.")
    input_group.add_argument("--model_path", type=str,
        help="Path to model checkpoint.")
    return parser

def main():

    parser = make_parser()
    args = parser.parse_args()
    estimator = HumanMeshEstimator(mesh_opacity=args.opacity,
                                   same_mesh_color=args.same_mesh_color,
                                   save_smpl_obj=args.save_smpl_obj,
                                   use_smplify=args.use_smplify,
                                   export_init_npz=args.export_init_npz,
                                   model_path=args.model_path)
    if args.video:
        estimator.run_on_video(args.video, args.output_folder, args.output_cam)
    else:
        if not args.output_folder:
            raise ValueError("--output_folder is required when using --image_folder")
        estimator.run_on_images(args.image_folder, args.output_folder, args.output_cam)

if __name__=='__main__':
    main()
