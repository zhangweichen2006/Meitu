import argparse
from mesh_estimator import HumanMeshEstimator


def make_parser():
    parser = argparse.ArgumentParser(description='CameraHMR Regressor')
    parser.add_argument("--image_folder", "--image_folder", type=str,
        help="Path to input image folder.")
    parser.add_argument("--output_folder", "--output_folder", type=str,
        help="Path to folder output folder.")
    parser.add_argument("--opacity", type=float, default=0.8,
        help="Overlay opacity for meshes in [0,1]; lower = more transparent.")
    parser.add_argument("--same_mesh_color", action="store_true",
        help="Use same color for all people instead of per-person hues.")
    parser.add_argument("--img_output_only", action="store_true",
        help="Only output the image with the mesh overlayed on top.")
    return parser

def main():

    parser = make_parser()
    args = parser.parse_args()
    estimator = HumanMeshEstimator(mesh_opacity=args.opacity, same_mesh_color=args.same_mesh_color, img_output_only=args.img_output_only)
    estimator.run_on_images(args.image_folder, args.output_folder)

if __name__=='__main__':
    main()

