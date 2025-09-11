import os
import argparse
import numpy as np
import torch
from cam_smplify import SMPLify
import cv2

def main(args):

    init_param_file = args.input
    image_base_dir = args.image_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_file_path =  os.path.join(args.output_dir, "output.npz")

    smplify = SMPLify(vis=args.vis, verbose=args.verbose, save_vis_dir=args.save_vis_dir)
    if args.save_dense_kp_dir is not None and len(str(args.save_dense_kp_dir).strip()) > 0:
        os.makedirs(args.save_dense_kp_dir, exist_ok=True)
    coco_data = np.load(init_param_file, allow_pickle=True)

    processed_data = {key: [] for key in coco_data}

    for i in range(len(coco_data['imgname'])):
        img_path = os.path.join(image_base_dir, coco_data["imgname"][i])
        print(f"Processing: {img_path}")

        if not os.path.exists(img_path):
            # print(f"File not found: ÷{img_path}")
            img_path = coco_data["imgname"][i]
            if not os.path.exists(img_path):
                print(f"File not found: {img_path}")
                continue

        # Extract data
        pose = np.expand_dims(coco_data["pose"][i], axis=0)
        betas = np.expand_dims(coco_data["shape"][i], axis=0)
        cam_int = torch.tensor(coco_data["cam_int"][i])
        cam_t = torch.tensor(coco_data["cam_t"][i])
        center = torch.tensor(coco_data["center"][i])
        scale = torch.tensor(coco_data["scale"][i] / 200.0)
        dense_kp = coco_data["dense_kp"][i]
        if "gt_keypoints" in coco_data:
            keypoints_2d = coco_data["gt_keypoints"][i]
        else:
            keypoints_2d = None

        # Optionally render and save provided keypoints
        if args.save_dense_kp_dir is not None and ("gt_keypoints" in coco_data):
            try:
                img_bgr = cv2.imread(img_path) if os.path.exists(img_path) else None
                if img_bgr is None:
                    alt = coco_data["imgname"][i] if "imgname" in coco_data else None
                    if alt and os.path.exists(alt):
                        img_bgr = cv2.imread(alt)
                if img_bgr is not None:
                    kps = coco_data["gt_keypoints"][i]
                    if hasattr(kps, 'shape') and kps.shape[-1] >= 2:
                        vis_img = img_bgr.copy()
                        for (x, y, c) in kps:
                            if c is None:
                                continue
                            try:
                                conf = float(c)
                            except Exception:
                                conf = 1.0
                            color = (0, 255, 0) if conf >= 0.5 else (0, 165, 255)
                            cv2.circle(vis_img, (int(x), int(y)), 3, color, thickness=-1)
                        stem = os.path.splitext(os.path.basename(img_path))[0]
                        out_path = os.path.join(args.save_dense_kp_dir, f"{stem}_kp.png")
                        cv2.imwrite(out_path, vis_img)
            except Exception as e:
                print(f"[WARN] Failed to save keypoint visualization for {img_path}: {e}")

        # Run SMPLify optimization
        result = smplify(args, pose, betas, cam_t, center, scale, cam_int, img_path, keypoints_2d, dense_kp, i)

        if result:
            processed_data["imgname"].append(img_path)
            processed_data["center"].append(coco_data["center"][i])
            processed_data["scale"].append(coco_data["scale"][i])
            processed_data["cam_int"].append(coco_data["cam_int"][i])
            if "gt_keypoints" in coco_data:
                processed_data["gt_keypoints"].append(coco_data["gt_keypoints"][i])
            else:
                processed_data["gt_keypoints"].append(None)
            processed_data["cam_t"].append(result["camera_translation"].detach().cpu().numpy())
            processed_data["shape"].append(result["betas"][0].detach().cpu().numpy())

            body_pose = torch.hstack([result["global_orient"], result["pose"]]).detach().cpu().numpy()[0]
            processed_data["pose"].append(body_pose)

    # Save results
    np.savez(output_file_path, **processed_data)
    print(f"Processed data saved to {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SMPLify on a dataset")
    parser.add_argument("--input", type=str, default='data/demo_files_for_optimization/init_params/filtered_aic.npz', help="Path to the initial parameter file (.npz)")
    parser.add_argument("--output_dir", type=str, default='out_params', help="Directory to save output data")
    parser.add_argument("--image_dir", type=str, default='data/demo_files_for_optimization/demo_images', help="Path to the image dataset directory")
    parser.add_argument("--vis", type=bool, required=False, help="Visualization of fitting")
    parser.add_argument("--save_dense_kp_dir", type=str, default=None, help="Directory to save keypoints")
    parser.add_argument("--save_vis_dir", type=str, default=None, help="Directory to save headless visualizations")
    parser.add_argument("--verbose", type=bool, required=False, help="Print losses")
    parser.add_argument("--vis_int", type=int, default=100, required=False, help="Visualize result after every 100 iteration of optimization")
    parser.add_argument("--loss_cut", type=int, default=100, required=False, help="If initial loss is more than 100 we use high loss threshold else low loss threshold")
    parser.add_argument("--high_threshold", type=int, default=50, required=False, help="Loss threshold value to select the optimization result")
    parser.add_argument("--low_threshold", type=int, default=30, required=False, help="Loss threshold value to select the optimization result")

    args = parser.parse_args()
    main(args)

