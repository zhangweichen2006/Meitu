import torch
import argparse
import os
import os.path as osp
import math
from torchvision import transforms
from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3
import numpy as np
from PIL import Image

if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run inference with the Pi3 model.")

    parser.add_argument("--data_path", type=str, default='examples/skating.mp4',
                        help="Path to the input image directory or a video file.")
    parser.add_argument("--save_path", type=str, default='examples/result.ply',
                        help="Path to save the output .ply file.")
    parser.add_argument("--interval", type=int, default=-1,
                        help="Interval to sample image. Default: 1 for images dir, 10 for video")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to the model checkpoint file. Default: None")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")
    parser.add_argument("--individual_image", action='store_true',
                        help="Process a single image file instead of directory/video")
    parser.add_argument("--mask_path", type=str, default=None,
                        help="Path to alpha mask image for background removal")

    args = parser.parse_args()
    if args.interval < 0:
        args.interval = 10 if args.data_path.endswith('.mp4') else 1
    print(f'Sampling interval: {args.interval}')

    # from pi3.utils.debug import setup_debug
    # setup_debug()

    # 1. Prepare model
    print(f"Loading model...")
    device = torch.device(args.device)
    if args.ckpt is not None:
        model = Pi3().to(device).eval()
        if args.ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(args.ckpt)
        else:
            weight = torch.load(args.ckpt, map_location=device, weights_only=False)

        model.load_state_dict(weight)
    else:
        model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
        # or download checkpoints from `https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors`, and `--ckpt ckpts/model.safetensors`

    # Helper to decide if filenames follow a similar pattern (e.g., same prefix)
    def has_similar_name_pattern(directory_path: str) -> bool:
        image_files = sorted([f for f in os.listdir(directory_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        stems = [osp.splitext(f)[0] for f in image_files]
        if len(stems) <= 1:
            return True
        common = os.path.commonprefix(stems)
        return len(common) >= 5

    # If data_path is a directory and filenames are dissimilar, run per-image
    if osp.isdir(args.data_path) and not has_similar_name_pattern(args.data_path):
        out_dir = osp.dirname(args.save_path) if osp.splitext(args.save_path)[1] else args.save_path
        out_dir = out_dir if out_dir else 'output'
        os.makedirs(out_dir, exist_ok=True)

        # Precompute device/autocast settings
        use_cuda = (device.type == 'cuda') and torch.cuda.is_available()
        if use_cuda:
            try:
                dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            except Exception:
                dtype = torch.float16
        else:
            dtype = torch.float32

        # Image loader matching utils resizing policy
        to_tensor = transforms.ToTensor()

        image_files = sorted([f for f in os.listdir(args.data_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        print(f"Detected dissimilar filenames in directory. Running per-image inference on {len(image_files)} images...")
        for fname in image_files:
            in_path = osp.join(args.data_path, fname)
            stem = osp.splitext(fname)[0]
            ply_path = osp.join(out_dir, f"{stem}.ply")
            png_path = osp.join(out_dir, f"{stem}.png")

            try:
                img_pil = Image.open(in_path).convert('RGB')
            except Exception as e:
                print(f"[WARN] Skipping {fname}: {e}")
                continue

            # Resize to respect pixel limit and multiples of 14 (same as utils)
            W_orig, H_orig = img_pil.size
            PIXEL_LIMIT = 255000
            scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
            W_target, H_target = W_orig * scale, H_orig * scale
            k, m = round(W_target / 14), round(H_target / 14)
            while (k * 14) * (m * 14) > PIXEL_LIMIT:
                if k / m > W_target / H_target:
                    k -= 1
                else:
                    m -= 1
            TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
            img_resized = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            img_tensor = to_tensor(img_resized).unsqueeze(0).to(device)  # (N=1, C, H, W)

            print(f"Running model inference for {fname}...")
            with torch.no_grad():
                if use_cuda:
                    with torch.amp.autocast('cuda', dtype=dtype):
                        res = model(img_tensor[None])
                else:
                    res = model(img_tensor[None].to(dtype))

            # Masks
            masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
            non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
            masks = torch.logical_and(masks, non_edge)[0]

            # Save PLY
            print(f"Saving point cloud to: {ply_path}")
            write_ply(res['points'][0][masks].cpu(), img_tensor.permute(0, 2, 3, 1)[masks], ply_path)

            # Render PNGs: default, +45° yaw (right), -45° yaw (left)
            try:
                H, W = img_tensor.shape[-2], img_tensor.shape[-1]
                cam2world = res['camera_poses'][0, 0]
                base_world2cam = torch.linalg.inv(cam2world)
                pts_world = res['points'][0].reshape(-1, 3)
                cols = img_tensor.permute(0, 2, 3, 1).reshape(-1, 3)
                mask_flat = masks.reshape(-1)
                pts_world = pts_world[mask_flat]
                cols = cols[mask_flat]

                def render_with_world2cam(world2cam_mat: torch.Tensor, out_path: str):
                    if pts_world.numel() == 0:
                        print("[WARN] No valid points after masking; skipping PNG save.")
                        return
                    ones = torch.ones((pts_world.shape[0], 1), device=pts_world.device, dtype=pts_world.dtype)
                    pts_world_h = torch.cat([pts_world, ones], dim=1)
                    pts_cam_h = (world2cam_mat @ pts_world_h.T).T
                    pts_cam = pts_cam_h[:, :3]
                    z = pts_cam[:, 2]
                    front = z > 1e-6
                    if not front.any():
                        print("[WARN] No points in front of camera; skipping", out_path)
                        return
                    pts_cam = pts_cam[front]
                    cols2 = cols[front]
                    z = z[front]
                    fx = float(max(H, W))
                    fy = float(max(H, W))
                    cx = float(W) / 2.0
                    cy = float(H) / 2.0
                    u = fx * (pts_cam[:, 0] / z) + cx
                    v = fy * (pts_cam[:, 1] / z) + cy
                    x = torch.round(u).to(torch.int64)
                    y = torch.round(v).to(torch.int64)
                    in_bounds = (x >= 0) & (x < W) & (y >= 0) & (y < H)
                    if not in_bounds.any():
                        print("[WARN] No projected points inside bounds; skipping", out_path)
                        return
                    x = x[in_bounds]
                    y = y[in_bounds]
                    z = z[in_bounds]
                    cols2 = cols2[in_bounds]
                    lin_idx = (y * W + x).detach().cpu().numpy().astype(np.int64)
                    z_np = z.detach().cpu().numpy()
                    cols_np = (cols2.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                    order = np.lexsort((z_np, lin_idx))
                    lin_sorted = lin_idx[order]
                    first_pos = np.unique(lin_sorted, return_index=True)[1]
                    sel = order[first_pos]
                    img_out = np.zeros((H, W, 3), dtype=np.uint8)
                    xs = lin_idx[sel] % W
                    ys = lin_idx[sel] // W
                    img_out[ys, xs] = cols_np[sel]
                    Image.fromarray(img_out).save(out_path)
                    print(f"Saved rendered image to: {out_path}")

                # 1. Default/Original view
                render_with_world2cam(base_world2cam, png_path)
                print("[DEBUG] Finished rendering original view, starting rotated views...")

                # 2-7. Six rotated views: +/- yaw, pitch, roll (75° each)
                rotation_angle = math.radians(75.0)
                rotations = [
                    ('yaw', rotation_angle, '_yaw_pos'),
                    ('yaw', -rotation_angle, '_yaw_neg'),
                    ('pitch', rotation_angle, '_pitch_pos'),
                    ('pitch', -rotation_angle, '_pitch_neg'),
                    ('roll', rotation_angle, '_roll_pos'),
                    ('roll', -rotation_angle, '_roll_neg')
                ]

                print(f"Generating {len(rotations)} additional rotated views...")
                for axis, angle, suffix in rotations:
                    rotated_world2cam = create_rotated_camera(axis, angle)
                    rotated_path = png_path.rsplit('.', 1)[0] + suffix + '.png'

                    # Render with the rotated camera
                    if pts_world.numel() == 0:
                        print(f"[WARN] No valid points for {suffix} view")
                        continue
                    ones = torch.ones((pts_world.shape[0], 1), device=pts_world.device, dtype=pts_world.dtype)
                    pts_world_h = torch.cat([pts_world, ones], dim=1)
                    pts_cam_h = (rotated_world2cam @ pts_world_h.T).T
                    pts_cam = pts_cam_h[:, :3]
                    z = pts_cam[:, 2]

                    # Debug: Check camera transformation
                    centroid_h = torch.cat([pts_centroid, torch.ones(1, device=pts_world.device, dtype=pts_world.dtype)])
                    centroid_cam = (rotated_world2cam @ centroid_h)[:3]
                    print(f"[DEBUG] {suffix}: centroid in camera coords = {centroid_cam.cpu().numpy()}")

                    # Use a very conservative threshold and ensure centroid is in front
                    front = z > 0.01  # More conservative threshold
                    if not front.any() or centroid_cam[2] <= 0.01:
                        print(f"[DEBUG] No points in front of camera for {suffix} view (centroid_z={centroid_cam[2]:.3f})")
                        continue
                    pts_cam = pts_cam[front]
                    cols_filtered = cols[front]
                    z = z[front]

                    # Match original FOV to keep object size consistent
                    fov_scale = 1.0
                    fx = float(max(H, W)) * fov_scale
                    fy = float(max(H, W)) * fov_scale
                    cx = float(W) / 2.0
                    cy = float(H) / 2.0
                    u = fx * (pts_cam[:, 0] / z) + cx
                    v = fy * (pts_cam[:, 1] / z) + cy
                    x = torch.round(u).to(torch.int64)
                    y = torch.round(v).to(torch.int64)
                    in_bounds = (x >= 0) & (x < W) & (y >= 0) & (y < H)
                    if not in_bounds.any():
                        print(f"[WARN] No projected points inside bounds for {suffix} view")
                        continue
                    x = x[in_bounds]
                    y = y[in_bounds]
                    z = z[in_bounds]
                    cols_filtered = cols_filtered[in_bounds]
                    lin_idx = (y * W + x).detach().cpu().numpy().astype(np.int64)
                    z_np = z.detach().cpu().numpy()
                    cols_np = (cols_filtered.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                    order = np.lexsort((z_np, lin_idx))
                    lin_sorted = lin_idx[order]
                    first_pos = np.unique(lin_sorted, return_index=True)[1]
                    sel = order[first_pos]
                    img_out = np.zeros((H, W, 3), dtype=np.uint8)
                    xs = lin_idx[sel] % W
                    ys = lin_idx[sel] // W
                    img_out[ys, xs] = cols_np[sel]
                    Image.fromarray(img_out).save(rotated_path)
                    print(f"Saved {axis} {math.degrees(angle):.0f}° view to: {rotated_path}")
                # Compose a 2x3 grid of the six rotated views
                try:
                    base_noext = png_path.rsplit('.', 1)[0]
                    suffixes = ['_yaw_pos', '_pitch_pos', '_roll_pos', '_yaw_neg', '_pitch_neg', '_roll_neg']
                    tile_paths = [base_noext + s + '.png' for s in suffixes]

                    tiles = []
                    for p in tile_paths:
                        if osp.isfile(p):
                            tiles.append(Image.open(p).convert('RGB').resize((W, H), Image.Resampling.NEAREST))
                        else:
                            tiles.append(Image.new('RGB', (W, H), (0, 0, 0)))

                    grid = Image.new('RGB', (W * 3, H * 2))
                    for idx, tile in enumerate(tiles):
                        r = idx // 3
                        c = idx % 3
                        grid.paste(tile, (c * W, r * H))

                    grid_path = base_noext + '_grid6.png'
                    grid.save(grid_path)
                    print(f"Saved 2x3 grid of six views to: {grid_path}")
                except Exception as e_grid:
                    print(f"[WARN] Failed to compose 2x3 grid: {e_grid}")
            except Exception as e:
                print(f"[WARN] Failed to render multi-view PNGs for {fname}: {e}")

        # Finished per-image processing
        exit(0)

    # 2. Prepare input data (sequence, video, or individual image)
    if args.individual_image:
        # Handle single image file
        if not osp.isfile(args.data_path):
            raise ValueError(f"Individual image file not found: {args.data_path}")

        # Load single image and convert to tensor format expected by model
        from PIL import Image, ImageFile
        import math

        # Enable loading of truncated images
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        print(f"Loading single image: {args.data_path}")
        try:
            img_pil = Image.open(args.data_path).convert('RGB')
        except (OSError, IOError) as e:
            print(f"Error loading image {args.data_path}: {e}")
            print("Skipping this image...")
            exit(1)

        # Load and apply mask if provided
        mask_tensor = None
        if args.mask_path and osp.isfile(args.mask_path):
            print(f"Loading mask: {args.mask_path}")
            try:
                # Load mask as grayscale and convert to binary mask
                mask_pil = Image.open(args.mask_path).convert('L')
                # Resize mask to match image size before processing
                W_orig, H_orig = img_pil.size
                mask_pil = mask_pil.resize((W_orig, H_orig), Image.Resampling.LANCZOS)
                # Convert to binary mask (threshold at 128)
                mask_array = np.array(mask_pil) > 128
                # Apply mask to image (set background to black)
                img_array = np.array(img_pil)
                img_array[~mask_array] = [0, 0, 0]  # Set background pixels to black
                img_pil = Image.fromarray(img_array)
                print("Applied alpha mask to remove background")
            except Exception as e:
                print(f"Warning: Could not load mask {args.mask_path}: {e}")
                print("Proceeding without mask...")
        elif args.mask_path:
            print(f"Warning: Mask file not found: {args.mask_path}")
            print("Proceeding without mask...")

        # Apply same resizing logic as load_images_as_tensor
        W_orig, H_orig = img_pil.size
        PIXEL_LIMIT = 255000
        scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
        W_target, H_target = W_orig * scale, H_orig * scale
        k, m = round(W_target / 14), round(H_target / 14)
        while (k * 14) * (m * 14) > PIXEL_LIMIT:
            if k / m > W_target / H_target:
                k -= 1
            else:
                m -= 1
        TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
        print(f"Resizing image to: ({TARGET_W}, {TARGET_H})")

        # Resize and convert to tensor
        resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
        img_tensor = transforms.ToTensor()(resized_img).unsqueeze(0).to(device)  # (1, 3, H, W)
        imgs = img_tensor

        # Create additional mask tensor for background filtering if mask was applied
        additional_mask = None
        if args.mask_path and osp.isfile(args.mask_path):
            try:
                # Create a tensor mask at the target resolution for additional filtering
                mask_resized = mask_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
                mask_tensor_full = transforms.ToTensor()(mask_resized).squeeze(0)  # (H, W)
                additional_mask = (mask_tensor_full > 0.5).to(device)  # Binary mask
            except Exception as e:
                print(f"Warning: Could not create additional mask tensor: {e}")
                additional_mask = None
    else:
        # The load_images_as_tensor function will print the loading path
        imgs = load_images_as_tensor(args.data_path, interval=args.interval).to(device) # (N, 3, H, W)

    # 3. Infer
    print("Running model inference...")
    # Determine precision and autocast context based on device
    use_cuda = (device.type == 'cuda') and torch.cuda.is_available()
    if use_cuda:
        try:
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        except Exception:
            dtype = torch.float16
    else:
        dtype = torch.float32

    with torch.no_grad():
        if use_cuda:
            with torch.amp.autocast('cuda', dtype=dtype):
                res = model(imgs[None]) # Add batch dimension
        else:
            # CPU path without CUDA autocast
            res = model(imgs[None].to(dtype))

    # 4. process mask
    masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
    non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
    masks = torch.logical_and(masks, non_edge)[0]

    # Apply additional alpha mask if available (for individual image processing)
    if args.individual_image and 'additional_mask' in locals() and additional_mask is not None:
        print("Applying additional alpha mask to filter background points...")
        # Expand additional_mask to match the masks shape if needed
        if additional_mask.shape != masks.shape[-2:]:
            additional_mask = torch.nn.functional.interpolate(
                additional_mask.unsqueeze(0).unsqueeze(0).float(),
                size=masks.shape[-2:],
                mode='nearest'
            ).squeeze().bool()
        # Apply the mask to filter out background points
        masks = torch.logical_and(masks, additional_mask.unsqueeze(0))

    # 5. Save points
    print(f"Saving point cloud to: {args.save_path}")
    write_ply(res['points'][0][masks].cpu(), imgs.permute(0, 2, 3, 1)[masks], args.save_path)
    print("Done.")

    # 6. Render and save a PNG from estimated camera (same resolution as input)
    try:
        H, W = imgs.shape[-2], imgs.shape[-1]
        # Choose reference camera (first frame)
        cam2world = res['camera_poses'][0, 0]  # (4,4)
        world2cam = torch.linalg.inv(cam2world)

        # Gather masked global points and their colors
        pts_world = res['points'][0].reshape(-1, 3)
        cols = imgs.permute(0, 2, 3, 1).reshape(-1, 3)
        mask_flat = masks.reshape(-1)
        pts_world = pts_world[mask_flat]
        cols = cols[mask_flat]

        if pts_world.numel() > 0:
            ones = torch.ones((pts_world.shape[0], 1), device=pts_world.device, dtype=pts_world.dtype)
            pts_world_h = torch.cat([pts_world, ones], dim=1)  # (M,4)
            pts_cam_h = (world2cam @ pts_world_h.T).T  # (M,4)
            pts_cam = pts_cam_h[:, :3]

            # Keep points in front of the camera
            z = pts_cam[:, 2]
            front = z > 1e-6
            pts_cam = pts_cam[front]
            cols = cols[front]
            z = z[front]

            # Simple intrinsics (pinhole) with principal point at center
            fx = float(max(H, W))
            fy = float(max(H, W))
            cx = float(W) / 2.0
            cy = float(H) / 2.0

            u = fx * (pts_cam[:, 0] / z) + cx
            v = fy * (pts_cam[:, 1] / z) + cy

            x = torch.round(u).to(torch.int64)
            y = torch.round(v).to(torch.int64)

            in_bounds = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            if in_bounds.any():
                x = x[in_bounds]
                y = y[in_bounds]
                z = z[in_bounds]
                cols_def = cols[in_bounds]

                # Prepare z-buffer compositing on CPU with NumPy
                lin_idx = (y * W + x).detach().cpu().numpy().astype(np.int64)
                z_np = z.detach().cpu().numpy()
                cols_np = (cols_def.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

                # Sort by pixel index (primary) then depth (secondary)
                order = np.lexsort((z_np, lin_idx))
                lin_sorted = lin_idx[order]
                first_pos = np.unique(lin_sorted, return_index=True)[1]
                sel = order[first_pos]

                img_out = np.zeros((H, W, 3), dtype=np.uint8)
                xs = lin_idx[sel] % W
                ys = lin_idx[sel] // W
                img_out[ys, xs] = cols_np[sel]

                png_path = args.save_path.rsplit('.', 1)[0] + '.png'
                Image.fromarray(img_out).save(png_path)
                print(f"Saved rendered image to: {png_path}")

                # Also render +75° (right) and -75° (left) yaw views
                try:
                    # Calculate the centroid of all 3D points to look at
                    pts_centroid = pts_world.mean(dim=0)  # (3,) - center of point cloud

                    # Extract camera position from original camera matrix
                    base_cam2world = torch.linalg.inv(world2cam)
                    original_cam_pos = base_cam2world[:3, 3]  # (3,) - original camera position

                    # Calculate distance from camera to centroid for consistent viewing distance
                    cam_to_centroid = pts_centroid - original_cam_pos
                    viewing_distance = torch.norm(cam_to_centroid)

                    rotation_angle = math.radians(75.0)  # 75 degrees rotation

                    def create_rotated_camera(axis: str, angle: float) -> torch.Tensor:
                        """Step 1: Rotate camera position around object center on specified axis. Step 2: Point camera at object center"""

                        # Get the vector from object center to original camera position
                        center_to_cam = original_cam_pos - pts_centroid  # Vector from center to camera

                        # Keep camera distance to object center constant (no scaling)
                        # This preserves subject size across rotated views
                        # center_to_cam remains unchanged here

                        # Step 1: Rotate the camera position around the specified axis
                        cos_angle = math.cos(angle)
                        sin_angle = math.sin(angle)

                        if axis == 'yaw':  # Rotation around Y-axis
                            rotated_center_to_cam = torch.tensor([
                                cos_angle * center_to_cam[0] + sin_angle * center_to_cam[2],
                                center_to_cam[1],  # Y unchanged
                                -sin_angle * center_to_cam[0] + cos_angle * center_to_cam[2]
                            ], device=pts_world.device, dtype=pts_world.dtype)
                        elif axis == 'pitch':  # Rotation around X-axis
                            rotated_center_to_cam = torch.tensor([
                                center_to_cam[0],  # X unchanged
                                cos_angle * center_to_cam[1] - sin_angle * center_to_cam[2],
                                sin_angle * center_to_cam[1] + cos_angle * center_to_cam[2]
                            ], device=pts_world.device, dtype=pts_world.dtype)
                        elif axis == 'roll':  # Rotation around Z-axis
                            rotated_center_to_cam = torch.tensor([
                                cos_angle * center_to_cam[0] - sin_angle * center_to_cam[1],
                                sin_angle * center_to_cam[0] + cos_angle * center_to_cam[1],
                                center_to_cam[2]  # Z unchanged
                            ], device=pts_world.device, dtype=pts_world.dtype)

                        # New camera position after rotation around object center
                        rotated_cam_pos = pts_centroid + rotated_center_to_cam

                        # Step 2: Point the camera back toward the object center (look-at)
                        look_dir = torch.nn.functional.normalize(pts_centroid - rotated_cam_pos, dim=0)

                        # Create orthogonal camera basis
                        world_up = torch.tensor([0.0, 1.0, 0.0], device=pts_world.device, dtype=pts_world.dtype)

                        # Handle degenerate case where look_dir is parallel to world_up
                        if torch.abs(torch.dot(look_dir, world_up)) > 0.99:
                            world_up = torch.tensor([1.0, 0.0, 0.0], device=pts_world.device, dtype=pts_world.dtype)

                        right_dir = torch.nn.functional.normalize(torch.cross(look_dir, world_up, dim=0), dim=0)
                        up_dir = torch.nn.functional.normalize(torch.cross(right_dir, look_dir, dim=0), dim=0)

                        print(f"[DEBUG] {axis} {math.degrees(angle):.1f}°: orig_pos={original_cam_pos.cpu().numpy()}, rotated_pos={rotated_cam_pos.cpu().numpy()}, center={pts_centroid.cpu().numpy()}")

                        # Build camera-to-world matrix
                        cam2world = torch.eye(4, device=pts_world.device, dtype=pts_world.dtype)
                        cam2world[:3, 0] = right_dir
                        cam2world[:3, 1] = up_dir
                        # Model projection assumes camera looks along +Z in camera frame.
                        cam2world[:3, 2] = look_dir
                        cam2world[:3, 3] = rotated_cam_pos

                        # Return world-to-camera matrix
                        return torch.linalg.inv(cam2world)

                    def render_with_world2cam(world2cam_mat: torch.Tensor, out_suffix: str):
                        if pts_world.numel() == 0:
                            return
                        ones2 = torch.ones((pts_world.shape[0], 1), device=pts_world.device, dtype=pts_world.dtype)
                        pts_world_h2 = torch.cat([pts_world, ones2], dim=1)
                        pts_cam_h2 = (world2cam_mat @ pts_world_h2.T).T
                        pts_cam2 = pts_cam_h2[:, :3]
                        z2 = pts_cam2[:, 2]
                        front2 = z2 > 1e-6
                        if not front2.any():
                            print(f"[DEBUG] No points in front of camera for {out_suffix} view")
                            return
                        pts_cam2 = pts_cam2[front2]
                        cols2 = cols[front2]
                        z2 = z2[front2]
                        # Use original FOV scale to preserve apparent size
                        fov_scale = 1.0
                        fx2 = float(max(H, W)) * fov_scale
                        fy2 = float(max(H, W)) * fov_scale
                        cx2 = float(W) / 2.0
                        cy2 = float(H) / 2.0
                        u2 = fx2 * (pts_cam2[:, 0] / z2) + cx2
                        v2 = fy2 * (pts_cam2[:, 1] / z2) + cy2
                        x2 = torch.round(u2).to(torch.int64)
                        y2 = torch.round(v2).to(torch.int64)
                        in_bounds2 = (x2 >= 0) & (x2 < W) & (y2 >= 0) & (y2 < H)
                        if not in_bounds2.any():
                            print(f"[DEBUG] No projected points in bounds for {out_suffix} view")
                            return
                        x2 = x2[in_bounds2]
                        y2 = y2[in_bounds2]
                        z2 = z2[in_bounds2]
                        cols2 = cols2[in_bounds2]
                        lin_idx2 = (y2 * W + x2).detach().cpu().numpy().astype(np.int64)
                        z_np2 = z2.detach().cpu().numpy()
                        cols_np2 = (cols2.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                        order2 = np.lexsort((z_np2, lin_idx2))
                        lin_sorted2 = lin_idx2[order2]
                        first_pos2 = np.unique(lin_sorted2, return_index=True)[1]
                        sel2 = order2[first_pos2]
                        img_out2 = np.zeros((H, W, 3), dtype=np.uint8)
                        xs2 = lin_idx2[sel2] % W
                        ys2 = lin_idx2[sel2] // W
                        img_out2[ys2, xs2] = cols_np2[sel2]
                        out_path = png_path.rsplit('.', 1)[0] + out_suffix + '.png'
                        Image.fromarray(img_out2).save(out_path)
                        print(f"Saved rendered image to: {out_path}")

                    # Create 6 rotated views: yaw, pitch, roll (+/-75°)
                    rotation_angle = math.radians(75.0)
                    rotations = [
                        ('yaw', rotation_angle, '_yaw_pos'),
                        ('yaw', -rotation_angle, '_yaw_neg'),
                        ('pitch', rotation_angle, '_pitch_pos'),
                        ('pitch', -rotation_angle, '_pitch_neg'),
                        ('roll', rotation_angle, '_roll_pos'),
                        ('roll', -rotation_angle, '_roll_neg'),
                    ]
                    for axis, angle, suffix in rotations:
                        rotated_world2cam = create_rotated_camera(axis, angle)
                        render_with_world2cam(rotated_world2cam, suffix)
                    # Compose a 2x3 grid of the six rotated views
                    try:
                        base_noext = png_path.rsplit('.', 1)[0]
                        suffixes = ['_yaw_pos', '_pitch_pos', '_roll_pos', '_yaw_neg', '_pitch_neg', '_roll_neg']
                        tile_paths = [base_noext + s + '.png' for s in suffixes]
                        tiles = []
                        for p in tile_paths:
                            if osp.isfile(p):
                                tiles.append(Image.open(p).convert('RGB').resize((W, H), Image.Resampling.NEAREST))
                            else:
                                tiles.append(Image.new('RGB', (W, H), (0, 0, 0)))
                        grid = Image.new('RGB', (W * 3, H * 2))
                        for idx, tile in enumerate(tiles):
                            r = idx // 3
                            c = idx % 3
                            grid.paste(tile, (c * W, r * H))
                        grid_path = base_noext + '_grid6.png'
                        grid.save(grid_path)
                        print(f"Saved 2x3 grid of six views to: {grid_path}")
                    except Exception as e_grid:
                        print(f"[WARN] Failed to compose 2x3 grid: {e_grid}")
                except Exception as _e:
                    print(f"[WARN] Failed to render yawed views: {_e}")

                # (duplicate yaw block removed)
            else:
                print("[WARN] No projected points fell inside image bounds; skipping PNG save.")
        else:
            print("[WARN] No valid points after masking; skipping PNG save.")
    except Exception as e:
        print(f"[WARN] Failed to render PNG: {e}")