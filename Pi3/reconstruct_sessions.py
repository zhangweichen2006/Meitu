import os
import os.path as osp
import json
import argparse
from typing import List, Tuple

import torch
import numpy as np

from pi3.models.pi3 import Pi3
from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".MP4", ".MOV"}


def path_has_images(path: str) -> bool:
    try:
        for name in os.listdir(path):
            _, ext = osp.splitext(name)
            if ext.lower() in SUPPORTED_IMAGE_EXTS:
                return True
        return False
    except Exception:
        return False


def collect_assets(session_dir: str) -> List[Tuple[str, str]]:
    """
    Collect processable assets in a session directory.
    Returns a list of (asset_path, asset_name) pairs. Asset can be:
      - the session_dir itself if it directly contains images
      - any immediate subdirectory that contains images
      - any immediate video file
    """
    assets: List[Tuple[str, str]] = []

    # Case 1: session root itself has images
    if path_has_images(session_dir):
        assets.append((session_dir, osp.basename(session_dir.rstrip("/"))))

    # Case 2: subdirectories with images
    try:
        for entry in os.listdir(session_dir):
            entry_path = osp.join(session_dir, entry)
            if osp.isdir(entry_path) and path_has_images(entry_path):
                assets.append((entry_path, entry))
    except FileNotFoundError:
        pass

    # Case 3: videos in session root
    try:
        for entry in os.listdir(session_dir):
            entry_path = osp.join(session_dir, entry)
            if osp.isfile(entry_path):
                _, ext = osp.splitext(entry)
                if ext in SUPPORTED_VIDEO_EXTS:
                    asset_name = osp.splitext(entry)[0]
                    assets.append((entry_path, asset_name))
    except FileNotFoundError:
        pass

    # Deduplicate in case of overlaps
    seen = set()
    unique_assets: List[Tuple[str, str]] = []
    for p, n in assets:
        if p not in seen:
            seen.add(p)
            unique_assets.append((p, n))

    return unique_assets


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_camera_geometry(camera_poses_tensor: torch.Tensor, out_dir: str, asset_name: str) -> None:
    """
    Save camera poses to both .npy and .json for portability.
    camera_poses_tensor expected shape: (N, 4, 4)
    """
    ensure_dir(out_dir)
    poses_cpu = camera_poses_tensor.detach().cpu()
    npy_path = osp.join(out_dir, f"{asset_name}_camera_poses.npy")
    json_path = osp.join(out_dir, f"{asset_name}_camera_poses.json")

    # Save .npy
    np.save(npy_path, poses_cpu.numpy())

    # Save .json (nested lists)
    poses_list = poses_cpu.tolist()
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"camera_poses": poses_list}, f)


def extract_pi3_embeddings(model: Pi3, imgs: torch.Tensor, device: torch.device, use_cuda: bool, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract intermediate Pi3 embeddings per frame.
    Returns:
      - embed_tokens: (N, patch_h, patch_w, D) tokens from decode (excluding special tokens)
      - embed_pooled: (N, D) average pooled per-frame embedding
    """
    # Normalize same as model.forward
    imgs_norm = (imgs - model.image_mean) / model.image_std
    N, _, H, W = imgs.shape

    with torch.no_grad():
        if use_cuda:
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                hidden = model.encoder(imgs_norm, is_training=True)
        else:
            hidden = model.encoder(imgs_norm, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        if use_cuda:
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                decoded_hidden, _ = model.decode(hidden, N, H, W)
        else:
            decoded_hidden, _ = model.decode(hidden, N, H, W)

    # Remove special tokens
    patch_start = model.patch_start_idx
    decoded_hidden = decoded_hidden[:, patch_start:, :]  # (N, hw, D)
    hw = decoded_hidden.shape[1]
    patch_h, patch_w = H // model.patch_size, W // model.patch_size
    assert hw == patch_h * patch_w, f"Mismatch: hw={hw}, patch_h*patch_w={patch_h*patch_w}"

    # Reshape to (N, patch_h, patch_w, D)
    embed_tokens = decoded_hidden.reshape(N, patch_h, patch_w, decoded_hidden.shape[-1])
    # Pooled per-frame embeddings (N, D)
    embed_pooled = decoded_hidden.mean(dim=1)

    return embed_tokens, embed_pooled


def run_inference_on_asset(model: Pi3, device: torch.device, asset_path: str, asset_name: str, interval: int, out_session_dir: str) -> None:
    # Determine default interval if user set negative
    if interval < 0:
        interval = 10 if asset_path.lower().endswith(".mp4") else 1

    print(f"[Pi3] Loading data from: {asset_path} (interval={interval})")
    imgs = load_images_as_tensor(asset_path, interval=interval)
    if imgs.numel() == 0:
        print(f"[Pi3][WARN] No frames/images loaded for {asset_path}. Skipping.")
        return
    imgs = imgs.to(device)  # (N, 3, H, W)

    print("[Pi3] Running model inference...")
    use_cuda = device.type == "cuda" and torch.cuda.is_available()
    try:
        has_sm8 = use_cuda and torch.cuda.get_device_capability()[0] >= 8
    except Exception:
        has_sm8 = False
    dtype = torch.bfloat16 if has_sm8 else torch.float16

    with torch.no_grad():
        if use_cuda:
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                res = model(imgs[None])  # Add batch dimension
        else:
            res = model(imgs[None])

    # Also extract intermediate embeddings
    try:
        embed_tokens, embed_pooled = extract_pi3_embeddings(model, imgs, device, use_cuda, dtype)
    except Exception as e:
        print(f"[Pi3][WARN] Failed to extract embeddings for {asset_name}: {e}")
        embed_tokens, embed_pooled = None, None

    # Build masks like example.py
    masks = torch.sigmoid(res["conf"][..., 0]) > 0.1
    non_edge = ~depth_edge(res["local_points"][..., 2], rtol=0.03)
    masks = torch.logical_and(masks, non_edge)[0]

    # Prepare output dirs and filenames
    asset_out_dir = osp.join(out_session_dir, asset_name)
    ensure_dir(asset_out_dir)

    ply_path = osp.join(asset_out_dir, f"{asset_name}.ply")
    print(f"[Pi3] Saving point cloud: {ply_path}")
    write_ply(res["points"][0][masks].cpu(), imgs.permute(0, 2, 3, 1)[masks], ply_path)

    print("[Pi3] Saving camera geometry (.npy and .json)...")
    save_camera_geometry(res["camera_poses"][0], asset_out_dir, asset_name)

    # Save embeddings if available
    embeds_info = {}
    if embed_tokens is not None and embed_pooled is not None:
        tokens_path = osp.join(asset_out_dir, f"{asset_name}_embed_tokens.npy")
        pooled_path = osp.join(asset_out_dir, f"{asset_name}_embed.npy")
        np.save(tokens_path, embed_tokens.detach().cpu().numpy().astype(np.float16))
        np.save(pooled_path, embed_pooled.detach().cpu().numpy().astype(np.float16))
        embeds_info = {
            "embed_tokens_path": tokens_path,
            "embed_path": pooled_path,
            "embed_tokens_shape": list(embed_tokens.shape),
            "embed_dim": int(embed_pooled.shape[-1]),
        }

    # Also save a simple metadata file
    meta = {
        "asset_path": asset_path,
        "asset_name": asset_name,
        "interval": interval,
        "num_frames": int(imgs.shape[0]),
        "ply_path": ply_path,
        **embeds_info,
    }
    with open(osp.join(asset_out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[Pi3] Done: {asset_name}")


def main():
    parser = argparse.ArgumentParser(description="Pi3 session reconstruction entrypoint")
    parser.add_argument(
        "--input_root",
        type=str,
        default="/home/cevin/ViPicks/ViPicks_Models/input",
        help="Root directory containing session folders (e.g., session1, session2)",
    )
    parser.add_argument(
        "--outputs_root",
        type=str,
        default="/home/cevin/ViPicks/ViPicks_Models/outputs",
        help="Root directory to write outputs into",
    )
    parser.add_argument(
        "--session",
        type=str,
        default="all",
        help="Session name to process (e.g., session1). Use 'all' to process every session under input_root.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=-1,
        help="Frame sampling interval. If <0, uses 10 for video and 1 for images by default.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default='3DLearn/Pi3/model.safetensors',
        help="Optional local checkpoint path (.safetensors or .pth). If not provided, will use from_pretrained.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda or cpu)",
    )

    args = parser.parse_args()

    # Prepare model
    print("[Pi3] Loading model...")
    device = torch.device(args.device)
    if args.ckpt is not None:
        model = Pi3().to(device).eval()
        if args.ckpt.endswith(".safetensors"):
            from safetensors.torch import load_file
            weight = load_file(args.ckpt)
        else:
            weight = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(weight)
    else:
        model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()

    # Determine which sessions to process
    if args.session == "all":
        try:
            sessions = [d for d in os.listdir(args.input_root) if osp.isdir(osp.join(args.input_root, d))]
        except FileNotFoundError:
            print(f"[Pi3][ERROR] input_root not found: {args.input_root}")
            return
    else:
        sessions = [args.session]

    if not sessions:
        print("[Pi3][WARN] No sessions to process.")
        return

    # Process sessions
    for sess in sorted(sessions):
        session_dir = osp.join(args.input_root, sess)
        if not osp.isdir(session_dir):
            print(f"[Pi3][WARN] Session directory not found: {session_dir}. Skipping.")
            continue

        print(f"[Pi3] Processing session: {sess}")
        out_session_dir = osp.join(args.outputs_root, sess)
        ensure_dir(out_session_dir)

        assets = collect_assets(session_dir)
        if not assets:
            print(f"[Pi3][WARN] No assets found in {session_dir}. Skipping.")
            continue

        for asset_path, asset_name in assets:
            try:
                run_inference_on_asset(model, device, asset_path, asset_name, args.interval, out_session_dir)
            except Exception as e:
                print(f"[Pi3][ERROR] Failed to process asset '{asset_name}' in session '{sess}': {e}")

    print("[Pi3] All done.")


if __name__ == "__main__":
    main()
