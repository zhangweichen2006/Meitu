#!/usr/bin/env python3
import os
import json
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

# Optional: use WAI IO helpers to store scene_meta with extras
try:
    from mapanything.utils.wai.core import store_data as wai_store_data
    USE_WAI_STORE = True
except Exception:
    USE_WAI_STORE = False

def safe_symlink_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)

def to_frame_name(p: Path) -> str:
    return p.stem

def inv4x4(M):
    try:
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return np.eye(4, dtype=np.float32)

def convert_scene(images_dir: Path, labels_npz: Path, output_root: Path,
                  dataset_name: str = "closeup-suburbd-bbox44",
                  scene_name: str | None = None):
    assert images_dir.is_dir(), f"images_dir not found: {images_dir}"
    assert labels_npz.is_file(), f"labels_npz not found: {labels_npz}"
    output_root.mkdir(parents=True, exist_ok=True)

    # Scene name
    if scene_name is None:
        # default: parent of images_dir (e.g., 20221011_1_250_batch01hand_closeup_suburb_d_6fps)
        scene_name = images_dir.parent.name

    # WAI layout: <WAI_ROOT>/<dataset_name>/<scene_name>/
    scene_out = output_root / dataset_name / scene_name
    images_out = scene_out / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    data = np.load(str(labels_npz), allow_pickle=True)
    imgnames = data["imgname"]
    K_all = data["cam_int"] if "cam_int" in data.files else None
    ext_all = data["cam_ext"] if "cam_ext" in data.files else None

    frames = []
    for i, rel in enumerate(imgnames):
        # Source
        rel = str(rel)
        src = (images_dir / rel).resolve()
        if not src.is_file():
            print(f"Skip missing: {src}")
            continue

        # Read image to get size
        img = cv2.imread(str(src), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            print(f"Skip unreadable: {src}")
            continue
        h, w = img.shape[:2]

        # Derive intrinsics
        if K_all is not None:
            K = np.array(K_all[i]).astype(np.float32)
            fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
        else:
            # fallback
            fx = fy = float(max(w, h))
            cx, cy = w/2.0, h/2.0

        # Derive cam2world (OpenCV convention) for WAI
        if ext_all is not None:
            # ext likely world2camera; invert to get cam2world
            w2c = np.array(ext_all[i]).reshape(4,4).astype(np.float32)
            c2w = inv4x4(w2c)
        else:
            c2w = np.eye(4, dtype=np.float32)

        # Place image in WAI images/ (symlink or copy)
        src_ext = src.suffix.lower()  # keep original extension
        frame_name = to_frame_name(Path(rel))
        dst_rel = Path("images") / f"{frame_name}{src_ext}"
        dst_abs = scene_out / dst_rel
        safe_symlink_or_copy(src, dst_abs)

        # Frame entry
        frames.append({
            "frame_name": frame_name,
            "image": str(dst_rel).replace("\\", "/"),
            "file_path": str(dst_rel).replace("\\", "/"),
            "transform_matrix": c2w.tolist(),
            "h": int(h),
            "w": int(w),
            "fl_x": float(fx),
            "fl_y": float(fy),
            "cx": float(cx),
            "cy": float(cy),
        })

    # scene_meta
    scene_meta = {
        "scene_name": scene_name,
        "dataset_name": dataset_name,
        "version": "0.1",
        "shared_intrinsics": False,
        "camera_model": "PINHOLE",
        "camera_convention": "opencv",
        "scene_modalities": {},
        "frames": frames,
        "frame_modalities": {
            "image": {"frame_key": "image", "format": "image"}
        },
    }

    scene_meta_path = scene_out / "scene_meta.json"
    if USE_WAI_STORE:
        wai_store_data(scene_meta_path, scene_meta, "scene_meta")
    else:
        scene_meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scene_meta_path, "w") as f:
            json.dump(scene_meta, f, indent=2)
    print(f"WAI scene written: {scene_out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", type=Path,
                    help="Path to images directory (ends with /png for this dataset)")
    ap.add_argument("--labels_npz", type=Path,
                    help="Path to labels npz (training/traintest labels for this scene)")
    ap.add_argument("--wai_root", type=Path,
                    help="Output WAI root directory")
    ap.add_argument("--dataset_name", default="closeup-suburbd-bbox44", type=str)
    ap.add_argument("--scene_name", default=None, type=str)    args = ap.parse_args()

#         python3 tools/convert_closeup_bedlam_to_wai.py \
#   --images_dir /home/cevin/Meitu/CameraHMR/data/training-images/20221011_1_250_batch01hand_closeup_suburb_d_6fps/png \
#   --labels_npz /home/cevin/Meitu/CameraHMR/data/traintest-labels/bedlam-labels/20221011_1_250_batch01hand_closeup_suburb_d_6fps.npz \
#   --wai_root /home/cevin/Meitu/WAI_DATA \
#   --dataset_name closeup-suburbd-bbox44

    if args.images_dir is None:
        args.images_dir = Path("/home/cevin/Meitu/CameraHMR/data/training-images/20221011_1_250_batch01hand_closeup_suburb_d_6fps/png")
    if args.labels_npz is None:
        args.labels_npz = Path("/home/cevin/Meitu/CameraHMR/data/traintest-labels/bedlam-labels/20221011_1_250_batch01hand_closeup_suburb_d_6fps.npz")
    if args.wai_root is None:
        args.wai_root = Path("/home/cevin/Meitu/CameraHMR/data/wai_datasets")
    if args.dataset_name is None:
        args.dataset_name = "closeup-suburbd-bbox44"


    convert_scene(args.images_dir, args.labels_npz, args.wai_root,
                  dataset_name=args.dataset_name, scene_name=args.scene_name)