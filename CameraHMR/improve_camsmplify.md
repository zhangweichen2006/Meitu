## CameraHMR + CamSMPLify: Overall Workflow (Design)

### Core components
- `Detectron2` person detector: finds human boxes per image/frame.
- `FLNet` camera intrinsics estimator: predicts FoV → focal length for the full image.
- `CameraHMR` network: regresses SMPL parameters and weak-perspective camera from crops + intrinsics.
- `SMPL` layer (via `smplx.SMPLLayer`): generates mesh vertices and joints from SMPL parameters.
- `Renderer` (PyRD): overlays mesh on the original image/frame.
- `DenseKP` model: predicts dense 2D correspondences needed by CamSMPLify.
- `CamSMPLify` (`CamSMPLify/cam_smplify.py`): optimization-based refinement of pose/shape/camera.

## End-to-end workflow (demo)

### Entry: `demo.py`
- CLI options:
  - `--image_folder` or `--video`
  - `--output_folder`, `--output_cam` (optional JSON export)
  - `--opacity`, `--same_mesh_color`, `--save_smpl_obj`
  - `--use_smplify` (enable CamSMPLify refinement)
- Instantiates `HumanMeshEstimator` (`mesh_estimator.py`) with flags.
- Routes to `run_on_images(...)` or `run_on_video(...)`.

### Per-image flow (in `HumanMeshEstimator.process_image`)
1) Input image I is loaded; basic I/O robustness for formats not handled by OpenCV.
2) Detect persons with Detectron2 → boxes, scores; keep class=person and score>thr.
3) Estimate camera intrinsics from the full image using `FLNet` → focal length f; build `cam_int`.
4) Build `Dataset` with: full image, `box_center`, `box_size` (scale), `img_size`, `cam_int`, `imgname`.
5) DataLoader (B up to 32 persons) → batch to device.
6) CameraHMR forward pass → `(out_smpl_params, out_cam, focal_length_)` where:
   - `out_smpl_params`: `global_orient (B,1,3,3)`, `body_pose (B,23,3,3)`, `betas (B,10)`
   - `out_cam`: weak-perspective camera (s, tx, ty)
   - `focal_length_`: per-sample scalar from `FLNet` (matches `cam_int`)
7) Convert weak-perspective camera to full-image translation `cam_t` via `convert_to_full_img_cam(...)`.
8) SMPL forward pass (`get_output_mesh`) → `vertices`, `joints3d`, `cam_t`.
9) Optional refinement (CamSMPLify) — see section below.
10) Render overlay with `Renderer` using `(vertices + cam_t)` and save to `output_folder`.
11) Optional exports: `.obj` mesh, and JSON camera/SMPL params for downstream (IDOL).

### Per-video flow (in `HumanMeshEstimator.run_on_video` / `process_frame`)
- Same per-frame pipeline as above, writing frames and optional JSON per frame.
- Optionally composes an overlay video from rendered frames.

## CamSMPLify refinement (online, in `mesh_estimator.py`)

### When enabled
- CLI flag `--use_smplify` causes `HumanMeshEstimator` to lazily initialize:
  - `CamSMPLify.Smplify` with thresholds from `CamSMPLify/constants.py`.
  - `DenseKP` checkpoint for dense 2D correspondences.

### Inputs to CamSMPLify (per person in batch)
- Initial SMPL from network:
  - Pose as axis-angle (24×3) converted from predicted rotmats (`smpl_rotmat_to_axis_angle`) → `global_orient`, `body_pose`.
  - Shape `betas` (10).
- Camera translation `cam_t` (full-image coordinates).
- Bounding box center `center` and scale `scale` (note: scale normalized by 200 as in datasets).
- Camera intrinsics `cam_int` (from `FLNet`-estimated f and principal point at image center).
- Image name/path for visualization/diagnostics.
- Dense 2D keypoints `dense_kp` from `DenseKP` on the same batch crop.
- (Optionally) sparse 2D keypoints if provided.

### Optimization objective (conceptual)
- Minimize robust 2D reprojection error between projected SMPL surface/joints and target 2D correspondences:
  - Dense correspondence loss using `dense_kp`.
  - Optional sparse 2D joint loss if available.
- Regularizers/prior terms:
  - Pose prior/angle limits to discourage unrealistic articulations.
  - Shape prior to keep `betas` within a reasonable range.
  - Camera translation consistency and magnitude regularization.
- Gating/thresholds:
  - `LOSS_CUT`, `LOW_THRESHOLD`, `HIGH_THRESHOLD` guide acceptance, early stopping, or step sizing.

### Outputs and reintegration
1) CamSMPLify returns refined parameters as axis-angle: `global_orient`, `pose` (23×3), `betas`, and `camera_translation`.
2) Convert refined axis-angle to rotmats via `aa_to_rotmat` and rebuild a param dict for the runtime SMPL layer.
3) Run SMPL to get refined vertices; replace the current `output_vertices[bi]` and `output_cam_trans[bi]`.
4) If optimization fails or is rejected by thresholds, keep the original network prediction.

### Why this improves results
- The network’s initial pose/shape/camera gives a fast, plausible solution but can suffer from:
  - Global orientation drift and limb twists.
  - Depth–scale ambiguity due to weak-perspective camera.
  - Misalignments under uncommon poses or atypical intrinsics.
- CamSMPLify uses pixel-level correspondences (dense 2D) with known intrinsics to refine parameters so that the projected mesh tightly aligns with the image evidence, typically yielding:
  - Lower 2D reprojection error (tighter overlays).
  - More accurate body orientation and articulations.
  - Improved camera translation consistent with the estimated focal length.

## CamSMPLify offline workflow (`CamSMPLify/optimize.py`)

### Inputs
- `.npz` file with arrays (one per detection):
  - `pose` (24×3 axis-angle) — network initialization
  - `shape` (10) — betas
  - `cam_int` (3×3)
  - `cam_t` (3)
  - `center` (2) and `scale` (scalar, normalized as used in training)
  - `dense_kp` (K×3) dense correspondences
  - `gt_keypoints` (optional sparse 2D)
  - `imgname`
- `--image_dir` pointing to the images referenced by `imgname`.

### Procedure
1) Load `.npz`; iterate samples.
2) For each record, call `SMPLify(args, pose, betas, cam_t, center, scale, cam_int, img_path, keypoints_2d, dense_kp, ind)`.
3) Collect refined `global_orient`, `pose`, `betas`, `camera_translation` and any diagnostics.
4) Save an output `.npz` (e.g., `output.npz`) with refined parameters for later use or comparison.

### Relationship to online flow
- Uses the same optimizer/objective as the online refinement but runs decoupled from inference, enabling batch processing, ablations, and metric-based comparisons.

## Data contracts and conventions
- `Dataset` provides per-person crops plus metadata: `img`, `img_size`, `box_center`, `box_size` (scale), `cam_int`, `imgname`.
- `CameraHMR` outputs rotmats for `global_orient` and `body_pose`; convert to axis-angle before SMPLify init.
- Weak-perspective camera `(s, tx, ty)` is converted to `cam_t` using image size, box, and `cam_int`.
- `scale` passed to SMPLify is normalized by 200; be consistent across online/offline flows.
- `DenseKP` predictions are run on the same normalized input used by the network batch.

## Integration notes (design)
- Initialization:
  - Lazily import CamSMPLify/DenseKP only when `--use_smplify` to keep default path lightweight.
  - Provide clear error messages and fallback to non-refined rendering on initialization failure.
- Per-batch refinement:
  - Run CamSMPLify per person; catch/continue on per-person failures.
  - Use refined params only if the optimizer returns a valid result.
- Device management:
  - Keep all tensors on the same device (GPU preferred); convert only at caller boundaries for JSON exports.
- I/O and diagnostics:
  - Optional visualization hooks inside CamSMPLify controlled by flags (e.g., `vis`, `save_vis_dir`).
- Extensibility:
  - A split `mesh_estimator_camsmplify.py` can import from `mesh_estimator.py` and override only refinement-related hooks, sharing the rest of the pipeline.

## Example invocations (design)
- Image folder, baseline only:
  - `python demo.py --image_folder <imgs> --output_folder <out>`
- Image folder with CamSMPLify refinement:
  - `python demo.py --image_folder <imgs> --output_folder <out> --use_smplify`
- Video with CamSMPLify + JSON export:
  - `python demo.py --video <in.mp4> --output_folder <out_root> --output_cam <json_out> --use_smplify`
- Offline CamSMPLify:
  - `python CamSMPLify/optimize.py --input <init.npz> --image_dir <imgs> --output_dir <out> [--vis]`

## High-level flow diagram (design)
```mermaid
graph TD
  A[Input Image/Frame] --> B[Detectron2 Person Detection]
  B --> C[Dataset: crops + metadata]
  A --> D[FLNet: Intrinsics (f)]
  C --> E[CameraHMR]
  D --> E
  E --> F[SMPL params + weak-persp cam]
  F --> G[Convert to cam_t]
  G --> H[SMPL Layer → vertices]
  H --> I{--use_smplify?}
  I -- No --> J[Render + Export]
  I -- Yes --> K[DenseKP]
  K --> L[CamSMPLify Optimize]
  L --> M[Refined pose/betas/cam_t]
  M --> N[SMPL Layer → refined vertices]
  N --> J

  subgraph Offline
    O[init .npz + images] --> P[CamSMPLify/optimize.py]
    P --> Q[Refined .npz]
  end
```

## Expected improvements from CamSMPLify
- Tighter overlay alignment via minimized dense 2D reprojection error.
- Better global orientation and limb articulation, fewer twisted limbs.
- More consistent depth/translation with the estimated focal length.
- Robust fallback to baseline when optimization is not confident.

## Acceptance (design)
- Baseline (no `--use_smplify`) produces overlays and optional JSON/OBJ identical to current behavior.
- With `--use_smplify`, the system runs DenseKP + CamSMPLify per person and updates vertices/camera only when valid results are returned; otherwise falls back.
- Offline `optimize.py` can refine batches from precomputed `.npz` with the same objective.


