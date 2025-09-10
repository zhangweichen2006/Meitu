import os
import argparse
import numpy as np
import torch
from cam_smplify import SMPLify
from core.backbones import create_backbone
from core.constants import DETECTRON_CKPT, VITPOSE_BACKBONE, IMAGE_MEAN, IMAGE_STD
from detectron2.config import LazyConfig
from core.utils.utils_detectron2 import DefaultPredictor_Lazy


def convert_coco17_to_body25(coco_kps):
    """Convert COCO-17 keypoints (17x3) to OpenPose BODY_25 (25x3).
    Missing foot keypoints are zeroed.
    Neck = midpoint of shoulders; MidHip = midpoint of hips.
    """
    body25 = np.zeros((25, 3), dtype=np.float32)
    if coco_kps is None or len(coco_kps) == 0:
        return body25
    c = coco_kps
    # direct mappings
    # Nose
    body25[0] = c[0]
    # R/L Shoulders, Elbows, Wrists
    body25[2] = c[6]  # RShoulder
    body25[3] = c[8]  # RElbow
    body25[4] = c[10] # RWrist
    body25[5] = c[5]  # LShoulder
    body25[6] = c[7]  # LElbow
    body25[7] = c[9]  # LWrist
    # Hips, Knees, Ankles
    body25[9]  = c[12]  # RHip
    body25[10] = c[14]  # RKnee
    body25[11] = c[16]  # RAnkle
    body25[12] = c[11]  # LHip
    body25[13] = c[13]  # LKnee
    body25[14] = c[15]  # LAnkle
    # Eyes / Ears
    body25[15] = c[2]  # REye
    body25[16] = c[1]  # LEye
    body25[17] = c[4]  # REar
    body25[18] = c[3]  # LEar
    # Neck = midpoint of shoulders
    ls, rs = c[5], c[6]
    body25[1, :2] = (ls[:2] + rs[:2]) / 2.0
    body25[1, 2] = (ls[2] + rs[2]) / 2.0
    # MidHip = midpoint of hips
    lh, rh = c[11], c[12]
    body25[8, :2] = (lh[:2] + rh[:2]) / 2.0
    body25[8, 2] = (lh[2] + rh[2]) / 2.0
    # Toes/Heels remain zeros
    return body25


def _infer_keypoints_mmpose(img_path, det_cfg, det_ckpt, pose_cfg, pose_ckpt, device='cuda'):
    """Run MMPose (COCO-17) with an MMDetection person detector.
    Returns list of dicts: { 'keypoints': (17,3) np.array, 'bbox': (4,) } for each person.
    """
    try:
        from mmdet.apis import init_detector as mmdet_init_detector, inference_detector as mmdet_inference_detector
        from mmpose.apis import init_model as mmpose_init_model, inference_topdown
    except Exception as e:
        print(f"MMPose/MMDet not available: {e}")
        return []
    det_model = mmdet_init_detector(det_cfg, det_ckpt, device=device)
    pose_model = mmpose_init_model(pose_cfg, pose_ckpt, device=device)
    # Detect persons
    det_results = mmdet_inference_detector(det_model, img_path)
    # filter person class (usually label 0)
    bboxes = []
    if isinstance(det_results, tuple):
        det_results = det_results[0]
    for label, cls_bboxes in enumerate(det_results):
        if label != 0:
            continue
        if cls_bboxes is None:
            continue
        for b in cls_bboxes:
            x1, y1, x2, y2, score = b.tolist()
            if score < 0.5:
                continue
            bboxes.append([x1, y1, x2, y2, score])
    if len(bboxes) == 0:
        return []
    # Pose inference
    results = inference_topdown(pose_model, img_path, bboxes)
    persons = []
    for r in results:
        kps = r.get('keypoints', None)
        if kps is None:
            # mmpose 1.x uses 'pred_instances'
            inst = r.get('pred_instances', None)
            if inst is not None and hasattr(inst, 'keypoints'):
                kps = inst.keypoints
                if hasattr(inst, 'keypoint_scores'):
                    scores = inst.keypoint_scores
                else:
                    scores = np.ones((kps.shape[0], kps.shape[1]), dtype=np.float32)
                kps = np.concatenate([kps, scores[..., None]], axis=-1)
        if isinstance(kps, torch.Tensor):
            kps = kps.detach().cpu().numpy()
        # r may include bbox
        bbox = r.get('bbox', None)
        if bbox is None and 'bboxes' in r:
            bbox = r['bboxes']
        if isinstance(bbox, torch.Tensor):
            bbox = bbox.detach().cpu().numpy()
        if kps is None or kps.shape[1] < 17:
            continue
        # ensure (17,3)
        kps17 = kps[0, :17, :]
        persons.append({'keypoints': kps17.astype(np.float32), 'bbox': bbox[0] if isinstance(bbox, np.ndarray) else bbox})
    return persons


class SimplePoseDecoder(torch.nn.Module):
    """A minimal heatmap head for ViTPose backbone features.
    Converts feature map (B, C, H', W') to COCO-17 heatmaps (B, 17, 64, 48).
    """
    def __init__(self, in_channels=1280, num_joints=17, out_h=64, out_w=48):
        super().__init__()
        self.proj = torch.nn.Conv2d(in_channels, num_joints, kernel_size=1, bias=True)
        self.out_h = out_h
        self.out_w = out_w

    def forward(self, x):
        # x: (B, C, H', W')
        h = self.proj(x)
        h = torch.nn.functional.interpolate(h, size=(self.out_h, self.out_w), mode='bilinear', align_corners=False)
        return h


def _keypoints_from_heatmaps(hm):
    """Argmax decode heatmaps to (N,17,3) with [x,y,conf] in image pixels (256x192)."""
    B, J, H, W = hm.shape
    hm_reshaped = hm.view(B, J, -1)
    conf, idx = torch.max(hm_reshaped, dim=-1)  # (B,J)
    ys = (idx // W).float()
    xs = (idx % W).float()
    # map from heatmap (W=48,H=64) to image (W=192,H=256)
    xs_img = xs * (192.0 / W)
    ys_img = ys * (256.0 / H)
    kps = torch.stack([xs_img, ys_img, conf], dim=-1)  # (B,J,3)
    return kps


def _infer_keypoints_vitpose(img_path, predictor, backbone, decoder, device='cuda'):
    import cv2
    img = cv2.imread(img_path)
    if img is None:
        return []
    out = predictor(img)
    inst = out['instances']
    valid = (inst.pred_classes == 0) & (inst.scores > 0.5)
    boxes = inst.pred_boxes.tensor[valid].cpu().numpy()
    persons = []
    if boxes.shape[0] == 0:
        return persons
    # Normalize util
    mean = torch.tensor(IMAGE_MEAN, dtype=torch.float32).view(1,3,1,1)
    std = torch.tensor(IMAGE_STD, dtype=torch.float32).view(1,3,1,1)
    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in b[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1]-1, x2), min(img.shape[0]-1, y2)
        crop = img[y1:y2, x1:x2, :]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (192, 256), interpolation=cv2.INTER_LINEAR)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(crop_rgb.transpose(2,0,1)).float()/255.0
        t = (t - mean.squeeze())/std.squeeze()
        t = t.unsqueeze(0).to(device)
        backbone = backbone.to(device)
        decoder = decoder.to(device)
        with torch.no_grad():
            feats = backbone(t)            # (1,C,H',W')
            if isinstance(feats, (list, tuple)):
                feats = feats[-1]
            heatmaps = decoder(feats)      # (1,17,64,48)
            kps = _keypoints_from_heatmaps(heatmaps)[0].cpu().numpy()
        persons.append({'keypoints': kps, 'bbox': [x1,y1,x2,y2]})
    return persons

def main(args):

    init_param_file = args.input
    image_base_dir = args.image_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_file_path =  os.path.join(args.output_dir, "output.npz")

    smplify = SMPLify(vis=args.vis, verbose=args.verbose, save_vis_dir=args.save_vis_dir)
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

        # If no gt_keypoints, run MMPose to get COCO-17 and convert to BODY_25
        if keypoints_2d is None or (hasattr(keypoints_2d, 'size') and keypoints_2d.size == 0):
            # Try ViTPose backbone + simple decoder first
            backbone = create_backbone()
            sd = torch.load(VITPOSE_BACKBONE, map_location='cpu')
            sd = sd.get('state_dict', sd)
            missing, unexpected = backbone.load_state_dict(sd, strict=False)
            backbone.eval()
            pose_dec = SimplePoseDecoder(in_channels=backbone.embed_dim if hasattr(backbone, 'embed_dim') else 1280,
                                         num_joints=17)
            # If the checkpoint is a ViTPose+ pretrain (no head), we won’t have decoder weights; use random init
            # Try detectron2 person detector from our constants to get bboxes, then run backbone+decoder
            det_cfg = LazyConfig.load(str('core/utils/cascade_mask_rcnn_vitdet_h_75ep.py'))
            det_cfg.train.init_checkpoint = DETECTRON_CKPT
            predictor = DefaultPredictor_Lazy(det_cfg)
            persons = _infer_keypoints_vitpose(img_path, predictor, backbone, pose_dec, device=args.device)
            # Fallback to MMPose if no persons found or empty
            if len(persons) == 0 and args.pose_config and args.pose_checkpoint:
                persons = _infer_keypoints_mmpose(
                    img_path,
                    det_cfg=args.det_config,
                    det_ckpt=args.det_checkpoint,
                    pose_cfg=args.pose_config,
                    pose_ckpt=args.pose_checkpoint,
                    device=args.device
                )
            if len(persons) > 0:
                # pick the person whose bbox best matches our center/scale (IoU)
                cx, cy = center[0].item(), center[1].item()
                h = float(scale.item())
                w = h
                best = None
                best_iou = -1.0
                for p in persons:
                    x1, y1, x2, y2 = p['bbox'][:4]
                    # IoU with our square bbox
                    a1x1, a1y1 = cx - w/2, cy - h/2
                    a1x2, a1y2 = cx + w/2, cy + h/2
                    inter_x1, inter_y1 = max(a1x1, x1), max(a1y1, y1)
                    inter_x2, inter_y2 = min(a1x2, x2), min(a1y2, y2)
                    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
                    area_a = w * h
                    area_b = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                    union = area_a + area_b - inter + 1e-6
                    iou = inter / union
                    if iou > best_iou:
                        best_iou = iou
                        best = p
                if best is not None:
                    body25 = convert_coco17_to_body25(best['keypoints'])
                    keypoints_2d = body25
            # If still None, leave as None; SMPLify will fallback

        # Run SMPLify optimization
        result = smplify(args, pose, betas, cam_t, center, scale, cam_int, img_path, keypoints_2d, dense_kp, i)

        if result:
            processed_data["imgname"].append(img_path)
            processed_data["center"].append(coco_data["center"][i])
            processed_data["scale"].append(coco_data["scale"][i])
            processed_data["cam_int"].append(coco_data["cam_int"][i])
            if keypoints_2d is not None:
                processed_data["gt_keypoints"].append(keypoints_2d)
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
    # MMPose / MMDetection configs (optional)
    parser.add_argument("--det_config", type=str, default=os.environ.get('MMDET_DET_CFG', ''), help="Path to mmdetection detector config")
    parser.add_argument("--det_checkpoint", type=str, default=os.environ.get('MMDET_DET_CKPT', DETECTRON_CKPT), help="Path to mmdetection/detectron detector checkpoint")
    parser.add_argument("--pose_config", type=str, default=os.environ.get('MMPOSE_POSE_CFG', ''), help="Path to mmpose pose config")
    parser.add_argument("--pose_checkpoint", type=str, default=os.environ.get('MMPOSE_POSE_CKPT', VITPOSE_BACKBONE), help="Path to mmpose pose checkpoint")
    parser.add_argument("--device", type=str, default='cuda', help="Device for inference")
    parser.add_argument("--vis", type=bool, required=False, help="Visualization of fitting")
    parser.add_argument("--save_vis_dir", type=str, default=None, help="Directory to save headless visualizations")
    parser.add_argument("--verbose", type=bool, required=False, help="Print losses")
    parser.add_argument("--vis_int", type=int, default=100, required=False, help="Visualize result after every 100 iteration of optimization")
    parser.add_argument("--loss_cut", type=int, default=100, required=False, help="If initial loss is more than 100 we use high loss threshold else low loss threshold")
    parser.add_argument("--high_threshold", type=int, default=50, required=False, help="Loss threshold value to select the optimization result")
    parser.add_argument("--low_threshold", type=int, default=30, required=False, help="Loss threshold value to select the optimization result")

    args = parser.parse_args()
    main(args)

