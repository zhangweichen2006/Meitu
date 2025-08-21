import os
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
import cv2
import datetime
from tqdm import tqdm
from pathlib import Path
from human_models.human_models import SMPLX
from ultralytics import YOLO
from main.base import Tester
from main.config import Config
from utils.data_utils import load_img, process_bbox, generate_patch_image
from utils.visualization_utils import render_mesh
from utils.inference_utils import non_max_suppression


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, dest='num_gpus')
    parser.add_argument('--file_name', type=str, default='test')
    parser.add_argument('--ckpt_name', type=str, default='model_dump')
    parser.add_argument('--start', type=str, default=1)
    parser.add_argument('--end', type=str, default=1)
    parser.add_argument('--multi_person', action='store_true')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Absolute path to a single input image. If set, run single-image inference and ignore --start/--end')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Absolute path to a directory of input images. If set, process all images in the folder and ignore --start/--end')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Optional absolute path to save outputs for --image_dir/--image_path modes. Defaults to demo/output_frames/<name>')
    parser.add_argument('--human_model_path', type=str, default=None,
                        help='Absolute path to human models directory containing smplx assets (e.g., .../human_models/models)')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cudnn.benchmark = True

    # init config
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    root_dir = Path(__file__).resolve().parent.parent
    # Resolve config and checkpoint paths relative to repo root, with fallback
    config_path = osp.join(root_dir, 'pretrained_models', args.ckpt_name, 'config_base.py')
    if not osp.exists(config_path):
        config_path = osp.join(root_dir, 'config_base.py')
    cfg = Config.load_config(config_path)
    checkpoint_path = osp.join(root_dir, 'pretrained_models', args.ckpt_name, f'{args.ckpt_name}.pth.tar')
    img_folder = osp.join(root_dir, 'demo', 'input_frames', args.file_name)
    output_folder = osp.join(root_dir, 'demo', 'output_frames', args.file_name)
    os.makedirs(output_folder, exist_ok=True)
    exp_name = f'inference_{args.file_name}_{args.ckpt_name}_{time_str}'

    # resolve human model path (CLI > ENV > default in repo)
    env_human_path = os.environ.get('HUMAN_MODEL_PATH')
    human_model_path = args.human_model_path or env_human_path or osp.join(root_dir, 'human_models', 'models')

    # validate required assets exist to provide actionable error early
    new_config = {
        "model": {
            "pretrained_model_path": checkpoint_path,
            "human_model_path": human_model_path,
        },
        "log":{
            'exp_name':  exp_name,
            'log_dir': osp.join(root_dir, 'outputs', exp_name, 'log'),
            }
    }
    cfg.update_config(new_config)
    cfg.prepare_log()

    # init human models
    smpl_x = SMPLX(cfg.model.human_model_path)

    # init tester
    demoer = Tester(cfg)
    demoer.logger.info(f"Using 1 GPU.")
    demoer.logger.info(f'Inference [{args.file_name}] with [{cfg.model.pretrained_model_path}].')
    demoer._make_model()

    # init detector
    bbox_model = getattr(cfg.inference.detection, "model_path",
                        './pretrained_models/yolov8x.pt')
    if not osp.isabs(bbox_model):
        bbox_model = osp.join(root_dir, bbox_model)
    detector = YOLO(bbox_model)

    # Single-image inference path
    if args.image_path is not None and args.image_dir is None:
        # If user did not override file_name, derive it from image basename
        if args.file_name == 'test':
            args.file_name = Path(args.image_path).stem
        output_folder = args.output_dir or osp.join(root_dir, 'demo', 'output_frames', args.file_name)
        os.makedirs(output_folder, exist_ok=True)

        # prepare input image
        img_path = args.image_path

        transform = transforms.ToTensor()
        original_img = load_img(img_path)
        vis_img = original_img.copy()
        original_img_height, original_img_width = original_img.shape[:2]

        # detection, concatenate xyxy with confidence to shape (N,5)
        yolo_result = detector.predict(original_img,
                                device='cuda',
                                classes=0,
                                conf=cfg.inference.detection.conf,
                                save=cfg.inference.detection.save,
                                verbose=cfg.inference.detection.verbose
                                    )[0]
        xyxy = yolo_result.boxes.xyxy.detach().cpu().numpy()
        conf = yolo_result.boxes.conf.detach().cpu().numpy().reshape(-1, 1)
        yolo_bbox = np.concatenate([xyxy, conf], axis=1) if xyxy.size != 0 else xyxy

        # If no bbox found, write the original image and exit
        if yolo_bbox.size == 0:
            frame_name = os.path.basename(img_path)
            cv2.imwrite(os.path.join(output_folder, frame_name), original_img[:, :, ::-1])
            return

        # Single person: select largest bbox; Multi person: apply NMS and keep all
        if not args.multi_person:
            areas = (yolo_bbox[:, 2] - yolo_bbox[:, 0]) * (yolo_bbox[:, 3] - yolo_bbox[:, 1])
            largest_idx = int(np.argmax(areas))
            yolo_bbox = yolo_bbox[largest_idx:largest_idx+1]
        else:
            yolo_bbox = non_max_suppression(yolo_bbox, cfg.inference.detection.iou_thr)

        num_bbox = len(yolo_bbox)

        # loop all detected bboxes
        # color palette for different persons (BGR)
        color_palette = [
            (179, 222, 245),  # beige (BGR)
            (0, 255, 0),      # green
            (255, 0, 0),      # blue
            (0, 255, 255),    # yellow
            (255, 0, 255),    # magenta
            (255, 255, 0),    # cyan
            (128, 0, 255),
            (0, 128, 255),
            (255, 128, 0),
        ]

        for bbox_id in range(num_bbox):
            yolo_bbox_xywh = np.zeros((4))
            yolo_bbox_xywh[0] = yolo_bbox[bbox_id][0]
            yolo_bbox_xywh[1] = yolo_bbox[bbox_id][1]
            yolo_bbox_xywh[2] = abs(yolo_bbox[bbox_id][2] - yolo_bbox[bbox_id][0])
            yolo_bbox_xywh[3] = abs(yolo_bbox[bbox_id][3] - yolo_bbox[bbox_id][1])

            # xywh
            bbox = process_bbox(bbox=yolo_bbox_xywh,
                                img_width=original_img_width,
                                img_height=original_img_height,
                                input_img_shape=cfg.model.input_img_shape,
                                ratio=getattr(cfg.data, 'bbox_ratio', 1.25))
            img, _, _ = generate_patch_image(cvimg=original_img,
                                                bbox=bbox,
                                                scale=1.0,
                                                rot=0.0,
                                                do_flip=False,
                                                out_shape=cfg.model.input_img_shape)

            img = transform(img.astype(np.float32))/255
            img = img.cuda()[None,:,:,:]
            inputs = {'img': img}
            targets = {}
            meta_info = {}

            # mesh recovery
            with torch.no_grad():
                out = demoer.model(inputs, targets, meta_info, 'test')

            mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

            # render mesh
            focal = [cfg.model.focal[0] / cfg.model.input_body_shape[1] * bbox[2],
                     cfg.model.focal[1] / cfg.model.input_body_shape[0] * bbox[3]]
            princpt = [cfg.model.princpt[0] / cfg.model.input_body_shape[1] * bbox[2] + bbox[0],
                       cfg.model.princpt[1] / cfg.model.input_body_shape[0] * bbox[3] + bbox[1]]

            # draw the bbox on img
            vis_img = cv2.rectangle(vis_img, (int(yolo_bbox[bbox_id][0]), int(yolo_bbox[bbox_id][1])),
                                    (int(yolo_bbox[bbox_id][2]), int(yolo_bbox[bbox_id][3])), (0, 255, 0), 1)
            # draw mesh with per-person color
            color = color_palette[bbox_id % len(color_palette)]
            vis_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt}, mesh_as_vertices=False, color=color)

        # save rendered image
        frame_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_folder, frame_name), vis_img[:, :, ::-1])
        return

    # Directory inference path
    if args.image_dir is not None:
        image_dir = args.image_dir
        if not osp.isabs(image_dir):
            image_dir = osp.join(root_dir, image_dir)
        img_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff', '*.webp']:
            img_paths.extend(sorted(Path(image_dir).glob(ext)))
        if len(img_paths) == 0:
            raise FileNotFoundError(f"No images found in {image_dir}")

        # derive default output folder
        name = Path(image_dir).name
        output_folder = args.output_dir or osp.join(root_dir, 'demo', 'output_frames', name)
        os.makedirs(output_folder, exist_ok=True)

        # color palette for different persons (BGR)
        color_palette = [
            (179, 222, 245), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),
            (128, 0, 255), (0, 128, 255), (255, 128, 0)
        ]

        transform = transforms.ToTensor()
        for img_path in tqdm(img_paths):
            try:
                original_img = load_img(str(img_path))
                vis_img = original_img.copy()
                original_img_height, original_img_width = original_img.shape[:2]

                yolo_result = detector.predict(original_img,
                                        device='cuda',
                                        classes=0,
                                        conf=cfg.inference.detection.conf,
                                        save=cfg.inference.detection.save,
                                        verbose=cfg.inference.detection.verbose
                                            )[0]
                xyxy = yolo_result.boxes.xyxy.detach().cpu().numpy()
                conf = yolo_result.boxes.conf.detach().cpu().numpy().reshape(-1, 1)
                yolo_bbox = np.concatenate([xyxy, conf], axis=1) if xyxy.size != 0 else xyxy

                if yolo_bbox.size == 0:
                    cv2.imwrite(os.path.join(output_folder, img_path.name), original_img[:, :, ::-1])
                    continue

                if not args.multi_person:
                    areas = (yolo_bbox[:, 2] - yolo_bbox[:, 0]) * (yolo_bbox[:, 3] - yolo_bbox[:, 1])
                    largest_idx = int(np.argmax(areas))
                    yolo_bbox = yolo_bbox[largest_idx:largest_idx+1]
                else:
                    yolo_bbox = non_max_suppression(yolo_bbox, cfg.inference.detection.iou_thr)

                num_bbox = len(yolo_bbox)
                for bbox_id in range(num_bbox):
                    yolo_bbox_xywh = np.zeros((4))
                    yolo_bbox_xywh[0] = yolo_bbox[bbox_id][0]
                    yolo_bbox_xywh[1] = yolo_bbox[bbox_id][1]
                    yolo_bbox_xywh[2] = abs(yolo_bbox[bbox_id][2] - yolo_bbox[bbox_id][0])
                    yolo_bbox_xywh[3] = abs(yolo_bbox[bbox_id][3] - yolo_bbox[bbox_id][1])

                    bbox = process_bbox(bbox=yolo_bbox_xywh,
                                        img_width=original_img_width,
                                        img_height=original_img_height,
                                        input_img_shape=cfg.model.input_img_shape,
                                        ratio=getattr(cfg.data, 'bbox_ratio', 1.25))
                    img, _, _ = generate_patch_image(cvimg=original_img,
                                                        bbox=bbox,
                                                        scale=1.0,
                                                        rot=0.0,
                                                        do_flip=False,
                                                        out_shape=cfg.model.input_img_shape)

                    img = transform(img.astype(np.float32))/255
                    img = img.cuda()[None,:,:,:]
                    inputs = {'img': img}
                    targets = {}
                    meta_info = {}

                    with torch.no_grad():
                        out = demoer.model(inputs, targets, meta_info, 'test')

                    mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

                    focal = [cfg.model.focal[0] / cfg.model.input_body_shape[1] * bbox[2],
                             cfg.model.focal[1] / cfg.model.input_body_shape[0] * bbox[3]]
                    princpt = [cfg.model.princpt[0] / cfg.model.input_body_shape[1] * bbox[2] + bbox[0],
                               cfg.model.princpt[1] / cfg.model.input_body_shape[0] * bbox[3] + bbox[1]]

                    vis_img = cv2.rectangle(vis_img, (int(yolo_bbox[bbox_id][0]), int(yolo_bbox[bbox_id][1])),
                                            (int(yolo_bbox[bbox_id][2]), int(yolo_bbox[bbox_id][3])), (0, 255, 0), 1)
                    color = color_palette[bbox_id % len(color_palette)]
                    vis_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt}, mesh_as_vertices=False, color=color)

                cv2.imwrite(os.path.join(output_folder, img_path.name), vis_img[:, :, ::-1])
            except Exception as e:
                msg = f"Skip {img_path} due to error: {e}"
                try:
                    demoer.logger.warning(msg)
                except Exception:
                    print(msg)
                continue
        return

    start = int(args.start)
    end = int(args.end) + 1

    for frame in tqdm(range(start, end)):

        # prepare input image
        img_path =osp.join(img_folder, f'{int(frame):06d}.jpg')

        transform = transforms.ToTensor()
        original_img = load_img(img_path)
        vis_img = original_img.copy()
        original_img_height, original_img_width = original_img.shape[:2]

        # detection, concatenate xyxy with confidence to shape (N,5)
        yolo_result = detector.predict(original_img,
                                device='cuda',
                                classes=0,
                                conf=cfg.inference.detection.conf,
                                save=cfg.inference.detection.save,
                                verbose=cfg.inference.detection.verbose
                                    )[0]
        xyxy = yolo_result.boxes.xyxy.detach().cpu().numpy()
        conf = yolo_result.boxes.conf.detach().cpu().numpy().reshape(-1, 1)
        yolo_bbox = np.concatenate([xyxy, conf], axis=1) if xyxy.size != 0 else xyxy

        # If no bbox found, write the original image to keep frame indexing consistent and skip processing
        if yolo_bbox.size == 0:
            frame_name = os.path.basename(img_path)
            cv2.imwrite(os.path.join(output_folder, frame_name), original_img[:, :, ::-1])
            continue

        # Single person: select largest bbox; Multi person: apply NMS and keep all
        if not args.multi_person:
            areas = (yolo_bbox[:, 2] - yolo_bbox[:, 0]) * (yolo_bbox[:, 3] - yolo_bbox[:, 1])
            largest_idx = int(np.argmax(areas))
            yolo_bbox = yolo_bbox[largest_idx:largest_idx+1]
        else:
            yolo_bbox = non_max_suppression(yolo_bbox, cfg.inference.detection.iou_thr)

        num_bbox = len(yolo_bbox)

        # loop all detected bboxes
        # color palette for different persons (BGR)
        color_palette = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 0),
            (128, 0, 255),
            (0, 128, 255),
            (255, 128, 0),
        ]

        for bbox_id in range(num_bbox):
            yolo_bbox_xywh = np.zeros((4))
            yolo_bbox_xywh[0] = yolo_bbox[bbox_id][0]
            yolo_bbox_xywh[1] = yolo_bbox[bbox_id][1]
            yolo_bbox_xywh[2] = abs(yolo_bbox[bbox_id][2] - yolo_bbox[bbox_id][0])
            yolo_bbox_xywh[3] = abs(yolo_bbox[bbox_id][3] - yolo_bbox[bbox_id][1])

            # xywh
            bbox = process_bbox(bbox=yolo_bbox_xywh,
                                img_width=original_img_width,
                                img_height=original_img_height,
                                input_img_shape=cfg.model.input_img_shape,
                                ratio=getattr(cfg.data, "bbox_ratio", 1.25))
            img, _, _ = generate_patch_image(cvimg=original_img,
                                                bbox=bbox,
                                                scale=1.0,
                                                rot=0.0,
                                                do_flip=False,
                                                out_shape=cfg.model.input_img_shape)

            img = transform(img.astype(np.float32))/255
            img = img.cuda()[None,:,:,:]
            inputs = {'img': img}
            targets = {}
            meta_info = {}

            # mesh recovery
            with torch.no_grad():
                out = demoer.model(inputs, targets, meta_info, 'test')

            mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

            # render mesh
            focal = [cfg.model.focal[0] / cfg.model.input_body_shape[1] * bbox[2],
                     cfg.model.focal[1] / cfg.model.input_body_shape[0] * bbox[3]]
            princpt = [cfg.model.princpt[0] / cfg.model.input_body_shape[1] * bbox[2] + bbox[0],
                       cfg.model.princpt[1] / cfg.model.input_body_shape[0] * bbox[3] + bbox[1]]

            # draw the bbox on img
            vis_img = cv2.rectangle(vis_img, (int(yolo_bbox[bbox_id][0]), int(yolo_bbox[bbox_id][1])),
                                    (int(yolo_bbox[bbox_id][2]), int(yolo_bbox[bbox_id][3])), (0, 255, 0), 1)
            # draw mesh with per-person color
            color = color_palette[bbox_id % len(color_palette)]
            vis_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt}, mesh_as_vertices=False, color=color)

        # save rendered image
        frame_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_folder, frame_name), vis_img[:, :, ::-1])


if __name__ == "__main__":
    main()
