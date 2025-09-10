from PIL import Image
import numpy as np
import torch

def vis_img(img, bgr_to_rgb=False):
    import cv2
    if isinstance(img, np.ndarray):
        if bgr_to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(img, str):
        img = cv2.imread(img)
        if bgr_to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        if bgr_to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(img, Image.Image):
        img = np.array(img)
        if bgr_to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Invalid image type: {type(img)}")

    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def vis_pc(pc):
    import open3d as o3d
    pc = pc.detach().cpu().numpy()
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([pc])

def vis_smpl(smpl_output):
    import cv2
    import open3d as o3d
    import numpy as np
    import smplx
    try:
        from core.utils.renderer_pyrd import Renderer
    except Exception:
        # If running as a loose script (e.g., importing vis.py directly), core.* may not be on sys.path
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # add repo root
        from core.utils.renderer_pyrd import Renderer

    from smplx import SMPL
    smpl_model = SMPL(model_path='data/models/SMPL/SMPL_NEUTRAL.pkl', gender='NEUTRAL')

    # smpl_output = smplx.SMPLOutput.from_dict(smpl_output)
    smpl_output.vertices = smpl_output.vertices.detach().cpu().numpy()
    smpl_output.joints = smpl_output.joints.detach().cpu().numpy()
    smpl_output.faces = smpl_model.faces
    renderer = Renderer(focal_length=5000, img_w=256, img_h=256, faces=smpl_output.faces, same_mesh_color=True)
    render_img = renderer.render_front_view(smpl_output.vertices)
    cv2.imshow('SMPL', render_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()