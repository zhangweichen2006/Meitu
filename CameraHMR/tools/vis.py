from PIL import Image
import numpy as np
try:
    import torch  # Optional; not required for server
except Exception:
    torch = None
import threading
import json
import gradio as gr
import os

_latest_image = None
_latest_object = None
_state_lock = threading.Lock()
_demo = None
_persist_dir = os.environ.get('CAMERAHMR_VIS_DIR', '/tmp/camerahmr_vis')
_persist_img = os.path.join(_persist_dir, 'latest.png')


def vis_img(img, bgr_to_rgb=False):
    """Publish an image frame to the Gradio UI.

    - img: np.ndarray (RGB or BGR), PIL.Image, torch.Tensor, or file path.
    - bgr_to_rgb: set True if the input is BGR (e.g., from cv2.imread).
    """
    import cv2
    if isinstance(img, str):
        arr = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError(f"Could not load image from path: {img}")
        bgr_to_rgb = True
    elif (torch is not None) and isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
    elif isinstance(img, Image.Image):
        arr = np.array(img)
    elif isinstance(img, np.ndarray):
        arr = img
    else:
        raise ValueError(f"Invalid image type: {type(img)}")

    # Ensure valid image data
    if arr is None or arr.size == 0:
        raise ValueError("Empty or invalid image data")

    # Handle different data types and ranges
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    # Convert color channels
    if arr.ndim == 3 and arr.shape[2] == 3 and bgr_to_rgb:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 4 and bgr_to_rgb:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)

    global _latest_image
    with _state_lock:
        _latest_image = arr.copy()  # Make a copy to avoid issues
        # Persist to disk so other processes can read
        try:
            os.makedirs(_persist_dir, exist_ok=True)
            if arr.ndim == 3 and arr.shape[2] == 3:
                cv2.imwrite(_persist_img, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            elif arr.ndim == 3 and arr.shape[2] == 4:
                cv2.imwrite(_persist_img, cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA))
            else:
                cv2.imwrite(_persist_img, arr)
        except Exception as e:
            print(f"Warning: Could not persist image to disk: {e}")
            pass


def vis_dense_kp(img, dense_kp, conf_threshold=0.3, point_size=2):
    """Visualize dense keypoints on an image.
    
    Args:
        img: np.ndarray (RGB), PIL.Image, torch.Tensor, or file path
        dense_kp: np.ndarray of shape (N, 3) where each row is (x, y, confidence)
        conf_threshold: minimum confidence to show a keypoint
        point_size: radius of keypoint circles
    
    Returns:
        np.ndarray: Image with keypoints overlaid
    """
    import cv2
    
    # Convert image to numpy array
    if isinstance(img, str):
        arr = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError(f"Could not load image from path: {img}")
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    elif (torch is not None) and isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
    elif isinstance(img, Image.Image):
        arr = np.array(img)
    elif isinstance(img, np.ndarray):
        arr = img.copy()
    else:
        raise ValueError(f"Invalid image type: {type(img)}")
    
    # Ensure uint8 format
    if arr.dtype != np.uint8:
        if arr.dtype in [np.float32, np.float64] and arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    
    if dense_kp is None or len(dense_kp) == 0:
        return arr
    
    # Convert to numpy if needed
    if hasattr(dense_kp, 'detach'):  # torch tensor
        dense_kp = dense_kp.detach().cpu().numpy()
    
    # Draw keypoints
    vis_img = arr.copy()
    for i, (x, y, conf) in enumerate(dense_kp):
        if conf >= conf_threshold:
            # Use different colors for different keypoint groups
            if i < 24:  # SMPL joints
                color = (255, 0, 0)  # Red for main joints
            elif i < 100:  # Body keypoints
                color = (0, 255, 0)  # Green for body keypoints  
            else:  # Face/hand keypoints
                color = (0, 0, 255)  # Blue for face/hand keypoints
            
            cv2.circle(vis_img, (int(x), int(y)), point_size, color, thickness=-1)
            # Add small text with keypoint index for debugging
            if point_size >= 3:
                cv2.putText(vis_img, str(i), (int(x)+3, int(y)-3), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
    
    return vis_img


def vis_img_with_dense_kp(img, dense_kp=None, conf_threshold=0.3, point_size=2, bgr_to_rgb=False):
    """Convenience function to visualize image with dense keypoints and send to Gradio.
    
    Args:
        img: Image to display
        dense_kp: Dense keypoints array (N, 3) with (x, y, confidence)
        conf_threshold: Minimum confidence to show keypoint
        point_size: Size of keypoint circles
        bgr_to_rgb: Whether to convert BGR to RGB
    """
    if dense_kp is not None:
        img_with_kp = vis_dense_kp(img, dense_kp, conf_threshold, point_size)
        vis_img(img_with_kp, bgr_to_rgb=bgr_to_rgb)
    else:
        vis_img(img, bgr_to_rgb=bgr_to_rgb)


def vis_obj(obj):
    """Publish a Python object to the Gradio UI (JSON if possible, else str)."""
    global _latest_object
    with _state_lock:
        _latest_object = obj


def _poll_image():
    with _state_lock:
        arr = _latest_image
    if arr is None and os.path.exists(_persist_img):
        try:
            import cv2
            bgr = cv2.imread(_persist_img, cv2.IMREAD_UNCHANGED)
            if bgr is not None:
                if bgr.ndim == 3 and bgr.shape[2] == 3:
                    arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                elif bgr.ndim == 3 and bgr.shape[2] == 4:
                    arr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGBA)
                else:
                    arr = bgr
        except Exception as e:
            print(f"Error loading persisted image: {e}")
            arr = None

    # Ensure proper format for Gradio display
    if arr is not None and arr.size > 0:
        try:
            # Convert to PIL Image for better Gradio compatibility
            from PIL import Image
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

            # Debug: check image properties
            print(f"Image shape: {arr.shape}, dtype: {arr.dtype}, min: {arr.min()}, max: {arr.max()}")

            # Ensure we have valid image dimensions
            if arr.ndim == 3 and arr.shape[2] in [3, 4]:
                return Image.fromarray(arr)
            elif arr.ndim == 2:  # Grayscale
                return Image.fromarray(arr, mode='L')
            else:
                print(f"Unsupported image dimensions: {arr.shape}")
                return None
        except Exception as e:
            print(f"Error converting image for display: {e}")
            return None
    return None


def _poll_image_scaled(scale: float = 1.0):
    """Return the latest image scaled by the given factor."""
    img = _poll_image()
    if img is None:
        return None
    try:
        import numpy as _np
        from PIL import Image as _PILImage
        if isinstance(img, _PILImage.Image):
            arr = _np.array(img)
        else:
            arr = img
        if not isinstance(scale, (int, float)) or abs(scale - 1.0) < 1e-6 or scale <= 0:
            return img
        new_w = max(1, int(arr.shape[1] * float(scale)))
        new_h = max(1, int(arr.shape[0] * float(scale)))
        arr_u8 = _np.clip(arr, 0, 255).astype(_np.uint8)
        import cv2 as _cv2
        interp = _cv2.INTER_AREA if scale < 1.0 else _cv2.INTER_LINEAR
        resized = _cv2.resize(arr_u8, (new_w, new_h), interpolation=interp)
        return _PILImage.fromarray(resized)
    except Exception:
        return img


def _poll_object():
    with _state_lock:
        obj = _latest_object
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return None if obj is None else str(obj)


def start_gradio(host: str = '0.0.0.0', port: int = 7860, share: bool = False, inbrowser: bool = False):
    """Start a Gradio UI in the background.

    - Open http://<server-ip>:<port>/ from your Mac.
    - Call vis_img(...) or vis_obj(...) anywhere to update.
    """
    global _demo
    if _demo is not None:
        return _demo

    with gr.Blocks(css="""
        .gradio-container { max-width: none !important; }
        .image-container { max-height: 1024px !important; max-width: 1024px !important; }
        .tab-nav { margin-bottom: 10px; }
        .block { width: 100% !important; }
    """) as demo:
        gr.Markdown("## Gradio Img/3DObj Live Viewer")
        with gr.Tab("Image"):
            with gr.Row():
                img_out = gr.Image(label="Live Image", value=_poll_image(), height=1024, elem_classes=["image-container"])
            with gr.Row():
                scale_slider = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.05, label="Image scale")
        with gr.Tab("Object"):
            obj_out = gr.JSON(label="Latest Object", value=_poll_object())

        # Periodic polling (Gradio >= v4 via Timer). Fallback to single load if Timer not available.
        try:
            t1 = gr.Timer(0.25)
            t1.tick(fn=_poll_image, inputs=[], outputs=img_out)
            t2 = gr.Timer(0.5)
            t2.tick(fn=_poll_object, inputs=[], outputs=obj_out)
            # Scale slider only updates on manual change, doesn't interfere with timer
            scale_slider.release(fn=_poll_image_scaled, inputs=[scale_slider], outputs=img_out)
        except Exception:
            demo.load(_poll_image, None, img_out)
            demo.load(_poll_object, None, obj_out)
            scale_slider.release(fn=_poll_image_scaled, inputs=[scale_slider], outputs=img_out)

    demo.queue()
    demo.launch(server_name=host, server_port=port, share=share, inbrowser=inbrowser, show_error=True, prevent_thread_lock=True, debug=False)
    _demo = demo
    return demo



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