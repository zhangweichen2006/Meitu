import cv2
import torch
import joblib
import scipy.misc
from skimage.transform import rotate, resize
import numpy as np
try:
    import jpeg4py as jpeg
    _HAS_JPEG4PY = True
except Exception:
    jpeg = None
    _HAS_JPEG4PY = False
from loguru import logger
from trimesh.visual import color

def read_img(img_fn):
    img = cv2.cvtColor(cv2.imread(img_fn), cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)


def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1



def read_img(img_fn):

    fn_lower = img_fn.lower()
    is_jpeg = fn_lower.endswith('jpeg') or fn_lower.endswith('jpg')

    # Try jpeg4py first for JPEGs if available
    if is_jpeg and _HAS_JPEG4PY:
        try:
            # jpeg4py accepts a filename path directly
            img = np.array(jpeg.JPEG(img_fn).decode())
            return img.astype(np.float32)
        except Exception as e:
            # Fallback to OpenCV/ImageIO if jpeg4py fails for any reason (e.g., libjpeg-turbo missing)
            logger.warning(f"jpeg4py decode failed for {img_fn}: {e}. Falling back to OpenCV/ImageIO.")

    # OpenCV fallback
    img_cv = cv2.imread(img_fn)
    if img_cv is not None:
        img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32)

    # ImageIO ultimate fallback (handles uncommon formats or CV2 read failures)
    try:
        from imageio import v2 as imageio
        img = imageio.imread(img_fn)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        return img.astype(np.float32)
    except Exception as e:
        raise OSError(f"Failed to read image {img_fn} using jpeg4py, OpenCV, or imageio: {e}")


def crop(img, center, scale, res, rot=0):
    """Crop image according to the supplied bounding box."""
    # Upper left point

    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(transform([res[0] + 1,
                             res[1] + 1], center, scale, res, invert=1)) - 1

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]

    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1],
                                                    old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding

        new_img = rotate(new_img, rot) # scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    # resize image
    new_img = resize(new_img, res) # scipy.misc.imresize(new_img, res)
    return new_img
