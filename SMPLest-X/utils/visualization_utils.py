import os
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
# Configure headless OpenGL if available; degrade gracefully otherwise
if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
try:
    import pyrender
    import trimesh
    PYRENDER_AVAILABLE = True
except Exception:
    PYRENDER_AVAILABLE = False

def vis_keypoints_with_skeleton(img, kps, kps_lines, kp_thresh=0.4, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_keypoints(img, kps, alpha=1, radius=3, color=None):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    if color is None:
        colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
        colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for i in range(len(kps)):
        p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
        if color is None:
            cv2.circle(kp_mask, p, radius=radius, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(kp_mask, p, radius=radius, color=color, thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_mesh(img, mesh_vertex, alpha=0.5):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(mesh_vertex))]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    mask = np.copy(img)

    # Draw the mesh
    for i in range(len(mesh_vertex)):
        p = mesh_vertex[i][0].astype(np.int32), mesh_vertex[i][1].astype(np.int32)
        cv2.circle(mask, p, radius=1, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, mask, alpha, 0)

def vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d[i1,0], kpt_3d[i2,0]])
        y = np.array([kpt_3d[i1,1], kpt_3d[i2,1]])
        z = np.array([kpt_3d[i1,2], kpt_3d[i2,2]])

        if kpt_3d_vis[i1,0] > 0 and kpt_3d_vis[i2,0] > 0:
            ax.plot(x, z, -y, c=colors[l], linewidth=2)
        if kpt_3d_vis[i1,0] > 0:
            ax.scatter(kpt_3d[i1,0], kpt_3d[i1,2], -kpt_3d[i1,1], c=colors[l], marker='o')
        if kpt_3d_vis[i2,0] > 0:
            ax.scatter(kpt_3d[i2,0], kpt_3d[i2,2], -kpt_3d[i2,1], c=colors[l], marker='o')

    x_r = np.array([0, cfg.input_shape[1]], dtype=np.float32)
    y_r = np.array([0, cfg.input_shape[0]], dtype=np.float32)
    z_r = np.array([0, 1], dtype=np.float32)

    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.legend()

    plt.show()
    cv2.waitKey(0)

def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '\n')
    obj_file.close()

def perspective_projection(vertices, cam_param):
    # vertices: [N, 3]
    # cam_param: [3]
    fx, fy= cam_param['focal']
    cx, cy = cam_param['princpt']
    vertices[:, 0] = vertices[:, 0] * fx / vertices[:, 2] + cx
    vertices[:, 1] = vertices[:, 1] * fy / vertices[:, 2] + cy
    return vertices

def render_mesh(img, vertices, faces, cam_param, mesh_as_vertices=False, color=None):
    if mesh_as_vertices or not PYRENDER_AVAILABLE:
        # to run on cluster where headless pyrender is not supported for A100/V100
        vertices_2d = perspective_projection(vertices, cam_param)
        draw_color = color if color is not None else (0, 0, 255)
        img = vis_keypoints(img, vertices_2d, alpha=0.8, radius=2, color=draw_color)
    else:
        focal, princpt = cam_param['focal'], cam_param['princpt']
        camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
        # the inverse is same
        pyrender2opencv = np.array([[1.0, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]])


        # render material
        if color is not None:
            # convert BGR [0-255] to RGBA [0-1]
            b, g, r = color
            base_color = (r/255.0, g/255.0, b/255.0, 1.0)
        else:
            base_color = (0.7, 0.7, 0.7, 1.0)
        material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.1,
                roughnessFactor=0.4,
                alphaMode='OPAQUE',
                emissiveFactor=(0.1, 0.1, 0.1),
                baseColorFactor=base_color)

        # get body mesh
        body_trimesh = trimesh.Trimesh(vertices, faces, process=False)
        body_mesh = pyrender.Mesh.from_trimesh(body_trimesh, material=material)

        # prepare camera and light
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        cam_pose = pyrender2opencv @ np.eye(4)

        # build scene
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                        ambient_light=(0.3, 0.3, 0.3))
        scene.add(camera, pose=cam_pose)
        scene.add(light, pose=cam_pose)
        scene.add(body_mesh, 'mesh')

        # render scene
        r = pyrender.OffscreenRenderer(viewport_width=img.shape[1],
                                        viewport_height=img.shape[0],
                                        point_size=1.0)

        color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        alpha = 0.8 # set transparency in [0.0, 1.0]

        valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
        valid_mask = valid_mask * alpha
        img = img / 255
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        output_img = (color[:, :, :] * valid_mask + (1 - valid_mask) * img)

        img = (output_img * 255).astype(np.uint8)
    return img