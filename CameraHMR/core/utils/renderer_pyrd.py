
import os
import trimesh
import pyrender
import numpy as np
import colorsys
import cv2

class Renderer(object):

    def __init__(self, focal_length=600, img_w=512, img_h=512, faces=None,
                 same_mesh_color=False, mesh_opacity=0.5):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_w,
                                                   viewport_height=img_h,
                                                   point_size=1.0)
        self.camera_center = [img_w // 2, img_h // 2]
        self.focal_length = focal_length
        self.faces = faces
        self.same_mesh_color = same_mesh_color
        # 0 -> fully transparent overlay; 1 -> fully opaque mesh
        self.mesh_opacity = float(mesh_opacity)
        try:
            print(f"[DEBUG][Renderer.__init__] viewport=({img_w},{img_h}), focal_length={self.focal_length}, "
                  f"camera_center={self.camera_center}, same_mesh_color={self.same_mesh_color}, "
                  f"mesh_opacity={self.mesh_opacity}, faces_shape={None if faces is None else np.array(faces).shape}")
        except Exception as e:
            print(f"[DEBUG][Renderer.__init__] debug print failed: {e}")


    def render_front_view(self, verts, bg_img_rgb=None, bg_color=(1., 1., 1., 1.)):
        # Create a scene for each image and render all meshes
        try:
            verts_arr = np.asarray(verts)
            print(f"[DEBUG][Renderer.render_front_view] verts.shape={verts_arr.shape}, dtype={getattr(verts_arr, 'dtype', type(verts_arr))}; "
                  f"bg_img_rgb.shape={None if bg_img_rgb is None else bg_img_rgb.shape}")
            print(f"[DEBUG][Renderer.render_front_view] focal_length={self.focal_length}, camera_center={self.camera_center}, mesh_opacity={self.mesh_opacity}")
            if verts_arr.size > 0:
                v0 = verts_arr[0]
                print(f"[DEBUG][Renderer.render_front_view] verts[0] stats: min={v0.min():.6f}, max={v0.max():.6f}, mean={v0.mean():.6f}")
        except Exception as e:
            print(f"[DEBUG][Renderer.render_front_view] pre-render inspect failed: {e}")
        scene = pyrender.Scene(bg_color=bg_color, ambient_light=np.ones(3) * 0)
        # Create camera. Camera will always be at [0,0,0]
        camera = pyrender.camera.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                                  cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=np.eye(4))

        # Create light source
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        # for DirectionalLight, only rotation matters
        light_pose = trimesh.transformations.rotation_matrix(np.radians(-45), [1, 0, 0])
        scene.add(light, pose=light_pose)
        light_pose = trimesh.transformations.rotation_matrix(np.radians(45), [0, 1, 0])
        scene.add(light, pose=light_pose)

        # Need to flip x-axis
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        # multiple person
        num_people = len(verts)
        # for every person in the scene
        for n in range(num_people):

            mesh = trimesh.Trimesh(verts[n], self.faces)
            mesh.apply_transform(rot)
            if self.same_mesh_color:
                # Use a neutral gray for all meshes
                mesh_color = (1.0, 0.6, 1.0)
            else:
                # Default to neutral gray when there's only one person
                if num_people == 1:
                    mesh_color = (1.0, 0.6, 1.0)
                else:
                    # Distinct hues for multiple people
                    mesh_color = colorsys.hsv_to_rgb((float(n) / num_people), 0.6, 1.0)
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.2,
                alphaMode='OPAQUE',
                baseColorFactor=(mesh_color[0], mesh_color[1], mesh_color[2], 1.0))
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material, wireframe=False)
            scene.add(mesh, 'mesh')

        # Alpha channel was not working previously, need to check again
        # Until this is fixed use hack with depth image to get the opacity
        color_rgba, depth_map = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        try:
            dmin = float(depth_map.min()) if depth_map.size else 0.0
            dmax = float(depth_map.max()) if depth_map.size else 0.0
            vcount = int((depth_map > 0).sum())
            print(f"[DEBUG][Renderer.render_front_view] depth_map stats: min={dmin:.6f}, max={dmax:.6f}, valid_pixels={vcount}/{depth_map.size}")
        except Exception as e:
            print(f"[DEBUG][Renderer.render_front_view] depth stats error: {e}")
        color_rgb = color_rgba[:, :, :3]
        if bg_img_rgb is None:
            return color_rgb
        else:
            valid_mask = (depth_map > 0)[:,:,None]
            # bg_img_rgb[mask] = color_rgb[mask]
            # return bg_img_rgb
            visible_weight = self.mesh_opacity
            if bg_img_rgb.shape[:2] != color_rgb.shape[:2]:
                print(f"[DEBUG][Renderer.render_front_view] WARNING: bg_img_rgb shape {bg_img_rgb.shape} != render shape {color_rgb.shape}")
            output_img = (
                color_rgb[:, :, :3] * valid_mask * visible_weight
                + bg_img_rgb * (1-valid_mask) +
                (valid_mask) * bg_img_rgb * (1-visible_weight)
            )
            return output_img

    def render_side_view(self, verts):
        centroid = verts.mean(axis=(0, 1))  # n*6890*3 -> 3
        # make the centroid at the image center (the X and Y coordinates are zeros)
        centroid[:2] = 0
        aroundy = cv2.Rodrigues(np.array([0, np.radians(270.), 0]))[0][np.newaxis, ...]  # 1*3*3
        pred_vert_arr_side = np.matmul((verts - centroid), aroundy) + centroid
        #To get more field of view
        pred_vert_arr_side[:,:,2]+=1
        side_view = self.render_front_view(pred_vert_arr_side)
        return side_view

    def delete(self):
        """
        Need to delete before creating the renderer next time
        """
        self.renderer.delete()
