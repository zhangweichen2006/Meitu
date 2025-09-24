import numpy as np
import torch
import torch.nn.functional as F

def compute_normals_open3d(vertices, faces):
    import open3d as o3d
    pcd = o3d.geometry.TriangleMesh()
    pcd.vertices = o3d.utility.Vector3dVector(vertices)
    pcd.triangles = o3d.utility.Vector3iVector(faces)
    pcd.compute_vertex_normals()
    return np.array(pcd.vertex_normals)

def compute_normals(vertices, faces, eps=1e-12):
    """
    Compute per-vertex normals (area-weighted) to match Open3D.

    Args:
        vertices: ndarray of shape (V, 3) or (B, V, 3)
        faces: ndarray of shape (F, 3) (shared topology)
        eps: small epsilon for normalization stability

    Returns:
        vertex_normals: (V, 3) or (B, V, 3)
        face_normals: (F, 3) or (B, F, 3) matching batchness of vertices
    """
    faces = np.asarray(faces, dtype=np.int64)

    if vertices.ndim == 2:
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        face_normals = np.cross(v1 - v0, v2 - v0)

        vertex_normals = np.zeros_like(vertices)
        for i in range(3):
            np.add.at(vertex_normals, faces[:, i], face_normals)
        vertex_normals /= np.linalg.norm(vertex_normals, axis=1, keepdims=True) + eps
        return vertex_normals, face_normals

    elif vertices.ndim == 3:
        B, V, _ = vertices.shape
        v0 = vertices[:, faces[:, 0], :]
        v1 = vertices[:, faces[:, 1], :]
        v2 = vertices[:, faces[:, 2], :]

        face_normals = np.cross(v1 - v0, v2 - v0)

        vertex_normals = np.zeros((B, V, 3), dtype=vertices.dtype)
        for i in range(3):
            np.add.at(vertex_normals, (slice(None), faces[:, i]), face_normals)
        norms = np.linalg.norm(vertex_normals, axis=-1, keepdims=True) + eps
        vertex_normals = vertex_normals / norms
        return vertex_normals, face_normals

    else:
        raise ValueError("vertices must have shape (V,3) or (B,V,3)")

def compute_normals_torch(vertices: torch.Tensor, faces: torch.Tensor, eps=1e-12) -> torch.Tensor:
    """
    Compute per-vertex normals for an SMPL mesh using PyTorch.

    Args:
        vertices: Tensor of shape (V, 3) or (B, V, 3). Can be on GPU.
        faces: Long tensor of shape (F, 3) with vertex indices. Should be on the same device.

    Returns:
        Tensor of shape (V, 3) or (B, V, 3) with L2-normalized vertex normals.
    """
    if vertices.dim() not in (2, 3):
        raise ValueError("vertices must have shape (V,3) or (B,V,3)")

    squeeze_back = False
    if vertices.dim() == 2:
        vertices = vertices.unsqueeze(0)  # (1, V, 3)
        squeeze_back = True

    B, V, _ = vertices.shape
    device = vertices.device
    faces = faces.to(device=device, dtype=torch.long)

    # Gather triangle corners (batched)
    v0 = vertices[:, faces[:, 0], :]  # (B, F, 3)
    v1 = vertices[:, faces[:, 1], :]
    v2 = vertices[:, faces[:, 2], :]

    # Face normals (area-weighted)
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (B, F, 3)

    # Accumulate to vertices
    vn = torch.zeros((B, V, 3), device=device, dtype=vertices.dtype)
    vn.index_add_(1, faces[:, 0], fn)
    vn.index_add_(1, faces[:, 1], fn)
    vn.index_add_(1, faces[:, 2], fn)

    # Normalize per-vertex
    vn = F.normalize(vn, dim=-1, eps=eps)

    return vn.squeeze(0) if squeeze_back else vn