"""
Mesh data structure and OBJ file I/O.
"""
import numpy as np
from pathlib import Path


class Mesh:
    """Triangle mesh with vertices, normals, UVs, and face connectivity."""

    def __init__(self):
        self.vertices = np.zeros((0, 3), dtype=np.float64)
        self.normals = np.zeros((0, 3), dtype=np.float64)
        self.uvs = np.zeros((0, 2), dtype=np.float64)
        self.faces = []             # list of list[int] — vertex indices (0-based)
        self.face_normals_idx = []  # list of list[int]
        self.face_uvs_idx = []      # list of list[int]
        self.face_normals_computed = np.zeros((0, 3), dtype=np.float64)
        self.name = ""

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    def compute_face_normals(self):
        normals = []
        for face in self.faces:
            if len(face) >= 3:
                v0 = self.vertices[face[0]]
                v1 = self.vertices[face[1]]
                v2 = self.vertices[face[2]]
                n = np.cross(v1 - v0, v2 - v0)
                length = np.linalg.norm(n)
                if length > 1e-10:
                    n /= length
                else:
                    n = np.array([0.0, 1.0, 0.0])
                normals.append(n)
            else:
                normals.append(np.array([0.0, 1.0, 0.0]))
        self.face_normals_computed = np.array(normals) if normals else np.zeros((0, 3))

    def compute_vertex_normals(self):
        if len(self.face_normals_computed) == 0:
            self.compute_face_normals()
        vn = np.zeros_like(self.vertices)
        for i, face in enumerate(self.faces):
            for vi in face:
                vn[vi] += self.face_normals_computed[i]
        lengths = np.linalg.norm(vn, axis=1, keepdims=True)
        lengths = np.maximum(lengths, 1e-10)
        self.normals = vn / lengths

    def triangulate(self):
        """Fan-triangulate all faces (modifies in place)."""
        new_faces, new_fuvs, new_fnorms = [], [], []
        has_uvs = len(self.face_uvs_idx) == len(self.faces)
        has_norms = len(self.face_normals_idx) == len(self.faces)
        for i, face in enumerate(self.faces):
            for j in range(1, len(face) - 1):
                new_faces.append([face[0], face[j], face[j + 1]])
                if has_uvs:
                    fuv = self.face_uvs_idx[i]
                    new_fuvs.append([fuv[0], fuv[j], fuv[j + 1]])
                if has_norms:
                    fn = self.face_normals_idx[i]
                    new_fnorms.append([fn[0], fn[j], fn[j + 1]])
        self.faces = new_faces
        self.face_uvs_idx = new_fuvs if has_uvs else []
        self.face_normals_idx = new_fnorms if has_norms else []
        self.compute_face_normals()

    # ------------------------------------------------------------------
    # Adjacency
    # ------------------------------------------------------------------

    def get_edge_face_map(self):
        """Return dict mapping (min_v, max_v) -> list of face indices."""
        edge_to_face = {}
        for fi, face in enumerate(self.faces):
            n = len(face)
            for i in range(n):
                e = tuple(sorted((face[i], face[(i + 1) % n])))
                edge_to_face.setdefault(e, []).append(fi)
        return edge_to_face

    def get_adjacency(self):
        """Return list-of-lists face adjacency."""
        edge_to_face = self.get_edge_face_map()
        adj = [[] for _ in range(len(self.faces))]
        for flist in edge_to_face.values():
            for i in range(len(flist)):
                for j in range(i + 1, len(flist)):
                    adj[flist[i]].append(flist[j])
                    adj[flist[j]].append(flist[i])
        return adj

    # ------------------------------------------------------------------
    # Bounding info
    # ------------------------------------------------------------------

    def get_bounding_box(self):
        if len(self.vertices) == 0:
            return np.zeros(3), np.zeros(3)
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def get_center(self):
        if len(self.vertices) == 0:
            return np.zeros(3)
        return self.vertices.mean(axis=0)

    def get_scale(self):
        mn, mx = self.get_bounding_box()
        return max(np.linalg.norm(mx - mn), 1e-10)


# ======================================================================
# OBJ I/O
# ======================================================================

def load_obj(filepath):
    """Load a Wavefront .obj file and return a Mesh."""
    mesh = Mesh()
    mesh.name = Path(filepath).stem
    verts, norms, uvs = [], [], []
    faces, fuvs, fnorms = [], [], []

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            key = parts[0]
            if key == "v" and len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif key == "vn" and len(parts) >= 4:
                norms.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif key == "vt" and len(parts) >= 3:
                uvs.append([float(parts[1]), float(parts[2])])
            elif key == "f":
                fv, ft, fn = [], [], []
                for token in parts[1:]:
                    idx = token.split("/")
                    fv.append(int(idx[0]) - 1)
                    if len(idx) > 1 and idx[1]:
                        ft.append(int(idx[1]) - 1)
                    if len(idx) > 2 and idx[2]:
                        fn.append(int(idx[2]) - 1)
                faces.append(fv)
                if ft:
                    fuvs.append(ft)
                if fn:
                    fnorms.append(fn)

    mesh.vertices = np.array(verts, dtype=np.float64) if verts else np.zeros((0, 3))
    mesh.normals = np.array(norms, dtype=np.float64) if norms else np.zeros((0, 3))
    mesh.uvs = np.array(uvs, dtype=np.float64) if uvs else np.zeros((0, 2))
    mesh.faces = faces
    mesh.face_uvs_idx = fuvs if len(fuvs) == len(faces) else []
    mesh.face_normals_idx = fnorms if len(fnorms) == len(faces) else []

    # Triangulate if needed
    needs_tri = any(len(f) > 3 for f in mesh.faces)
    if needs_tri:
        mesh.triangulate()

    mesh.compute_face_normals()

    return mesh


def save_obj(filepath, mesh):
    """Save a Mesh to a Wavefront .obj file.

    Preserves original vertex normals and per-face normal indices
    exactly as loaded.  Only adds/updates UV data.
    """
    has_uvs = len(mesh.face_uvs_idx) == len(mesh.faces)
    has_normals = (len(mesh.normals) > 0
                   and len(mesh.face_normals_idx) == len(mesh.faces))

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# UV Unwrap Tool Export\n")
        for v in mesh.vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if has_normals:
            for vn in mesh.normals:
                f.write(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}\n")
        if has_uvs:
            for vt in mesh.uvs:
                f.write(f"vt {vt[0]:.6f} {vt[1]:.6f}\n")
        for i, face in enumerate(mesh.faces):
            parts = []
            for j, vi in enumerate(face):
                s = str(vi + 1)
                uv_part = ""
                vn_part = ""
                if has_uvs:
                    uv_part = str(mesh.face_uvs_idx[i][j] + 1)
                if has_normals:
                    vn_part = str(mesh.face_normals_idx[i][j] + 1)
                if vn_part:
                    s += f"/{uv_part}/{vn_part}"
                elif uv_part:
                    s += f"/{uv_part}"
                parts.append(s)
            f.write("f " + " ".join(parts) + "\n")
