"""
UV Pelt unwrap – ARAP (As-Rigid-As-Possible) parameterization for
rocks, cliffs, and natural environments.

References
----------
* Takeo Igarashi, Tomer Moscovich, John F. Hughes,
  "As-Rigid-As-Possible Shape Manipulation", ACM TOG 2005.
* Olga Sorkine & Marc Alexa,
  "As-Rigid-As-Possible Surface Modeling", SGP 2007.
* Lévy, Petitjean, Ray & Maillot,
  "Least Squares Conformal Maps for Automatic Texture Atlas Generation",
  ACM SIGGRAPH 2002  (used for the cotangent-Laplacian formulation).

The implementation uses vectorized numpy/scipy with pre-factored sparse
solvers for interactive performance on meshes up to ~50k faces.
"""

import math
import logging
import numpy as np
from collections import deque
import heapq

logger = logging.getLogger("uv_unwrap")

try:
    from scipy import sparse
    from scipy.sparse.linalg import lsqr as sparse_lsqr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ======================================================================
# Pelt UV Unwrap – ARAP + Discrete Harmonic + Free Boundary
# ======================================================================
#
# Production-quality single-island unwrap.
#
# Pipeline:
#   1.  Build local mesh.  If closed -> auto-seam with curvature weighting.
#   2.  Initial flattening via cotangent-Laplacian harmonic map
#       (boundary pinned to circle).
#   3.  ARAP (As-Rigid-As-Possible) iterations with **free boundary**
#       -- the gold standard for stretch minimisation.
#   4.  Normalize into [0..1] UV space.
#
# Everything is vectorized with numpy/scipy for ~50 000-face meshes.
# ======================================================================


def _orient_pca_projection(eigvecs, face_normals):
    """
    Orient PCA eigenvectors so that the projection views the mesh from
    the dominant-normal side (i.e. the side most faces point towards).

    The projection plane is spanned by eigvecs[:, 2] and eigvecs[:, 1].
    The discarded axis eigvecs[:, 0] is the projection normal.  If it
    opposes the area-weighted average face normal we negate eigvec 2
    so the UV island shows the "top" / outer side of the mesh.
    """
    if len(face_normals) == 0:
        return eigvecs
    avg_n = face_normals.mean(axis=0)
    length = np.linalg.norm(avg_n)
    if length < 1e-10:
        return eigvecs
    avg_n /= length
    proj_normal = eigvecs[:, 0]
    if np.dot(proj_normal, avg_n) < 0:
        eigvecs = eigvecs.copy()
        eigvecs[:, 2] *= -1  # mirror U axis → view from opposite side
        logger.info("PCA projection auto-oriented to dominant normal side.")
    return eigvecs


def _build_local_mesh(mesh):
    v2local = {}
    order = []
    for face in mesh.faces:
        for vi in face:
            if vi not in v2local:
                v2local[vi] = len(order)
                order.append(vi)
    faces_local = []
    for face in mesh.faces:
        faces_local.append([v2local[vi] for vi in face])
    return order, faces_local, v2local


def _compute_face_quality(mesh):
    """
    Compute a per-face quality score in [0, 1].

    Quality is based on the ratio of triangle area to the area of an
    equilateral triangle with the same longest edge.  Degenerate (sliver
    or near-zero-area) triangles score close to 0; well-shaped triangles
    score close to 1.
    """
    verts = mesh.vertices
    n_faces = len(mesh.faces)
    quality = np.ones(n_faces, dtype=np.float64)
    for fi, face in enumerate(mesh.faces):
        if len(face) < 3:
            quality[fi] = 0.0
            continue
        v0 = verts[face[0]]
        v1 = verts[face[1]]
        v2 = verts[face[2]]
        e0 = v1 - v0
        e1 = v2 - v0
        e2 = v2 - v1
        area = 0.5 * np.linalg.norm(np.cross(e0, e1))
        max_edge = max(np.linalg.norm(e0), np.linalg.norm(e1),
                       np.linalg.norm(e2))
        if max_edge < 1e-14:
            quality[fi] = 0.0
            continue
        # Ratio vs equilateral triangle with same longest edge
        ideal_area = (math.sqrt(3) / 4.0) * max_edge * max_edge
        quality[fi] = min(area / ideal_area, 1.0)
    return quality


def _find_boundary_ordered(faces_local, n_verts):
    edge_count = {}
    for fl in faces_local:
        n = len(fl)
        for i in range(n):
            e = tuple(sorted((fl[i], fl[(i + 1) % n])))
            edge_count[e] = edge_count.get(e, 0) + 1
    boundary_adj = {}
    for (a, b), cnt in edge_count.items():
        if cnt == 1:
            boundary_adj.setdefault(a, []).append(b)
            boundary_adj.setdefault(b, []).append(a)
    if not boundary_adj:
        return []
    start = next(iter(boundary_adj))
    visited = {start}
    path = [start]
    cur = start
    while True:
        nbs = boundary_adj.get(cur, [])
        moved = False
        for nb in nbs:
            if nb not in visited:
                visited.add(nb)
                path.append(nb)
                cur = nb
                moved = True
                break
        if not moved:
            break
    return path


def _dijkstra(adj_w, src, n):
    dist = np.full(n, np.inf)
    prev = np.full(n, -1, dtype=int)
    dist[src] = 0.0
    heap = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for nb, w in adj_w[u]:
            nd = d + w
            if nd < dist[nb]:
                dist[nb] = nd
                prev[nb] = u
                heapq.heappush(heap, (nd, nb))
    return dist, prev


def _auto_seam(mesh, order, faces_local, n_verts):
    """
    Cut a closed mesh open using a geodesic cycle with curvature-weighted
    edges so seams prefer to run along sharp crevices (natural features
    of rocks / cliffs) rather than across flat surfaces.
    """
    if len(mesh.face_normals_computed) == 0:
        mesh.compute_face_normals()

    edge_to_faces = {}
    for fi, fl in enumerate(faces_local):
        n = len(fl)
        for i in range(n):
            e = tuple(sorted((fl[i], fl[(i + 1) % n])))
            edge_to_faces.setdefault(e, []).append(fi)

    # Edge weight = edge_length * flatness_factor
    # flatness = (1 + cos(dihedral)) / 2    in [0, 1]
    # 0 = 180 deg fold (sharp), 1 = flat
    edge_weight = {}
    for e, flist in edge_to_faces.items():
        a, b = e
        geo_len = float(np.linalg.norm(
            mesh.vertices[order[a]] - mesh.vertices[order[b]]))
        if len(flist) == 2:
            n1 = mesh.face_normals_computed[flist[0]]
            n2 = mesh.face_normals_computed[flist[1]]
            cos_d = float(np.clip(np.dot(n1, n2), -1, 1))
            flatness = (1.0 + cos_d) * 0.5
            w = geo_len * (0.1 + 0.9 * flatness)
        else:
            w = geo_len * 0.05
        edge_weight[e] = w

    adj_w = [[] for _ in range(n_verts)]
    for e, w in edge_weight.items():
        a, b = e
        adj_w[a].append((b, w))
        adj_w[b].append((a, w))

    # Find two far-apart vertices
    dist0, _ = _dijkstra(adj_w, 0, n_verts)
    far1 = int(np.argmax(dist0))
    dist1, prev1 = _dijkstra(adj_w, far1, n_verts)
    far2 = int(np.argmax(dist1))

    # Path 1
    path1 = []
    cur = far2
    while cur != -1:
        path1.append(cur)
        cur = prev1[cur]
    if len(path1) < 2:
        return order, faces_local, n_verts

    path1_edges = set()
    for i in range(len(path1) - 1):
        path1_edges.add(tuple(sorted((path1[i], path1[i + 1]))))

    # Path 2: penalise path1 edges
    big_w = dist1[far2] * 100.0
    adj_w2 = [[] for _ in range(n_verts)]
    for e, w in edge_weight.items():
        a, b = e
        ww = big_w if e in path1_edges else w
        adj_w2[a].append((b, ww))
        adj_w2[b].append((a, ww))

    dist2, prev2 = _dijkstra(adj_w2, far1, n_verts)
    path2 = []
    cur = far2
    while cur != -1:
        path2.append(cur)
        cur = prev2[cur]
    if len(path2) < 2:
        return order, faces_local, n_verts

    path2_edges = set()
    for i in range(len(path2) - 1):
        path2_edges.add(tuple(sorted((path2[i], path2[i + 1]))))

    cycle_edges = path1_edges.symmetric_difference(path2_edges)
    if len(cycle_edges) < 3:
        cycle_edges = path1_edges | path2_edges

    # Partition faces
    face_adj = [[] for _ in range(len(faces_local))]
    for e, flist in edge_to_faces.items():
        if e in cycle_edges:
            continue
        for i in range(len(flist)):
            for j in range(i + 1, len(flist)):
                face_adj[flist[i]].append(flist[j])
                face_adj[flist[j]].append(flist[i])

    visited_f = [-1] * len(faces_local)
    comp_id = 0
    for start_f in range(len(faces_local)):
        if visited_f[start_f] >= 0:
            continue
        queue = deque([start_f])
        visited_f[start_f] = comp_id
        while queue:
            fi = queue.popleft()
            for nb in face_adj[fi]:
                if visited_f[nb] < 0:
                    visited_f[nb] = comp_id
                    queue.append(nb)
        comp_id += 1

    if comp_id < 2:
        return order, faces_local, n_verts

    cycle_verts = set()
    for (a, b) in cycle_edges:
        cycle_verts.add(a)
        cycle_verts.add(b)

    dup_map = {}
    for v in cycle_verts:
        dup_map[(v, 0)] = v
    next_idx = n_verts
    for c in range(1, comp_id):
        for v in cycle_verts:
            dup_map[(v, c)] = next_idx
            order.append(order[v])
            next_idx += 1

    for fi in range(len(faces_local)):
        c = visited_f[fi]
        fl = faces_local[fi]
        faces_local[fi] = [dup_map[(v, c)] if v in cycle_verts else v
                           for v in fl]

    return order, faces_local, next_idx


# ------------------------------------------------------------------
# Cotangent Laplacian
# ------------------------------------------------------------------

def _cotangent_weight(p0, p1, p2):
    e1 = p1 - p0
    e2 = p2 - p0
    dot = np.dot(e1, e2)
    cross_len = np.linalg.norm(np.cross(e1, e2))
    if cross_len < 1e-14:
        return 0.0
    return dot / cross_len


def _build_cotangent_laplacian(order, faces_local, n_verts, mesh_verts):
    rows, cols, vals = [], [], []
    diag = np.zeros(n_verts)
    for fl in faces_local:
        if len(fl) < 3:
            continue
        i0, i1, i2 = fl[0], fl[1], fl[2]
        p0 = mesh_verts[order[i0]]
        p1 = mesh_verts[order[i1]]
        p2 = mesh_verts[order[i2]]
        cot0 = _cotangent_weight(p0, p1, p2)
        cot1 = _cotangent_weight(p1, p0, p2)
        cot2 = _cotangent_weight(p2, p0, p1)
        for (a, b, w) in [(i1, i2, cot0), (i0, i2, cot1), (i0, i1, cot2)]:
            w2 = max(w * 0.5, 1e-8)
            rows.append(a); cols.append(b); vals.append(w2)
            rows.append(b); cols.append(a); vals.append(w2)
            diag[a] -= w2
            diag[b] -= w2
    for i in range(n_verts):
        rows.append(i); cols.append(i); vals.append(diag[i])
    return sparse.coo_matrix(
        (vals, (rows, cols)), shape=(n_verts, n_verts)).tocsc()


def _boundary_to_circle(boundary, order, mesh_verts):
    n = len(boundary)
    if n < 3:
        return np.zeros((n, 2))
    lengths = np.zeros(n)
    for i in range(1, n):
        lengths[i] = lengths[i - 1] + np.linalg.norm(
            mesh_verts[order[boundary[i]]] - mesh_verts[order[boundary[i - 1]]])
    total = lengths[-1] + np.linalg.norm(
        mesh_verts[order[boundary[0]]] - mesh_verts[order[boundary[-1]]])
    if total < 1e-12:
        total = 1.0
    angles = lengths / total * 2.0 * math.pi
    return np.column_stack([np.cos(angles), np.sin(angles)])


# ------------------------------------------------------------------
# ARAP (As-Rigid-As-Possible) -- vectorized with numpy
# ------------------------------------------------------------------

def _build_triangle_local_frames(tri_p):
    """
    For each triangle, compute a local 2D frame and project its 3D
    vertices into that frame.

    tri_p : (n_tri, 3, 3)  -- 3D positions of triangle vertices

    Returns
    -------
    local_2d : (n_tri, 3, 2)  -- 2D coords of each vertex in local frame
    valid    : (n_tri,) bool   -- True if the triangle is non-degenerate
    """
    n_tri = tri_p.shape[0]
    local_2d = np.zeros((n_tri, 3, 2))
    valid = np.ones(n_tri, dtype=bool)

    e1 = tri_p[:, 1] - tri_p[:, 0]  # (n_tri, 3)
    e2 = tri_p[:, 2] - tri_p[:, 0]

    # Tangent = e1 normalised
    e1_len = np.linalg.norm(e1, axis=1, keepdims=True)  # (n_tri, 1)
    e1_len = np.maximum(e1_len, 1e-14)
    t = e1 / e1_len  # (n_tri, 3)

    # Normal
    nn = np.cross(e1, e2)  # (n_tri, 3)
    nn_len = np.linalg.norm(nn, axis=1, keepdims=True)
    degenerate = (nn_len.ravel() < 1e-14)
    valid[degenerate] = False
    nn_len = np.maximum(nn_len, 1e-14)
    nn = nn / nn_len

    # Bitangent
    b = np.cross(nn, t)  # (n_tri, 3)

    # p0 is at origin
    # p1 in local frame
    local_2d[:, 1, 0] = np.sum(e1 * t, axis=1)
    local_2d[:, 1, 1] = np.sum(e1 * b, axis=1)
    # p2 in local frame
    local_2d[:, 2, 0] = np.sum(e2 * t, axis=1)
    local_2d[:, 2, 1] = np.sum(e2 * b, axis=1)

    return local_2d, valid


def _build_arap_data(order, faces_arr, n_verts, mesh_verts):
    """
    Pre-compute everything needed for ARAP iterations (vectorized).
    """
    n_tri = len(faces_arr)
    faces_np = np.array(faces_arr, dtype=int)  # (n_tri, 3)

    # Triangle 3D positions (n_tri, 3, 3) - vectorized gather
    order_arr = np.array(order, dtype=int)
    tri_p = mesh_verts[order_arr[faces_np]]  # (n_tri, 3, 3)

    # Vectorized cotangent half-weights
    # For vertex k in triangle, cot(angle_k) = dot(e1,e2) / |cross(e1,e2)|
    # where e1, e2 are the two edges meeting at vertex k
    cot_w = np.zeros((n_tri, 3))

    # At vertex 0: edges (p1-p0), (p2-p0)
    e01 = tri_p[:, 1] - tri_p[:, 0]
    e02 = tri_p[:, 2] - tri_p[:, 0]
    dots0 = np.sum(e01 * e02, axis=1)
    cross0 = np.linalg.norm(np.cross(e01, e02), axis=1)
    cot_w[:, 0] = np.where(cross0 > 1e-14, dots0 / cross0 * 0.5, 1e-8)

    # At vertex 1: edges (p0-p1), (p2-p1)
    e10 = tri_p[:, 0] - tri_p[:, 1]
    e12 = tri_p[:, 2] - tri_p[:, 1]
    dots1 = np.sum(e10 * e12, axis=1)
    cross1 = np.linalg.norm(np.cross(e10, e12), axis=1)
    cot_w[:, 1] = np.where(cross1 > 1e-14, dots1 / cross1 * 0.5, 1e-8)

    # At vertex 2: edges (p0-p2), (p1-p2)
    e20 = tri_p[:, 0] - tri_p[:, 2]
    e21 = tri_p[:, 1] - tri_p[:, 2]
    dots2 = np.sum(e20 * e21, axis=1)
    cross2 = np.linalg.norm(np.cross(e20, e21), axis=1)
    cot_w[:, 2] = np.where(cross2 > 1e-14, dots2 / cross2 * 0.5, 1e-8)

    cot_w = np.maximum(cot_w, 1e-8)

    # Local 2D frames for 3D triangles
    local_2d, valid = _build_triangle_local_frames(tri_p)

    # Build the cotangent Laplacian from edge arrays (vectorized)
    # 3 edges per triangle: (opp=0, a=1, b=2), (opp=1, a=0, b=2), (opp=2, a=0, b=1)
    edge_a = np.column_stack([faces_np[:, 1], faces_np[:, 0], faces_np[:, 0]]).ravel()
    edge_b = np.column_stack([faces_np[:, 2], faces_np[:, 2], faces_np[:, 1]]).ravel()
    edge_w = np.column_stack([cot_w[:, 0], cot_w[:, 1], cot_w[:, 2]]).ravel()

    # Off-diagonal entries (both directions)
    rows = np.concatenate([edge_a, edge_b])
    cols = np.concatenate([edge_b, edge_a])
    vals = np.concatenate([edge_w, edge_w])

    # Diagonal = -sum of off-diagonal per row
    diag = np.zeros(n_verts)
    np.add.at(diag, edge_a, -edge_w)
    np.add.at(diag, edge_b, -edge_w)

    rows = np.concatenate([rows, np.arange(n_verts)])
    cols = np.concatenate([cols, np.arange(n_verts)])
    vals = np.concatenate([vals, diag])

    L_cot = sparse.coo_matrix(
        (vals, (rows, cols)), shape=(n_verts, n_verts)).tocsc()

    return {
        "tri_p": tri_p,
        "cot_w": cot_w,
        "L_cot": L_cot,
        "n_tri": n_tri,
        "faces_np": faces_np,
        "local_2d": local_2d,
        "valid": valid,
    }


def _arap_local_step(uvs, arap_data):
    """
    Vectorized local step: best-fit rotation per triangle via batched SVD.
    """
    n_tri = arap_data["n_tri"]
    faces_np = arap_data["faces_np"]
    local_2d = arap_data["local_2d"]  # (n_tri, 3, 2)
    cot_w = arap_data["cot_w"]        # (n_tri, 3)
    valid = arap_data["valid"]

    rotations = np.zeros((n_tri, 2, 2))
    rotations[:, 0, 0] = 1.0
    rotations[:, 1, 1] = 1.0

    vm = valid  # mask for valid triangles

    if not np.any(vm):
        return rotations

    # Gather UV positions for all triangles: (n_tri, 3, 2)
    uv_tri = uvs[faces_np]  # advanced indexing: (n_tri, 3, 2)

    # Reference edges in local 2D: (n_tri, 3, 2)
    # Edge 0: v0->v1, Edge 1: v0->v2, Edge 2: v1->v2
    ref_e = np.zeros((n_tri, 3, 2))
    ref_e[:, 0] = local_2d[:, 1] - local_2d[:, 0]
    ref_e[:, 1] = local_2d[:, 2] - local_2d[:, 0]
    ref_e[:, 2] = local_2d[:, 2] - local_2d[:, 1]

    # UV edges: (n_tri, 3, 2)
    uv_e = np.zeros((n_tri, 3, 2))
    uv_e[:, 0] = uv_tri[:, 1] - uv_tri[:, 0]
    uv_e[:, 1] = uv_tri[:, 2] - uv_tri[:, 0]
    uv_e[:, 2] = uv_tri[:, 2] - uv_tri[:, 1]

    # Weights for edges: edge0 opposite v2 -> cot[2], etc.
    w = np.zeros((n_tri, 3))
    w[:, 0] = cot_w[:, 2]  # edge v0->v1 opposite v2
    w[:, 1] = cot_w[:, 1]  # edge v0->v2 opposite v1
    w[:, 2] = cot_w[:, 0]  # edge v1->v2 opposite v0

    # Covariance S = sum_e w_e * ref_e outer uv_e  -> (n_tri, 2, 2)
    # ref_e: (n_tri, 3, 2), uv_e: (n_tri, 3, 2), w: (n_tri, 3)
    # S[t,i,j] = sum_k w[t,k] * ref_e[t,k,i] * uv_e[t,k,j]
    S = np.einsum('tk,tki,tkj->tij', w, ref_e, uv_e)

    # Batched SVD on valid triangles only
    S_valid = S[vm]
    try:
        U, _, Vt = np.linalg.svd(S_valid)
    except np.linalg.LinAlgError:
        # Fall back to per-triangle SVD, skipping degenerate ones
        n_valid = S_valid.shape[0]
        U = np.zeros((n_valid, 2, 2))
        Vt = np.zeros((n_valid, 2, 2))
        for i in range(n_valid):
            try:
                U[i], _, Vt[i] = np.linalg.svd(S_valid[i])
            except np.linalg.LinAlgError:
                U[i] = np.eye(2)
                Vt[i] = np.eye(2)
    R = np.einsum('tji,tki->tjk', Vt, U)  # Vt.T @ U.T

    # Fix reflections: det(R) < 0
    dets = R[:, 0, 0] * R[:, 1, 1] - R[:, 0, 1] * R[:, 1, 0]
    flip = dets < 0
    if np.any(flip):
        Vt[flip, 1, :] *= -1
        R[flip] = np.einsum('tji,tki->tjk', Vt[flip], U[flip])

    rotations[vm] = R
    return rotations


def _run_arap(uvs, faces_arr, order, mesh_verts, n_verts, iterations,
              progress_cb=None):
    """
    Run ARAP iterations with free boundary.
    Pre-factors the Laplacian for fast repeated solves.
    """
    if not HAS_SCIPY or n_verts < 4 or iterations <= 0:
        return

    arap_data = _build_arap_data(order, faces_arr, n_verts, mesh_verts)

    # Soft-pin: anchor the centroid-nearest vertex
    center = uvs.mean(axis=0)
    dists = np.linalg.norm(uvs[:n_verts] - center, axis=1)
    pin_v = int(np.argmin(dists))
    pin_idx = np.array([pin_v], dtype=int)
    pin_pos = uvs[pin_idx].copy()

    # Pre-factor the pinned Laplacian (constant across iterations)
    L = arap_data["L_cot"].copy()
    diag_sum = abs(L.diagonal().sum())
    pin_weight = max(diag_sum * 0.001, 1.0)
    L_pin = L.tolil()
    for pi in pin_idx:
        L_pin[pi, pi] += pin_weight
    L_pin = L_pin.tocsc()

    from scipy.sparse.linalg import factorized
    try:
        solve = factorized(L_pin)
    except Exception:
        from scipy.sparse.linalg import spsolve
        solve = lambda b: spsolve(L_pin, b)

    # Store pin data for global step
    arap_data["pin_idx"] = pin_idx
    arap_data["pin_pos"] = pin_pos
    arap_data["pin_weight"] = pin_weight
    arap_data["solve"] = solve

    for it in range(iterations):
        rotations = _arap_local_step(uvs, arap_data)
        _arap_global_step_fast(uvs, arap_data, rotations)
        if progress_cb:
            progress_cb(it, iterations)


def _arap_global_step_fast(uvs, arap_data, rotations):
    """Global step using pre-factored solver."""
    faces_np = arap_data["faces_np"]
    local_2d = arap_data["local_2d"]
    cot_w = arap_data["cot_w"]
    valid = arap_data["valid"]
    n_verts = len(uvs)

    rhs = np.zeros((n_verts, 2))

    edge_pairs = np.array([[1, 2], [0, 2], [0, 1]], dtype=int)
    opp_idx = np.array([0, 1, 2], dtype=int)

    for k in range(3):
        ea, eb = edge_pairs[k]
        opp = opp_idx[k]
        e_ref = local_2d[:, eb] - local_2d[:, ea]
        e_rot = np.einsum('tij,tj->ti', rotations, e_ref)
        w = cot_w[:, opp]
        w_e = e_rot * w[:, None]
        w_e[~valid] = 0.0
        a_idx = faces_np[:, ea]
        b_idx = faces_np[:, eb]
        np.add.at(rhs, a_idx, w_e)
        np.add.at(rhs, b_idx, -w_e)

    # Add pin contribution
    for pi, pp in zip(arap_data["pin_idx"], arap_data["pin_pos"]):
        rhs[pi] += arap_data["pin_weight"] * pp

    solve = arap_data["solve"]
    uvs[:, 0] = solve(rhs[:, 0])
    uvs[:, 1] = solve(rhs[:, 1])


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def pelt_unwrap(mesh, arap_iterations=50, island_margin=0.01,
                flip_projection=False, quality_threshold=0.0):
    """
    Pelt UV unwrap -- single-island, ARAP parameterization.

    Parameters
    ----------
    arap_iterations : int
        Number of ARAP (As-Rigid-As-Possible) iterations.
        More = more even UVs.  50 is usually enough.
    island_margin : float
        Margin around the island in UV space.
    flip_projection : bool
        When True, project from the opposite side of the mesh.
        Useful for broken meshes where the default PCA orientation
        captures the wrong side.
    quality_threshold : float
        Faces with quality below this value (0–1) are excluded from UV
        computation and assigned collapsed UVs.  0 = keep all faces
        (default), 0.01–0.05 = remove severely degenerate slivers.
    """
    if len(mesh.faces) == 0:
        logger.warning("Mesh has no faces - nothing to unwrap.")
        return

    logger.info("Starting Pelt ARAP unwrap (%d faces, %d verts, "
                "iterations=%d, margin=%.3f).",
                len(mesh.faces), len(mesh.vertices),
                arap_iterations, island_margin)

    if len(mesh.face_normals_computed) == 0:
        mesh.compute_face_normals()

    # 0. Filter degenerate faces if quality_threshold > 0
    original_faces = mesh.faces
    original_normals = mesh.face_normals_computed
    removed_face_indices = []
    if quality_threshold > 0:
        face_quality = _compute_face_quality(mesh)
        good_mask = face_quality >= quality_threshold
        n_removed = int(np.sum(~good_mask))
        if n_removed > 0:
            logger.info("Quality filter: removing %d / %d degenerate faces "
                        "(threshold=%.4f).", n_removed, len(mesh.faces),
                        quality_threshold)
            removed_face_indices = list(np.where(~good_mask)[0])
            mesh.faces = [f for f, g in zip(original_faces, good_mask) if g]
            mesh.face_normals_computed = original_normals[good_mask]
        else:
            logger.info("Quality filter: all faces above threshold.")

    if len(mesh.faces) == 0:
        logger.warning("All faces were filtered out - restoring original.")
        mesh.faces = original_faces
        mesh.face_normals_computed = original_normals
        removed_face_indices = []

    # 1. Build local mesh
    order, faces_local, v2local = _build_local_mesh(mesh)
    n_verts = len(order)
    logger.info("Local mesh: %d vertices.", n_verts)

    # 2. Find or create boundary
    boundary = _find_boundary_ordered(faces_local, n_verts)
    logger.info("Boundary vertices: %d.", len(boundary))

    if len(boundary) < 3:
        logger.info("Boundary too small - running auto-seam.")
        order, faces_local, n_verts = _auto_seam(
            mesh, order, faces_local, n_verts)
        boundary = _find_boundary_ordered(faces_local, n_verts)
        logger.info("After auto-seam: %d boundary verts, %d total verts.",
                    len(boundary), n_verts)

    if len(boundary) < 3:
        # Fallback: PCA planar projection
        logger.warning("Still no valid boundary - falling back to PCA projection.")
        verts = mesh.vertices
        center = verts.mean(axis=0)
        centered = verts - center
        cov = centered.T @ centered
        _, eigvecs = np.linalg.eigh(cov)
        eigvecs = _orient_pca_projection(eigvecs, mesh.face_normals_computed)
        if flip_projection:
            eigvecs = eigvecs.copy()
            eigvecs[:, 2] *= -1
            logger.info("Flip-projection enabled - viewing from opposite side.")
        uvs_flat = centered @ eigvecs[:, [2, 1]]
        for k in range(2):
            mn, mx = uvs_flat[:, k].min(), uvs_flat[:, k].max()
            rng = mx - mn if (mx - mn) > 1e-10 else 1.0
            uvs_flat[:, k] = (uvs_flat[:, k] - mn) / rng * (1 - 2 * island_margin) + island_margin
        mesh.uvs = uvs_flat
        mesh.face_uvs_idx = [list(f) for f in mesh.faces]
        return

    boundary_set = set(boundary)
    interior = [v for v in range(n_verts) if v not in boundary_set]

    # 3. Initial harmonic map (boundary -> circle)
    circle_pos = _boundary_to_circle(boundary, order, mesh.vertices)
    uvs = np.zeros((n_verts, 2))
    for i, bi in enumerate(boundary):
        uvs[bi] = circle_pos[i]

    if HAS_SCIPY and len(interior) > 0:
        L = _build_cotangent_laplacian(order, faces_local, n_verts,
                                       mesh.vertices)
        int_idx = np.array(interior, dtype=int)
        bnd_idx = np.array(boundary, dtype=int)
        L_ii = L[np.ix_(int_idx, int_idx)]
        L_ib = L[np.ix_(int_idx, bnd_idx)]
        bnd_uv = uvs[bnd_idx]
        rhs = -L_ib @ bnd_uv

        from scipy.sparse.linalg import spsolve
        try:
            sol_u = spsolve(L_ii.tocsc(), rhs[:, 0])
            sol_v = spsolve(L_ii.tocsc(), rhs[:, 1])
            for k, vi in enumerate(int_idx):
                uvs[vi, 0] = sol_u[k]
                uvs[vi, 1] = sol_v[k]
        except Exception:
            _harmonic_jacobi_fast(uvs, faces_local, order, mesh.vertices,
                                  boundary_set, n_verts, 200)
    else:
        _harmonic_jacobi_fast(uvs, faces_local, order, mesh.vertices,
                              boundary_set, n_verts, 200)

    # 4. ARAP iterations (free boundary)
    logger.info("Running %d ARAP iterations…", arap_iterations)
    faces_arr = [list(fl) for fl in faces_local]
    _run_arap(uvs, faces_arr, order, mesh.vertices, n_verts, arap_iterations)

    # 4b. Sanitize: replace NaN / Inf with 0 so the island stays visible
    bad_mask = ~np.isfinite(uvs)
    n_bad = int(np.sum(bad_mask))
    if n_bad:
        uvs[bad_mask] = 0.0
        logger.warning("Sanitized %d NaN/Inf UV values.", n_bad)

    # 4c. Check if UVs collapsed to a point (degenerate mesh)
    span = uvs.max(axis=0) - uvs.min(axis=0)
    if span[0] < 1e-10 and span[1] < 1e-10:
        # Fall back to simple PCA planar projection so something is visible
        logger.warning("ARAP produced collapsed UVs - falling back to PCA "
                       "projection. The mesh may have degenerate triangles.")
        verts3d = mesh.vertices[order]
        center3d = verts3d.mean(axis=0)
        centered3d = verts3d - center3d
        cov3d = centered3d.T @ centered3d
        _, evecs = np.linalg.eigh(cov3d)
        evecs = _orient_pca_projection(evecs, mesh.face_normals_computed)
        if flip_projection:
            evecs = evecs.copy()
            evecs[:, 2] *= -1
            logger.info("Flip-projection enabled - viewing from opposite side.")
        uvs = centered3d @ evecs[:, [2, 1]]

    # 5. Normalize to [margin, 1-margin]
    for k in range(2):
        mn, mx = uvs[:, k].min(), uvs[:, k].max()
        rng = mx - mn
        if rng < 1e-10:
            rng = 1.0
        uvs[:, k] = ((uvs[:, k] - mn) / rng
                      * (1 - 2 * island_margin) + island_margin)

    # 6. Write back
    mesh.uvs = uvs
    mesh.face_uvs_idx = [list(fl) for fl in faces_local]

    # 6b. Restore filtered faces and assign them collapsed UVs
    if removed_face_indices:
        mesh.faces = original_faces
        mesh.face_normals_computed = original_normals
        # Map good-face index back to original index
        good_indices = sorted(set(range(len(original_faces))) -
                              set(removed_face_indices))
        # Rebuild face_uvs_idx for original face ordering
        good_fuv = mesh.face_uvs_idx  # from the unwrapped good faces
        full_fuv = [None] * len(original_faces)
        for new_fi, orig_fi in enumerate(good_indices):
            full_fuv[orig_fi] = good_fuv[new_fi]
        # For removed faces: map their verts to existing UV indices (or add
        # a collapsed UV point at the island centre).
        centre_uv = mesh.uvs.mean(axis=0)
        collapse_idx = len(mesh.uvs)
        mesh.uvs = np.vstack([mesh.uvs, centre_uv[None, :]])
        for orig_fi in removed_face_indices:
            full_fuv[orig_fi] = [collapse_idx] * len(original_faces[orig_fi])
        mesh.face_uvs_idx = full_fuv
        logger.info("Restored %d filtered faces with collapsed UVs.",
                    len(removed_face_indices))

    logger.info("Unwrap complete - %d UV coordinates generated.", len(mesh.uvs))


def _harmonic_jacobi_fast(uvs, faces_local, order, mesh_verts,
                          boundary_set, n_verts, iterations):
    """Vectorized Jacobi harmonic solve (fallback when scipy unavailable)."""
    rows, cols, vals = [], [], []
    for fl in faces_local:
        if len(fl) < 3:
            continue
        i0, i1, i2 = fl[0], fl[1], fl[2]
        p0 = mesh_verts[order[i0]]
        p1 = mesh_verts[order[i1]]
        p2 = mesh_verts[order[i2]]
        cot0 = max(_cotangent_weight(p0, p1, p2) * 0.5, 1e-8)
        cot1 = max(_cotangent_weight(p1, p0, p2) * 0.5, 1e-8)
        cot2 = max(_cotangent_weight(p2, p0, p1) * 0.5, 1e-8)
        for a, b, w in [(i1, i2, cot0), (i0, i2, cot1), (i0, i1, cot2)]:
            rows.extend([a, b]); cols.extend([b, a]); vals.extend([w, w])

    W = sparse.coo_matrix((vals, (rows, cols)),
                          shape=(n_verts, n_verts)).tocsr()

    bnd_mask = np.zeros(n_verts, dtype=bool)
    for v in boundary_set:
        bnd_mask[v] = True

    row_sums = np.array(W.sum(axis=1)).flatten()
    row_sums = np.maximum(row_sums, 1e-12)

    for _ in range(iterations):
        new_uvs = (W @ uvs) / row_sums[:, None]
        new_uvs[bnd_mask] = uvs[bnd_mask]
        uvs[:] = new_uvs


# ======================================================================
# Public API
# ======================================================================

ALGORITHM_PARAMS = {
    "arap_iterations": (1, 200, 50),
    "island_margin": (0.0, 0.05, 0.01),
    "quality_threshold": (0.0, 0.5, 0.0),
}
