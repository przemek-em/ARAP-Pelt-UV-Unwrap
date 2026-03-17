"""
3D viewport widget for Tkinter.

Uses PyOpenGL + pyopengltk when available; falls back to a pure-Tkinter
Canvas wireframe renderer otherwise.
"""

import math
import numpy as np

# ---------- Try OpenGL ----------
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from pyopengltk import OpenGLFrame
    HAS_OPENGL = True
except Exception:
    HAS_OPENGL = False

import tkinter as tk


# ======================================================================
# OpenGL viewport (preferred)
# ======================================================================

if HAS_OPENGL:
    class GLViewport(OpenGLFrame):
        """Interactive OpenGL 3D viewport embedded in Tkinter."""

        def __init__(self, master=None, **kw):
            # Instance state - survives GL context recreation on resize
            self._az = 45.0
            self._el = 25.0
            self._dist = 5.0
            self._target = np.zeros(3)
            self._mesh = None
            self._show_wire = True
            self._show_uv_seams = True
            self._show_checker = False
            self._checker_tex = None
            self._checker_scale = 8.0
            self._last_x = 0
            self._last_y = 0
            self._last_w = 0
            self._last_h = 0
            super().__init__(master, **kw)
            self.bind("<ButtonPress-1>", self._on_lmb_press)
            self.bind("<B1-Motion>", self._on_lmb_drag)
            self.bind("<ButtonPress-3>", self._on_rmb_press)
            self.bind("<B3-Motion>", self._on_rmb_drag)
            self.bind("<MouseWheel>", self._on_scroll)
            self.bind("<ButtonPress-2>", self._on_mmb_press)
            self.bind("<B2-Motion>", self._on_mmb_drag)

        def initgl(self):
            # Only GL state here - called each time the context is (re)created
            self._checker_tex = None  # old texture ID is invalid
            self._last_w = self.winfo_width()
            self._last_h = self.winfo_height()

        # -- camera helpers --
        def _eye(self):
            az = math.radians(self._az)
            el = math.radians(self._el)
            x = self._target[0] + self._dist * math.cos(el) * math.sin(az)
            y = self._target[1] + self._dist * math.sin(el)
            z = self._target[2] + self._dist * math.cos(el) * math.cos(az)
            return x, y, z

        # -- mouse callbacks --
        def _on_lmb_press(self, e):
            self._last_x, self._last_y = e.x, e.y

        def _on_lmb_drag(self, e):
            dx = e.x - self._last_x
            dy = e.y - self._last_y
            self._az -= dx * 0.4
            self._el += dy * 0.4
            self._el = max(-89, min(89, self._el))
            self._last_x, self._last_y = e.x, e.y

        def _on_rmb_press(self, e):
            self._last_x, self._last_y = e.x, e.y

        def _on_rmb_drag(self, e):
            dx = e.x - self._last_x
            dy = e.y - self._last_y
            scale = self._dist * 0.002
            az = math.radians(self._az)
            right = np.array([math.cos(az), 0, -math.sin(az)])
            up = np.array([0, 1, 0])
            self._target -= right * dx * scale
            self._target += up * dy * scale
            self._last_x, self._last_y = e.x, e.y

        def _on_mmb_press(self, e):
            self._last_x, self._last_y = e.x, e.y

        def _on_mmb_drag(self, e):
            dx = e.x - self._last_x
            dy = e.y - self._last_y
            scale = self._dist * 0.002
            az = math.radians(self._az)
            right = np.array([math.cos(az), 0, -math.sin(az)])
            up = np.array([0, 1, 0])
            self._target -= right * dx * scale
            self._target += up * dy * scale
            self._last_x, self._last_y = e.x, e.y

        def _on_scroll(self, e):
            factor = 1.1 if e.delta < 0 else 1 / 1.1
            self._dist *= factor
            self._dist = max(0.01, self._dist)

        # -- public --
        def set_mesh(self, mesh):
            self._mesh = mesh
            if mesh is not None:
                c = mesh.get_center()
                self._target = c.copy()
                self._dist = mesh.get_scale() * 1.2

        def set_wireframe(self, val):
            self._show_wire = val

        def set_checker(self, val):
            self._show_checker = val

        def set_checker_scale(self, val):
            self._checker_scale = max(1.0, float(val))

        def _ensure_checker_texture(self):
            """Create a 2x2 checker tile; repetition is handled via UV scale."""
            if self._checker_tex is not None:
                return
            # Simple 2x2 texel checker - tiled by GL_REPEAT
            pixels = np.array([
                [200, 200, 200,  80,  80,  80],
                [ 80,  80,  80, 200, 200, 200],
            ], dtype=np.uint8)
            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 2, 2, 0,
                         GL_RGB, GL_UNSIGNED_BYTE, pixels.tobytes())
            self._checker_tex = tex_id

        # -- rendering --
        def redraw(self):
            w = self.winfo_width()
            h = self.winfo_height() or 1

            # Detect resize → invalidate checker texture (old ID is
            # stale if pyopengltk recreated the GL context).
            if w != self._last_w or h != self._last_h:
                self._last_w = w
                self._last_h = h
                self._checker_tex = None

            # Re-apply GL state every frame - pyopengltk can
            # recreate the context/FBO on resize, losing all state.
            glClearColor(0.18, 0.20, 0.22, 1.0)
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            glEnable(GL_COLOR_MATERIAL)
            glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
            glLightfv(GL_LIGHT0, GL_POSITION, [2.0, 4.0, 3.0, 0.0])
            glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.85, 0.85, 0.85, 1.0])
            glLightfv(GL_LIGHT0, GL_AMBIENT, [0.25, 0.25, 0.28, 1.0])
            glLightfv(GL_LIGHT0, GL_SPECULAR, [0.3, 0.3, 0.3, 1.0])

            glViewport(0, 0, w, h)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(45, w / h, self._dist * 0.01, self._dist * 100)

            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            ex, ey, ez = self._eye()
            gluLookAt(ex, ey, ez,
                      self._target[0], self._target[1], self._target[2],
                      0, 1, 0)

            if self._mesh is not None:
                self._draw_mesh()

        def _draw_mesh(self):
            mesh = self._mesh
            has_uvs = (len(mesh.face_uvs_idx) == len(mesh.faces)
                       and len(mesh.uvs) > 0)
            use_checker = self._show_checker and has_uvs

            # -- solid fill --
            glEnable(GL_POLYGON_OFFSET_FILL)
            glPolygonOffset(1, 1)

            if use_checker:
                self._ensure_checker_texture()
                glDisable(GL_LIGHTING)
                glEnable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, self._checker_tex)
                glColor3f(1.0, 1.0, 1.0)

                glBegin(GL_TRIANGLES)
                for i, face in enumerate(mesh.faces):
                    if i < len(mesh.face_normals_computed):
                        n = mesh.face_normals_computed[i]
                        glNormal3f(float(n[0]), float(n[1]), float(n[2]))
                    fuv = mesh.face_uvs_idx[i]
                    sc = self._checker_scale
                    for ki, vi in enumerate(face):
                        uvi = fuv[ki] if ki < len(fuv) else 0
                        if uvi < len(mesh.uvs):
                            u, v = mesh.uvs[uvi]
                            glTexCoord2f(float(u * sc), float(v * sc))
                        vp = mesh.vertices[vi]
                        glVertex3f(float(vp[0]), float(vp[1]), float(vp[2]))
                glEnd()

                glDisable(GL_TEXTURE_2D)
            else:
                glEnable(GL_LIGHTING)
                glColor3f(0.55, 0.55, 0.52)

                glBegin(GL_TRIANGLES)
                for i, face in enumerate(mesh.faces):
                    if i < len(mesh.face_normals_computed):
                        n = mesh.face_normals_computed[i]
                        glNormal3f(float(n[0]), float(n[1]), float(n[2]))
                    for vi in face:
                        v = mesh.vertices[vi]
                        glVertex3f(float(v[0]), float(v[1]), float(v[2]))
                glEnd()

            glDisable(GL_POLYGON_OFFSET_FILL)

            # -- wireframe overlay --
            if self._show_wire:
                glDisable(GL_LIGHTING)
                glColor3f(0.0, 0.0, 0.0)
                glLineWidth(1.0)
                for face in mesh.faces:
                    glBegin(GL_LINE_LOOP)
                    for vi in face:
                        v = mesh.vertices[vi]
                        glVertex3f(float(v[0]), float(v[1]), float(v[2]))
                    glEnd()
                glEnable(GL_LIGHTING)

            # -- UV seams --
            if self._show_uv_seams and len(mesh.face_uvs_idx) == len(mesh.faces):
                self._draw_seams()

        def _draw_seams(self):
            """Highlight edges where adjacent faces have different UV connectivity."""
            mesh = self._mesh
            edge_uv = {}
            for fi, face in enumerate(mesh.faces):
                if fi >= len(mesh.face_uvs_idx):
                    break
                fuv = mesh.face_uvs_idx[fi]
                n = len(face)
                for i in range(n):
                    v0, v1 = face[i], face[(i + 1) % n]
                    uv0, uv1 = fuv[i], fuv[(i + 1) % n]
                    edge = tuple(sorted((v0, v1)))
                    pair = tuple(sorted((uv0, uv1)))
                    if edge in edge_uv:
                        if edge_uv[edge] != pair:
                            edge_uv[edge] = "seam"
                    else:
                        edge_uv[edge] = pair

            glDisable(GL_LIGHTING)
            glDisable(GL_DEPTH_TEST)
            glColor3f(0.0, 1.0, 0.3)
            glLineWidth(2.5)
            glBegin(GL_LINES)
            for (v0, v1), val in edge_uv.items():
                if val == "seam":
                    a = mesh.vertices[v0]
                    b = mesh.vertices[v1]
                    glVertex3f(float(a[0]), float(a[1]), float(a[2]))
                    glVertex3f(float(b[0]), float(b[1]), float(b[2]))
            glEnd()
            glLineWidth(1.0)
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_LIGHTING)


# ======================================================================
# Canvas fallback viewport (no OpenGL required)
# ======================================================================

class CanvasViewport(tk.Canvas):
    """Minimal wireframe 3D viewer using only Tkinter Canvas."""

    def __init__(self, parent, **kw):
        kw.setdefault("bg", "#2e3235")
        kw.setdefault("highlightthickness", 0)
        super().__init__(parent, **kw)

        self._mesh = None
        self._az = 45.0
        self._el = 25.0
        self._dist = 5.0
        self._target = np.zeros(3)
        self._last_x = 0
        self._last_y = 0

        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<MouseWheel>", self._on_scroll)
        self.bind("<Configure>", lambda e: self._redraw())

    def set_mesh(self, mesh):
        self._mesh = mesh
        if mesh is not None:
            self._target = mesh.get_center().copy()
            self._dist = mesh.get_scale() * 1.2
        self._redraw()

    def set_wireframe(self, val):
        pass  # always wireframe

    def _on_press(self, e):
        self._last_x, self._last_y = e.x, e.y

    def _on_drag(self, e):
        dx = e.x - self._last_x
        dy = e.y - self._last_y
        self._az -= dx * 0.5
        self._el += dy * 0.5
        self._el = max(-89, min(89, self._el))
        self._last_x, self._last_y = e.x, e.y
        self._redraw()

    def _on_scroll(self, e):
        factor = 1.1 if e.delta < 0 else 1 / 1.1
        self._dist *= factor
        self._dist = max(0.01, self._dist)
        self._redraw()

    def _project(self, point):
        """World coord -> canvas pixel."""
        az = math.radians(self._az)
        el = math.radians(self._el)
        p = point - self._target

        # Rotate around Y (azimuth)
        cx, cz = math.cos(az), math.sin(az)
        x = p[0] * cx + p[2] * cz
        z = -p[0] * cz + p[2] * cx

        # Rotate around X (elevation)
        ce, se = math.cos(el), math.sin(el)
        y = p[1] * ce - z * se
        z2 = p[1] * se + z * ce

        w = self.winfo_width() or 400
        h = self.winfo_height() or 400
        scale = min(w, h) / (self._dist * 1.5)
        sx = w / 2 + x * scale
        sy = h / 2 - y * scale
        return sx, sy, z2

    def _redraw(self):
        self.delete("all")
        if self._mesh is None:
            self.create_text(
                self.winfo_width() // 2, self.winfo_height() // 2,
                text="Load an OBJ file", fill="#888", font=("Segoe UI", 14),
            )
            return

        mesh = self._mesh
        # Draw faces as wireframe, sorted by depth (painter's order, approx)
        face_depths = []
        for fi, face in enumerate(mesh.faces):
            pts = [self._project(mesh.vertices[vi]) for vi in face]
            avg_z = sum(p[2] for p in pts) / len(pts)
            face_depths.append((avg_z, fi, pts))
        face_depths.sort(key=lambda x: -x[0])  # far first

        for _, fi, pts in face_depths:
            coords = []
            for sx, sy, _ in pts:
                coords.extend([sx, sy])
            # Filled face with outline
            brightness = 0.35
            if fi < len(mesh.face_normals_computed):
                n = mesh.face_normals_computed[fi]
                light = np.array([0.3, 0.7, 0.5])
                light /= np.linalg.norm(light)
                brightness = 0.3 + 0.5 * max(0, np.dot(n, light))
            grey = int(brightness * 255)
            colour = f"#{grey:02x}{grey:02x}{grey:02x}"
            self.create_polygon(coords, fill=colour, outline="#111", width=1)


# ======================================================================
# Public factory
# ======================================================================

def create_viewport(parent, **kw):
    """Return the best available 3D viewport widget."""
    if HAS_OPENGL:
        vp = GLViewport(parent, **kw)
        vp.animate = 30  # ms per frame
        return vp
    return CanvasViewport(parent, **kw)
