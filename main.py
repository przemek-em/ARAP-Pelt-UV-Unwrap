"""
UV As-Rigid-As-Possible Pelt Unwrap Tool – main application.

A Tkinter GUI for loading .obj meshes, UV-unwrapping them with algorithms
tuned for rocks / cliffs / natural environments, previewing in 3D, and
exporting the result.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import math
import numpy as np
import threading
import os

from mesh import Mesh, load_obj, save_obj
from uv_algorithms import pelt_unwrap, ALGORITHM_PARAMS
from viewer3d import create_viewport, HAS_OPENGL

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ======================================================================
# 2-D UV canvas
# ======================================================================

class UVCanvas(tk.Canvas):
    """Interactive 2-D view of the UV layout."""

    ISLAND_COLOURS = [
        "#e06c75", "#61afef", "#98c379", "#e5c07b",
        "#c678dd", "#56b6c2", "#d19a66", "#be5046",
        "#7ec8e3", "#c3e88d", "#f78c6c", "#89ddff",
    ]

    def __init__(self, parent, **kw):
        kw.setdefault("bg", "#1a1a2e")
        kw.setdefault("highlightthickness", 0)
        super().__init__(parent, **kw)
        self._mesh = None
        self._islands = None
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._last_x = 0
        self._last_y = 0
        self._show_grid = True
        self._show_fill = True

        self.bind("<MouseWheel>", self._on_scroll)
        self.bind("<ButtonPress-2>", self._on_mmb)
        self.bind("<B2-Motion>", self._on_mmb_drag)
        self.bind("<ButtonPress-3>", self._on_mmb)
        self.bind("<B3-Motion>", self._on_mmb_drag)
        self.bind("<Configure>", lambda e: self.redraw())

    # -- public --
    def set_mesh(self, mesh, islands=None):
        self._mesh = mesh
        self._islands = islands
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.redraw()

    def set_show_grid(self, v):
        self._show_grid = v
        self.redraw()

    def set_show_fill(self, v):
        self._show_fill = v
        self.redraw()

    # -- mouse --
    def _on_scroll(self, e):
        factor = 1.15 if e.delta > 0 else 1 / 1.15
        self._zoom *= factor
        self._zoom = max(0.1, min(50, self._zoom))
        self.redraw()

    def _on_mmb(self, e):
        self._last_x, self._last_y = e.x, e.y

    def _on_mmb_drag(self, e):
        self._pan_x += e.x - self._last_x
        self._pan_y += e.y - self._last_y
        self._last_x, self._last_y = e.x, e.y
        self.redraw()

    # -- UV → canvas transform --
    def _uv_to_px(self, u, v):
        w = self.winfo_width() or 400
        h = self.winfo_height() or 400
        s = min(w, h) * 0.85 * self._zoom
        ox = (w - s) / 2 + self._pan_x
        oy = (h - s) / 2 + self._pan_y
        return ox + u * s, oy + (1 - v) * s

    # -- drawing --
    def redraw(self):
        self.delete("all")
        w = self.winfo_width() or 400
        h = self.winfo_height() or 400

        if self._mesh is None or len(self._mesh.uvs) == 0:
            self.create_text(
                w // 2, h // 2,
                text="No UV data.\nLoad a mesh and run Unwrap.",
                fill="#555", font=("Segoe UI", 13), justify="center",
            )
            return

        # Grid (unit square)
        if self._show_grid:
            self._draw_grid()

        # Build face→island colour map
        face_colour = {}
        if self._islands:
            for ii, island in enumerate(self._islands):
                col = self.ISLAND_COLOURS[ii % len(self.ISLAND_COLOURS)]
                for fi in island:
                    face_colour[fi] = col

        mesh = self._mesh
        for fi in range(len(mesh.faces)):
            if fi >= len(mesh.face_uvs_idx):
                continue
            fuv = mesh.face_uvs_idx[fi]
            coords = []
            for uvi in fuv:
                if uvi >= len(mesh.uvs):
                    break
                u, v = mesh.uvs[uvi]
                px, py = self._uv_to_px(u, v)
                coords.extend([px, py])
            if len(coords) < 6:
                continue
            fill = ""
            if self._show_fill and fi in face_colour:
                fill = face_colour[fi]
            outline = face_colour.get(fi, "#557788")
            self.create_polygon(coords, outline=outline, fill=fill,
                                stipple="gray25" if fill else "", width=1)

    def _draw_grid(self):
        steps = 10
        for i in range(steps + 1):
            t = i / steps
            x0, y0 = self._uv_to_px(t, 0)
            x1, y1 = self._uv_to_px(t, 1)
            self.create_line(x0, y0, x1, y1, fill="#2a2a44", width=1)
            x0, y0 = self._uv_to_px(0, t)
            x1, y1 = self._uv_to_px(1, t)
            self.create_line(x0, y0, x1, y1, fill="#2a2a44", width=1)
        # Unit square border
        corners = [self._uv_to_px(0, 0), self._uv_to_px(1, 0),
                    self._uv_to_px(1, 1), self._uv_to_px(0, 1)]
        for i in range(4):
            x0, y0 = corners[i]
            x1, y1 = corners[(i + 1) % 4]
            self.create_line(x0, y0, x1, y1, fill="#3a3a5e", width=2)


# ======================================================================
# Main application
# ======================================================================

class UVUnwrapApp:
    def __init__(self, root):
        self.root = root
        self.root.title("UV As-Rigid-As-Possible Pelt Unwrap Tool")
        self.root.geometry("1300x780")
        self.root.minsize(900, 550)
        self.root.configure(bg="#222")

        self.mesh = None
        self.islands = None
        self._filepath = None

        self._build_menu()
        self._build_toolbar()
        self._build_panes()
        self._build_controls()
        self._build_status()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_menu(self):
        mb = tk.Menu(self.root)
        self.root.config(menu=mb)

        file_menu = tk.Menu(mb, tearoff=0)
        file_menu.add_command(label="Open OBJ…", command=self.load_mesh, accelerator="Ctrl+O")
        file_menu.add_command(label="Save OBJ…", command=self.export_obj, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Export UV Image…", command=self.export_uv_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        mb.add_cascade(label="File", menu=file_menu)

        uv_menu = tk.Menu(mb, tearoff=0)
        uv_menu.add_command(label="Unwrap (Pelt ARAP)", command=self.run_unwrap)
        mb.add_cascade(label="UV", menu=uv_menu)

        view_menu = tk.Menu(mb, tearoff=0)
        self._wire_var = tk.BooleanVar(value=True)
        view_menu.add_checkbutton(label="Wireframe overlay", variable=self._wire_var,
                                  command=self._toggle_wire)
        self._grid_var = tk.BooleanVar(value=True)
        view_menu.add_checkbutton(label="UV grid", variable=self._grid_var,
                                  command=lambda: self.uv_canvas.set_show_grid(self._grid_var.get()))
        self._fill_var = tk.BooleanVar(value=True)
        view_menu.add_checkbutton(label="UV island fill", variable=self._fill_var,
                                  command=lambda: self.uv_canvas.set_show_fill(self._fill_var.get()))
        view_menu.add_separator()
        self._checker_var = tk.BooleanVar(value=False)
        view_menu.add_checkbutton(label="Checker texture (3D)", variable=self._checker_var,
                                  command=self._toggle_checker)
        mb.add_cascade(label="View", menu=view_menu)

        self.root.bind_all("<Control-o>", lambda e: self.load_mesh())
        self.root.bind_all("<Control-s>", lambda e: self.export_obj())

    def _build_toolbar(self):
        tb = ttk.Frame(self.root)
        tb.pack(side="top", fill="x", padx=4, pady=(4, 0))

        ttk.Button(tb, text="Open OBJ", command=self.load_mesh).pack(side="left", padx=2)
        ttk.Button(tb, text="Save OBJ", command=self.export_obj).pack(side="left", padx=2)
        ttk.Separator(tb, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(tb, text="Unwrap", command=self.run_unwrap).pack(side="left", padx=2)
        ttk.Button(tb, text="Export UV Image", command=self.export_uv_image).pack(side="left", padx=2)

        ttk.Separator(tb, orient="vertical").pack(side="left", fill="y", padx=6)

        # Checker texture controls
        self._checker_tb_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tb, text="Checker", variable=self._checker_tb_var,
                        command=self._toggle_checker_tb).pack(side="left", padx=2)
        ttk.Label(tb, text="Size:").pack(side="left", padx=(2, 0))
        self._checker_scale_var = tk.DoubleVar(value=8.0)
        self._checker_scale_spin = ttk.Spinbox(
            tb, from_=1, to=64, increment=1,
            textvariable=self._checker_scale_var, width=4,
            command=self._on_checker_scale_changed,
        )
        self._checker_scale_spin.pack(side="left", padx=(0, 2))
        self._checker_scale_spin.bind("<Return>", lambda e: self._on_checker_scale_changed())

    def _build_panes(self):
        pw = ttk.PanedWindow(self.root, orient="horizontal")
        pw.pack(fill="both", expand=True, padx=4, pady=4)

        # 3D viewport
        vp_frame = ttk.LabelFrame(pw, text="3D Viewport")
        self.viewport = create_viewport(vp_frame, width=550, height=500)
        self.viewport.pack(fill="both", expand=True)
        pw.add(vp_frame, weight=1)

        # UV canvas
        uv_frame = ttk.LabelFrame(pw, text="UV Layout")
        self.uv_canvas = UVCanvas(uv_frame, width=550, height=500)
        self.uv_canvas.pack(fill="both", expand=True)
        pw.add(uv_frame, weight=1)

    def _build_controls(self):
        cf = ttk.LabelFrame(self.root, text="Parameters")
        cf.pack(fill="x", padx=4, pady=(0, 2))

        self.param_frame = ttk.Frame(cf)
        self.param_frame.pack(fill="x", padx=6, pady=4)
        self._param_widgets = {}
        self._update_param_ui()

    def _build_status(self):
        sf = ttk.Frame(self.root)
        sf.pack(fill="x", side="bottom", padx=4, pady=(0, 4))
        self.status_var = tk.StringVar(value="Ready. Load an OBJ file to begin.")
        ttk.Label(sf, textvariable=self.status_var, anchor="w").pack(side="left", fill="x", expand=True)
        self.info_var = tk.StringVar(value="")
        ttk.Label(sf, textvariable=self.info_var, anchor="e").pack(side="right")

    # ------------------------------------------------------------------
    # Parameter UI (dynamic based on algorithm)
    # ------------------------------------------------------------------

    def _update_param_ui(self):
        for w in self.param_frame.winfo_children():
            w.destroy()
        self._param_widgets.clear()

        col = 0
        for name, (lo, hi, default) in ALGORITHM_PARAMS.items():
            label = name.replace("_", " ").title()
            ttk.Label(self.param_frame, text=f"{label}:").grid(row=0, column=col, padx=(8, 2))
            col += 1
            if isinstance(default, float):
                var = tk.DoubleVar(value=default)
                sp = ttk.Spinbox(self.param_frame, from_=lo, to=hi, increment=0.001,
                                 textvariable=var, width=8)
            else:
                var = tk.IntVar(value=default) if isinstance(default, int) else tk.DoubleVar(value=default)
                sp = ttk.Spinbox(self.param_frame, from_=lo, to=hi, increment=1,
                                 textvariable=var, width=8)
            sp.grid(row=0, column=col, padx=(0, 8))
            col += 1
            self._param_widgets[name] = var

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def load_mesh(self):
        path = filedialog.askopenfilename(
            title="Open OBJ mesh",
            filetypes=[("Wavefront OBJ", "*.obj"), ("All files", "*.*")],
        )
        if not path:
            return
        self.status_var.set(f"Loading {os.path.basename(path)}…")
        self.root.update_idletasks()
        try:
            self.mesh = load_obj(path)
            self._filepath = path
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load OBJ:\n{exc}")
            self.status_var.set("Error loading file.")
            return
        nv = len(self.mesh.vertices)
        nf = len(self.mesh.faces)
        has_uv = "yes" if len(self.mesh.face_uvs_idx) == len(self.mesh.faces) else "no"
        self.info_var.set(f"Verts: {nv}  |  Faces: {nf}  |  Has UVs: {has_uv}")
        self.status_var.set(f"Loaded {os.path.basename(path)}")
        self.viewport.set_mesh(self.mesh)
        self.uv_canvas.set_mesh(self.mesh)
        self.islands = None

    def run_unwrap(self):
        if self.mesh is None:
            messagebox.showinfo("Info", "Load an OBJ mesh first.")
            return

        kwargs = {}
        for name in ALGORITHM_PARAMS:
            var = self._param_widgets.get(name)
            if var is not None:
                kwargs[name] = var.get()

        self.status_var.set("Unwrapping with Pelt ARAP…")
        self.root.update_idletasks()

        def _run():
            try:
                pelt_unwrap(self.mesh, **kwargs)
                self.islands = [list(range(len(self.mesh.faces)))]
                self.root.after(0, self._post_unwrap)
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror("Error", str(exc)))
                self.root.after(0, lambda: self.status_var.set("Unwrap failed."))

        threading.Thread(target=_run, daemon=True).start()

    def _post_unwrap(self):
        nuv = len(self.mesh.uvs)
        self.status_var.set(f"Pelt ARAP complete — {nuv} UV coords generated.")
        self.uv_canvas.set_mesh(self.mesh, self.islands)
        # Refresh 3D viewport to show seams
        if hasattr(self.viewport, "set_mesh"):
            self.viewport.set_mesh(self.mesh)

    def export_obj(self):
        if self.mesh is None:
            messagebox.showinfo("Info", "No mesh loaded.")
            return
        init_name = os.path.splitext(os.path.basename(self._filepath or "mesh"))[0] + "_uv.obj"
        path = filedialog.asksaveasfilename(
            title="Save OBJ",
            initialfile=init_name,
            filetypes=[("Wavefront OBJ", "*.obj"), ("All files", "*.*")],
            defaultextension=".obj",
        )
        if not path:
            return
        try:
            save_obj(path, self.mesh)
            self.status_var.set(f"Saved {os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to save:\n{exc}")

    def export_uv_image(self):
        if not HAS_PIL:
            messagebox.showwarning("Missing dependency",
                                   "Pillow is required for UV image export.\n"
                                   "Install with: pip install Pillow")
            return
        if self.mesh is None or len(self.mesh.uvs) == 0:
            messagebox.showinfo("Info", "No UV data to export.")
            return
        path = filedialog.asksaveasfilename(
            title="Export UV Image",
            initialfile="uv_layout.png",
            filetypes=[("PNG", "*.png"), ("All files", "*.*")],
            defaultextension=".png",
        )
        if not path:
            return

        size = 2048
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        colours = UVCanvas.ISLAND_COLOURS
        face_colour = {}
        if self.islands:
            for ii, island in enumerate(self.islands):
                for fi in island:
                    face_colour[fi] = colours[ii % len(colours)]

        for fi in range(len(self.mesh.faces)):
            if fi >= len(self.mesh.face_uvs_idx):
                continue
            fuv = self.mesh.face_uvs_idx[fi]
            pts = []
            for uvi in fuv:
                if uvi >= len(self.mesh.uvs):
                    break
                u, v = self.mesh.uvs[uvi]
                pts.append((u * size, (1 - v) * size))
            if len(pts) < 3:
                continue
            col = face_colour.get(fi, "#ffffff")
            draw.polygon(pts, outline=col, fill=None)

        img.save(path)
        self.status_var.set(f"UV image saved to {os.path.basename(path)}")

    def _toggle_wire(self):
        if hasattr(self.viewport, "set_wireframe"):
            self.viewport.set_wireframe(self._wire_var.get())

    def _toggle_checker(self):
        if hasattr(self.viewport, "set_checker"):
            self.viewport.set_checker(self._checker_var.get())
        self._checker_tb_var.set(self._checker_var.get())

    def _toggle_checker_tb(self):
        val = self._checker_tb_var.get()
        self._checker_var.set(val)
        if hasattr(self.viewport, "set_checker"):
            self.viewport.set_checker(val)

    def _on_checker_scale_changed(self):
        if hasattr(self.viewport, "set_checker_scale"):
            self.viewport.set_checker_scale(self._checker_scale_var.get())


# ======================================================================
# Entry point
# ======================================================================

def main():
    root = tk.Tk()

    # Use a modern-ish dark theme if available
    style = ttk.Style()
    try:
        available = style.theme_names()
        if "clam" in available:
            style.theme_use("clam")
    except Exception:
        pass

    # Dark-ish colours
    style.configure(".", background="#2b2b2b", foreground="#ddd",
                     fieldbackground="#333", troughcolor="#333")
    style.configure("TButton", padding=4)
    style.configure("TLabelframe", background="#2b2b2b", foreground="#aaa")
    style.configure("TLabelframe.Label", background="#2b2b2b", foreground="#aaa")
    style.configure("TLabel", background="#2b2b2b", foreground="#ccc")
    style.configure("TFrame", background="#2b2b2b")
    style.configure("TPanedwindow", background="#2b2b2b")

    app = UVUnwrapApp(root)

    # Print renderer info
    renderer = "OpenGL (pyopengltk)" if HAS_OPENGL else "Canvas (fallback)"
    print(f"[UV Unwrap Tool] 3D renderer: {renderer}")

    root.mainloop()


if __name__ == "__main__":
    main()
