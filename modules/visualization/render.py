# modules/visualization/render.py

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import imageio.v2 as imageio


class HeatVisualizer:
    def __init__(self, mesh, output_dir="outputs/frames", cmap="inferno"):
        self.mesh = mesh
        self.output_dir = output_dir
        self.cmap = cmap

        os.makedirs(self.output_dir, exist_ok=True)

        # --------------------------------------------------
        # Geometry
        # --------------------------------------------------
        coords = mesh.geometry.x
        self.x = coords[:, 0]
        self.y = coords[:, 1]
        self.z = coords[:, 2]

        # --------------------------------------------------
        # Connectivity (ROBUST extraction)
        # --------------------------------------------------
        mesh.topology.create_connectivity(2, 0)
        conn = mesh.topology.connectivity(2, 0)

        cells = conn.array
        offsets = conn.offsets

        num_cells = len(offsets) - 1
        triangles = np.zeros((num_cells, 3), dtype=np.int32)

        valid_count = 0

        for i in range(num_cells):
            start = offsets[i]
            end = offsets[i + 1]

            tri = cells[start:end]

            # Strict validation
            if len(tri) == 3 and np.all(tri < len(self.x)):
                triangles[valid_count] = tri
                valid_count += 1

        # Trim invalid entries (if any)
        triangles = triangles[:valid_count]

        if len(triangles) == 0:
            raise ValueError(" No valid triangles extracted")

        print(f"🔺 Triangles: {len(triangles)}")
        print(f"🔺 Points: {len(self.x)}")

        self.tri = Triangulation(self.x, self.y, triangles=triangles)

        self.frame_paths = []

    # --------------------------------------------------
    # Save one frame
    # --------------------------------------------------
    def save_frame(self, u, step, view="x"):
        values = u.x.array

        # -------------------------
        # Choose projection
        # -------------------------
        if view == "z":
            px, py = self.x, self.y
        elif view == "-z":
            px, py = self.x, -self.y
        elif view == "x":
            px, py = self.y, self.z
        elif view == "-x":
            px, py = self.y, -self.z
        elif view == "y":
            px, py = self.x, self.z
        elif view == "-y":
            px, py = self.x, -self.z
        else:
            raise ValueError(f" Invalid view: {view}")

        print(f"📸 Frame {step} | view={view}")

        # -------------------------
        # Plot
        # -------------------------
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)

        tri_proj = Triangulation(px, py, triangles=self.tri.triangles)

        tpc = ax.tripcolor(
            tri_proj,
            values,
            shading="gouraud",
            cmap=self.cmap
        )

        ax.set_title(f"Step {step} | view={view}")
        ax.set_aspect("equal")
        ax.axis("off")

        # Colorbar (fixed scale optional)
        tpc.set_clim(vmin=0, vmax=100)
        fig.colorbar(tpc, ax=ax, shrink=0.8)

        path = os.path.join(self.output_dir, f"frame_{step:04d}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        self.frame_paths.append(path)

    # --------------------------------------------------
    # Build GIF (non-streaming, optional)
    # --------------------------------------------------
    def create_gif(self, gif_path="outputs/heat_simulation.gif", fps=10):
        print("\n🎬 Creating GIF...")

        images = [imageio.imread(p) for p in self.frame_paths]
        imageio.mimsave(gif_path, images, fps=fps)

        print(f"✅ GIF saved: {gif_path}")
