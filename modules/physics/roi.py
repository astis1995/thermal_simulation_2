# modules/physics/roi.py

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np

from dolfinx import geometry


# ==========================================================
# ROI POINT
# ==========================================================

@dataclass
class ROIPoint:
    name: str
    location: np.ndarray
    variables: list[str]

    cell: int | None = None


# ==========================================================
# POINT TRACKER
# ==========================================================

class PointTracker:

    def __init__(
        self,
        mesh,
        function_space,
        config,
        output_dir,
        filename="roi.csv",
    ):

        print("PointTracker constructor")
        print(config)
        
        self.mesh = mesh
        self.V = function_space

        self.output_dir = output_dir
        self.filename = filename

        self.points: list[ROIPoint] = []
        self.history: list[dict] = []

        # --------------------------------------------------
        # Search configuration
        # --------------------------------------------------

        search_cfg = config.get("search", {}) if config else {}

        self.mode = search_cfg.get(
            "mode",
            "nearest",
        ).lower()

        self.tolerance = float(
            search_cfg.get(
                "tolerance",
                5e-4,
            )
        )

        # --------------------------------------------------
        # Build search trees ONCE
        # --------------------------------------------------

        self.bb_tree = geometry.bb_tree(
            self.mesh,
            self.mesh.topology.dim,
        )

        num_cells = (
            self.mesh.topology
            .index_map(
                self.mesh.topology.dim
            )
            .size_local
        )

        cells = np.arange(
            num_cells,
            dtype=np.int32,
        )

        self.midpoint_tree = geometry.create_midpoint_tree(
            self.mesh,
            self.mesh.topology.dim,
            cells,
        )

        # --------------------------------------------------
        # No ROI defined
        # --------------------------------------------------

        if config is None:
            return

        # --------------------------------------------------
        # Read points
        # --------------------------------------------------

        for p in config.get("points", []):

            point = ROIPoint(
                name=p["name"],
                location=np.asarray(
                    p["location"],
                    dtype=np.float64,
                ),
                variables=p.get(
                    "variables",
                    ["temperature"],
                ),
            )

            point.cell = self._find_cell(
                point.location
            )

            if point.cell is None:

                print(
                    f"⚠ ROI '{point.name}' "
                    f"could not be located."
                )

            else:

                print(
                    f"✔ ROI '{point.name}' "
                    f"using cell {point.cell}"
                )

            self.points.append(point)

    # ======================================================
    # FIND CELL
    # ======================================================

    def _find_cell(
        self,
        x: np.ndarray,
    ):

        x = np.asarray(
            x,
            dtype=np.float64,
        ).reshape((1, 3))

        # --------------------------------------------------
        # First try: point inside the mesh
        # --------------------------------------------------

        candidates = geometry.compute_collisions_points(
            self.bb_tree,
            x,
        )

        colliding = geometry.compute_colliding_cells(
            self.mesh,
            candidates,
            x,
        )

        if len(colliding.links(0)) > 0:
            return colliding.links(0)[0]

        # --------------------------------------------------
        # Exact mode
        # --------------------------------------------------

        if self.mode == "exact":
            return None

        # --------------------------------------------------
        # Nearest-cell fallback
        # --------------------------------------------------

        cell = geometry.compute_closest_entity(
            self.bb_tree,
            self.midpoint_tree,
            self.mesh,
            x,
        )

        if cell < 0:
            return None

        # --------------------------------------------------
        # Compute distance to nearest cell
        # --------------------------------------------------

        distance2 = geometry.squared_distance(
            self.mesh,
            self.mesh.topology.dim,
            np.array([cell], dtype=np.int32),
            x,
        )[0]

        distance = np.sqrt(distance2)

        if distance > self.tolerance:

            print(
                f"⚠ Closest cell is {distance:.6e} m "
                f"away (tolerance={self.tolerance:.6e} m)"
            )

            return None

        print(
            f"ℹ Using nearest cell {cell} "
            f"(distance={distance:.6e} m)"
        )

        return int(cell)

    # ======================================================
    # SAMPLE
    # ======================================================

    def sample(
        self,
        time,
        field,
    ):

        row = {
            "time": float(time)
        }

        for point in self.points:

            if point.cell is None:
                continue

            x = point.location.reshape((1, 3))

            value = field.eval(
                x,
                np.array(
                    [point.cell],
                    dtype=np.int32,
                ),
            )[0]

            if "temperature" in point.variables:
                row[f"{point.name}.temperature"] = float(value)

        self.history.append(row)

    # ======================================================
    # WRITE CSV
    # ======================================================

    def write_csv(self):

        if len(self.history) == 0:
            return

        path = os.path.join(
            self.output_dir,
            self.filename,
        )

        columns = []

        for row in self.history:
            for k in row.keys():
                if k not in columns:
                    columns.append(k)

        with open(
            path,
            "w",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=columns,
            )

            writer.writeheader()

            for row in self.history:
                writer.writerow(row)

        print(f"📈 ROI data saved to {path}")
