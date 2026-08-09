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

    # User-requested coordinates.
    # Can contain 2 or 3 values.
    requested: np.ndarray

    variables: list[str]

    # Actual mesh coordinate selected.
    actual: np.ndarray | None = None

    # Cell used to evaluate the finite-element field.
    cell: int | None = None

    # Geometry point / vertex index.
    vertex: int | None = None

    # Selection mode.
    mode: str | None = None

    # Distance from requested location.
    distance: float | None = None


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

        search_cfg = (
            config.get("search", {})
            if config
            else {}
        )

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
        # Mesh topology
        # --------------------------------------------------

        self.tdim = self.mesh.topology.dim

        # Vertex -> cell connectivity.
        self.mesh.topology.create_connectivity(
            0,
            self.tdim,
        )

        self.vertex_to_cells = (
            self.mesh.topology.connectivity(
                0,
                self.tdim,
            )
        )

        # --------------------------------------------------
        # Geometry coordinates
        # --------------------------------------------------

        self.coordinates = np.asarray(
            self.mesh.geometry.x,
            dtype=np.float64,
        )

        # --------------------------------------------------
        # Build search trees ONCE
        # --------------------------------------------------

        self.bb_tree = geometry.bb_tree(
            self.mesh,
            self.tdim,
        )

        num_cells = (
            self.mesh.topology
            .index_map(self.tdim)
            .size_local
        )

        cells = np.arange(
            num_cells,
            dtype=np.int32,
        )

        self.midpoint_tree = (
            geometry.create_midpoint_tree(
                self.mesh,
                self.tdim,
                cells,
            )
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

            location = np.asarray(
                p["location"],
                dtype=np.float64,
            )

            if location.size not in (2, 3):

                raise ValueError(
                    f"ROI '{p['name']}' must contain "
                    f"2 or 3 coordinates. "
                    f"Got {location.size}."
                )

            point = ROIPoint(
                name=p["name"],
                requested=location,
                variables=p.get(
                    "variables",
                    ["temperature"],
                ),
            )

            # --------------------------------------------------
            # Resolve requested location
            # --------------------------------------------------

            self._resolve_point(point)

            # --------------------------------------------------
            # Diagnostics
            # --------------------------------------------------

            if point.actual is None:

                print(
                    f"⚠ ROI '{point.name}' "
                    f"could not be located."
                )

            else:

                print(
                    f"✔ ROI '{point.name}'"
                )

                print(
                    f"   requested = "
                    f"{point.requested}"
                )

                print(
                    f"   actual    = "
                    f"{point.actual}"
                )

                print(
                    f"   distance  = "
                    f"{point.distance:.6e} m"
                )

                print(
                    f"   mode      = "
                    f"{point.mode}"
                )

                print(
                    f"   cell      = "
                    f"{point.cell}"
                )

                print(
                    f"   vertex    = "
                    f"{point.vertex}"
                )

            self.points.append(point)

    # ======================================================
    # RESOLVE POINT
    # ======================================================

    def _resolve_point(
        self,
        point: ROIPoint,
    ):

        if point.requested.size == 3:

            self._resolve_3d(point)

        elif point.requested.size == 2:

            self._resolve_2d(point)

        else:

            raise ValueError(
                "ROI coordinates must contain "
                "2 or 3 values."
            )

    # ======================================================
    # 3D POINT
    # ======================================================

    def _resolve_3d(
        self,
        point: ROIPoint,
    ):

        target = point.requested

        coords = self.coordinates

        # --------------------------------------------------
        # Euclidean 3D distance
        # --------------------------------------------------

        delta = coords - target

        distances = np.linalg.norm(
            delta,
            axis=1,
        )

        vertex = int(
            np.argmin(distances)
        )

        distance = float(
            distances[vertex]
        )

        # --------------------------------------------------
        # Tolerance
        # --------------------------------------------------

        if (
            self.mode == "exact"
            and distance > 0.0
        ):

            point.actual = None
            point.vertex = None
            point.cell = None
            point.distance = distance
            point.mode = "3D-exact-failed"

            return

        if distance > self.tolerance:

            print(
                f"⚠ ROI '{point.name}': "
                f"nearest 3D mesh point is "
                f"{distance:.6e} m away "
                f"(tolerance="
                f"{self.tolerance:.6e} m)"
            )

            point.actual = None
            point.vertex = None
            point.cell = None
            point.distance = distance
            point.mode = "3D-nearest-outside-tolerance"

            return

        # --------------------------------------------------
        # Store selected mesh point
        # --------------------------------------------------

        point.vertex = vertex

        point.actual = coords[
            vertex
        ].copy()

        point.distance = distance

        point.mode = "3D-nearest"

        # --------------------------------------------------
        # Find a cell containing this vertex
        # --------------------------------------------------

        links = self.vertex_to_cells.links(
            vertex
        )

        if len(links) == 0:

            point.cell = None

            print(
                f"⚠ ROI '{point.name}': "
                f"vertex {vertex} has no "
                f"connected cell."
            )

            return

        point.cell = int(
            links[0]
        )

    # ======================================================
    # 2D POINT
    # ======================================================

    def _resolve_2d(
        self,
        point: ROIPoint,
    ):

        target_xy = point.requested

        coords = self.coordinates

        # --------------------------------------------------
        # Distance only in XY
        # --------------------------------------------------

        delta_xy = (
            coords[:, :2]
            - target_xy
        )

        distances_xy = np.linalg.norm(
            delta_xy,
            axis=1,
        )

        # --------------------------------------------------
        # Find all mesh points close to requested XY
        # --------------------------------------------------

        candidates = np.where(
            distances_xy
            <= self.tolerance
        )[0]

        # --------------------------------------------------
        # No point inside tolerance
        # --------------------------------------------------

        if len(candidates) == 0:

            nearest = int(
                np.argmin(
                    distances_xy
                )
            )

            nearest_distance = float(
                distances_xy[nearest]
            )

            print(
                f"⚠ ROI '{point.name}': "
                f"no mesh point found within "
                f"XY tolerance "
                f"{self.tolerance:.6e} m."
            )

            print(
                f"   Nearest XY distance = "
                f"{nearest_distance:.6e} m"
            )

            point.actual = None
            point.vertex = None
            point.cell = None
            point.distance = nearest_distance
            point.mode = "2D-no-candidate"

            return

        # --------------------------------------------------
        # Highest Z wins
        # --------------------------------------------------

        candidate_z = coords[
            candidates,
            2,
        ]

        highest_z = np.max(
            candidate_z
        )

        highest = candidates[
            np.isclose(
                candidate_z,
                highest_z,
            )
        ]

        # --------------------------------------------------
        # If several have the same Z,
        # choose the closest in XY.
        # --------------------------------------------------

        if len(highest) > 1:

            best = highest[
                np.argmin(
                    distances_xy[
                        highest
                    ]
                )
            ]

        else:

            best = highest[0]

        vertex = int(best)

        distance = float(
            distances_xy[vertex]
        )

        # --------------------------------------------------
        # Store selected mesh point
        # --------------------------------------------------

        point.vertex = vertex

        point.actual = coords[
            vertex
        ].copy()

        point.distance = distance

        point.mode = "2D-highest-Z"

        # --------------------------------------------------
        # Find a cell containing this vertex
        # --------------------------------------------------

        links = self.vertex_to_cells.links(
            vertex
        )

        if len(links) == 0:

            point.cell = None

            print(
                f"⚠ ROI '{point.name}': "
                f"vertex {vertex} has no "
                f"connected cell."
            )

            return

        point.cell = int(
            links[0]
        )

    # ======================================================
    # SAMPLE
    # ======================================================

    def sample(
        self,
        time,
        field,
        energy=None,
    ):

        row = {
            "time": float(time)
        }

        # --------------------------------------------------
        # Energy information
        # --------------------------------------------------

        if energy is not None:

            for key, value in energy.items():

                row[key] = float(value)

        # --------------------------------------------------
        # ROI temperatures
        # --------------------------------------------------

        for point in self.points:

            if (
                point.cell is None
                or point.actual is None
            ):
                continue

            x = point.actual.reshape(
                (1, 3)
            )

            value = field.eval(
                x,
                np.array(
                    [point.cell],
                    dtype=np.int32,
                ),
            )[0]

            if "temperature" in point.variables:

                row[
                    f"{point.name}.temperature"
                ] = float(value)

        # --------------------------------------------------
        # Store row
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Basic simulation columns
        # --------------------------------------------------

        columns.append("time")

        # --------------------------------------------------
        # ROI metadata
        # --------------------------------------------------

        for point in self.points:

            columns.extend(
                [
                    f"{point.name}.requested_x",
                    f"{point.name}.requested_y",
                    f"{point.name}.requested_z",
                    f"{point.name}.actual_x",
                    f"{point.name}.actual_y",
                    f"{point.name}.actual_z",
                    f"{point.name}.distance",
                    f"{point.name}.mode",
                ]
            )

        # --------------------------------------------------
        # Dynamic columns
        # --------------------------------------------------

        for row in self.history:

            for key in row.keys():

                if key not in columns:

                    columns.append(key)

        # --------------------------------------------------
        # Add ROI metadata to every row
        # --------------------------------------------------

        output_rows = []

        for row in self.history:

            output = dict(row)

            for point in self.points:

                requested = point.requested

                output[
                    f"{point.name}.requested_x"
                ] = float(
                    requested[0]
                )

                output[
                    f"{point.name}.requested_y"
                ] = float(
                    requested[1]
                )

                output[
                    f"{point.name}.requested_z"
                ] = (
                    float(requested[2])
                    if requested.size == 3
                    else np.nan
                )

                if point.actual is not None:

                    output[
                        f"{point.name}.actual_x"
                    ] = float(
                        point.actual[0]
                    )

                    output[
                        f"{point.name}.actual_y"
                    ] = float(
                        point.actual[1]
                    )

                    output[
                        f"{point.name}.actual_z"
                    ] = float(
                        point.actual[2]
                    )

                else:

                    output[
                        f"{point.name}.actual_x"
                    ] = np.nan

                    output[
                        f"{point.name}.actual_y"
                    ] = np.nan

                    output[
                        f"{point.name}.actual_z"
                    ] = np.nan

                output[
                    f"{point.name}.distance"
                ] = (
                    float(point.distance)
                    if point.distance is not None
                    else np.nan
                )

                output[
                    f"{point.name}.mode"
                ] = point.mode

            output_rows.append(output)

        # --------------------------------------------------
        # Write CSV
        # --------------------------------------------------

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

            for row in output_rows:

                writer.writerow(row)

        print(
            f"📈 ROI data saved to {path}"
        )
