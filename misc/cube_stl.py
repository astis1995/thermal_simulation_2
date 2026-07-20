# cube_stl.py

import numpy as np


def write_cube_stl(
    filename="cube.stl",
    size=100.0,
    units="mm"
):
    """
    Create a cube STL.

    Parameters
    ----------
    filename : str
        Output STL filename.

    size : float
        Cube side length in the specified units.

    units : str
        "m", "cm", or "mm"

    Notes
    -----
    STL files do not store units.

    Internally, coordinates are converted to meters.

    Examples
    --------
    100 mm cube:

        write_cube_stl(
            size=100,
            units="mm"
        )

    10 cm cube:

        write_cube_stl(
            size=10,
            units="cm"
        )

    0.1 m cube:

        write_cube_stl(
            size=0.1,
            units="m"
        )
    """

    unit_scale = {
        "m": 1.0,
        "cm": 1e-2,
        "mm": 1e-3
    }

    if units not in unit_scale:
        raise ValueError(
            f"Unsupported units '{units}'. "
            f"Use 'm', 'cm', or 'mm'."
        )

    size_m = size * unit_scale[units]

    v = np.array([
        [0.0,    0.0,    0.0],
        [size_m, 0.0,    0.0],
        [size_m, size_m, 0.0],
        [0.0,    size_m, 0.0],

        [0.0,    0.0,    size_m],
        [size_m, 0.0,    size_m],
        [size_m, size_m, size_m],
        [0.0,    size_m, size_m]
    ], dtype=float)

    triangles = [

        # bottom
        [0, 1, 2],
        [0, 2, 3],

        # top
        [4, 6, 5],
        [4, 7, 6],

        # front
        [0, 5, 1],
        [0, 4, 5],

        # back
        [3, 2, 6],
        [3, 6, 7],

        # left
        [0, 3, 7],
        [0, 7, 4],

        # right
        [1, 5, 6],
        [1, 6, 2]
    ]

    with open(filename, "w") as f:

        f.write("solid cube\n")

        for tri in triangles:

            p1 = v[tri[0]]
            p2 = v[tri[1]]
            p3 = v[tri[2]]

            normal = np.cross(
                p2 - p1,
                p3 - p1
            )

            norm = np.linalg.norm(normal)

            if norm > 0:
                normal /= norm

            f.write(
                f"facet normal "
                f"{normal[0]} "
                f"{normal[1]} "
                f"{normal[2]}\n"
            )

            f.write(" outer loop\n")

            for p in (p1, p2, p3):

                f.write(
                    f"  vertex "
                    f"{p[0]} "
                    f"{p[1]} "
                    f"{p[2]}\n"
                )

            f.write(" endloop\n")
            f.write("endfacet\n")

        f.write("endsolid cube\n")

    print(
        f"✔ Cube STL written: {filename}"
    )

    print(
        f"✔ Size: "
        f"{size_m:.6f} m × "
        f"{size_m:.6f} m × "
        f"{size_m:.6f} m"
    )


if __name__ == "__main__":

    # Generates a 0.1 m × 0.1 m × 0.1 m cube

    write_cube_stl(
        filename="cube.stl",
        size=100,
        units="mm"
    )
