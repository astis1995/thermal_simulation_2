# rectangle_stl.py

import numpy as np


def write_rectangle_stl(
    filename="rectangle.stl",
    width=100.0,
    height=100.0,
    units="mm"
):
    """
    Create a rectangular STL surface.

    Parameters
    ----------
    filename : str
        Output STL filename.

    width : float
        Rectangle width in the specified units.

    height : float
        Rectangle height in the specified units.

    units : str
        "m", "cm", or "mm"

    Notes
    -----
    STL files do not store units.

    Internally, coordinates are converted to meters
    before being written.

    Examples
    --------
    100 mm x 100 mm:

        write_rectangle_stl(
            width=100,
            height=100,
            units="mm"
        )

    10 cm x 10 cm:

        write_rectangle_stl(
            width=10,
            height=10,
            units="cm"
        )

    0.1 m x 0.1 m:

        write_rectangle_stl(
            width=0.1,
            height=0.1,
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

    scale = unit_scale[units]

    width_m = width * scale
    height_m = height * scale

    vertices = np.array([
        [0.0,      0.0,       0.0],
        [width_m,  0.0,       0.0],
        [width_m,  height_m,  0.0],
        [0.0,      height_m,  0.0]
    ], dtype=float)

    triangles = [
        [0, 1, 2],
        [0, 2, 3]
    ]

    with open(filename, "w") as f:

        f.write("solid rectangle\n")

        for tri in triangles:

            p1 = vertices[tri[0]]
            p2 = vertices[tri[1]]
            p3 = vertices[tri[2]]

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

        f.write("endsolid rectangle\n")

    print(
        f"✔ Rectangle STL written: {filename}"
    )

    print(
        f"✔ Size: {width_m:.6f} m × "
        f"{height_m:.6f} m"
    )


if __name__ == "__main__":

    write_rectangle_stl(
        filename="rectangle.stl",
        width=100,
        height=100,
        units="mm"
    )
