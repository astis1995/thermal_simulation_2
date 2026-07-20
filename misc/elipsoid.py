# ellipsoid_stl.py

import numpy as np


def write_ellipsoid_stl(
    filename="ellipsoid.stl",
    major_axis=0.10,
    minor_axis=0.06,
    height=0.02,
    nx=100,
    ny=100
):
    """
    Generate the upper half of an ellipsoid and
    export it as an ASCII STL.

    major_axis : full length in x [m]
    minor_axis : full length in y [m]
    height     : maximum z [m]
    """

    A = major_axis / 2.0
    B = minor_axis / 2.0
    C = height

    x = np.linspace(-A, A, nx)
    y = np.linspace(-B, B, ny)

    X, Y = np.meshgrid(x, y)

    inside = (
        (X / A) ** 2
        + (Y / B) ** 2
    ) <= 1.0

    Z = np.full_like(
        X,
        np.nan,
        dtype=float
    )

    Z[inside] = (
        C
        * np.sqrt(
            1.0
            - (X[inside] / A) ** 2
            - (Y[inside] / B) ** 2
        )
    )

    with open(filename, "w") as f:

        f.write("solid ellipsoid\n")

        for j in range(ny - 1):
            for i in range(nx - 1):

                cells = [
                    (i, j),
                    (i + 1, j),
                    (i + 1, j + 1),
                    (i, j + 1)
                ]

                valid = True

                for ii, jj in cells:
                    if np.isnan(Z[jj, ii]):
                        valid = False
                        break

                if not valid:
                    continue

                p1 = np.array([
                    X[j, i],
                    Y[j, i],
                    Z[j, i]
                ])

                p2 = np.array([
                    X[j, i + 1],
                    Y[j, i + 1],
                    Z[j, i + 1]
                ])

                p3 = np.array([
                    X[j + 1, i + 1],
                    Y[j + 1, i + 1],
                    Z[j + 1, i + 1]
                ])

                p4 = np.array([
                    X[j + 1, i],
                    Y[j + 1, i],
                    Z[j + 1, i]
                ])

                triangles = [
                    (p1, p2, p3),
                    (p1, p3, p4)
                ]

                for a, b, c in triangles:

                    normal = np.cross(
                        b - a,
                        c - a
                    )

                    norm = np.linalg.norm(
                        normal
                    )

                    if norm > 0:
                        normal /= norm

                    f.write(
                        f"facet normal "
                        f"{normal[0]} "
                        f"{normal[1]} "
                        f"{normal[2]}\n"
                    )

                    f.write(" outer loop\n")

                    for p in (a, b, c):

                        f.write(
                            f"  vertex "
                            f"{p[0]} "
                            f"{p[1]} "
                            f"{p[2]}\n"
                        )

                    f.write(" endloop\n")
                    f.write("endfacet\n")

        f.write("endsolid ellipsoid\n")

    print(f"Created {filename}")

    print(
        f"Major axis : {major_axis} m"
    )

    print(
        f"Minor axis : {minor_axis} m"
    )

    print(
        f"Height     : {height} m"
    )


if __name__ == "__main__":

    write_ellipsoid_stl(
        filename="ellipsoid.stl",
        major_axis=0.10,  # 10 cm
        minor_axis=0.06,  # 6 cm
        height=0.02,      # 2 cm
        nx=30,
        ny=30
    )
