# mesh_3d_100x100x100.py

import numpy as np

def generar_malla_3d(nx=100, ny=100, nz=100,
                     largo=100.0, ancho=100.0, alto=100.0):

    x = np.linspace(0, largo, nx)
    y = np.linspace(0, ancho, ny)
    z = np.linspace(0, alto, nz)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    return X, Y, Z


if __name__ == "__main__":
    X, Y, Z = generar_malla_3d()

    nodos = np.column_stack([
        X.ravel(),
        Y.ravel(),
        Z.ravel()
    ])

    np.savetxt(
        "mesh_3d_100x100x100.csv",
        nodos,
        delimiter=",",
        header="x,y,z",
        comments=""
    )

    print("Malla guardada en mesh_3d_100x100x100.csv")
    print(f"Nodos: {len(nodos)}")
