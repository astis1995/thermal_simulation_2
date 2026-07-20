# mesh_100x100.py

import numpy as np

def generar_malla(nx=100, ny=100, ancho=100.0, alto=100.0):
    x = np.linspace(0, ancho, nx)
    y = np.linspace(0, alto, ny)

    X, Y = np.meshgrid(x, y)

    return X, Y

if __name__ == "__main__":
    X, Y = generar_malla()

    # Convertir la malla a lista de nodos
    nodos = np.column_stack((X.ravel(), Y.ravel()))

    # Guardar archivo
    np.savetxt(
        "mesh_100x100.csv",
        nodos,
        delimiter=",",
        header="x,y",
        comments=""
    )

    print(f"Malla guardada en mesh_100x100.csv")
    print(f"Número de nodos: {len(nodos)}")
