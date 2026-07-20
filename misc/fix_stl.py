import os
import trimesh


def fix_stl(stl_path):
    """
    Diagnose STL issues, keep the largest connected component,
    perform basic repairs, and save as *_fixed.stl.
    """

    if not os.path.isfile(stl_path):
        raise FileNotFoundError(f"File not found: {stl_path}")

    print(f"Loading: {stl_path}")

    mesh = trimesh.load_mesh(stl_path)

    print("\n=== ORIGINAL MESH ===")
    print(f"Watertight : {mesh.is_watertight}")
    print(f"Vertices   : {len(mesh.vertices)}")
    print(f"Faces      : {len(mesh.faces)}")
    print(f"Euler Num. : {mesh.euler_number}")

    try:
        print(f"Bodies     : {mesh.body_count}")
    except Exception:
        print("Bodies     : Unknown")

    try:
        print(f"Volume     : {mesh.volume}")
    except Exception:
        print("Volume     : Unknown")

    print("\n=== COMPONENT ANALYSIS ===")

    components = mesh.split(only_watertight=False)

    print(f"Connected components: {len(components)}")

    for i, comp in enumerate(components):
        print(
            f"[{i}] "
            f"Faces={len(comp.faces):8d} "
            f"Vertices={len(comp.vertices):8d} "
            f"Watertight={comp.is_watertight}"
        )

    if len(components) > 1:
        largest = max(components, key=lambda m: len(m.faces))

        print(
            f"\nKeeping largest component "
            f"({len(largest.faces)} faces)"
        )

        mesh = largest

    print("\n=== REPAIRING ===")

    # Remove bad geometry
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    try:
        mesh.remove_degenerate_faces()
    except Exception:
        pass

    try:
        mesh.remove_duplicate_faces()
    except Exception:
        pass

    try:
        mesh.remove_infinite_values()
    except Exception:
        pass

    # Fix orientation
    try:
        trimesh.repair.fix_normals(mesh)
    except Exception:
        pass

    # Fill small holes
    try:
        trimesh.repair.fill_holes(mesh)
    except Exception:
        pass

    try:
        mesh.process(validate=True)
    except Exception:
        pass

    print("\n=== FINAL MESH ===")
    print(f"Watertight : {mesh.is_watertight}")
    print(f"Vertices   : {len(mesh.vertices)}")
    print(f"Faces      : {len(mesh.faces)}")
    print(f"Euler Num. : {mesh.euler_number}")

    try:
        print(f"Volume     : {mesh.volume}")
    except Exception:
        pass

    try:
        print(f"Bodies     : {mesh.body_count}")
    except Exception:
        pass

    folder = os.path.dirname(stl_path)
    filename = os.path.basename(stl_path)
    name, ext = os.path.splitext(filename)

    output_path = os.path.join(
        folder,
        f"{name}_fixed{ext}"
    )

    mesh.export(output_path)

    print("\nSaved repaired STL:")
    print(output_path)

    return output_path


if __name__ == "__main__":
    stl_path = input("Enter STL path: ").strip()
    fix_stl(stl_path)
