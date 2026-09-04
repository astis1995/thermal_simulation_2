import os
from pathlib import Path

import trimesh


def fix_stl(
    stl_path,
    config=None,
    output_path=None,
    verbose=True
):
    """
    Diagnose and repair an STL according to the simulation
    geometry representation specified in config.

    Parameters
    ----------
    stl_path : str or Path
        Input STL file.

    config : dict, optional
        Simulation configuration dictionary.

        Expected structure:

            simulation:
              geometry:
                representation: volume

        Supported representations:
            - volume
            - surface

    output_path : str or Path, optional
        Output STL path.
        If None, saves as *_fixed.stl.

    verbose : bool
        Print diagnostic information.

    Returns
    -------
    Path
        Path to repaired STL.
    """

    # ========================================================
    # PATH
    # ========================================================

    stl_path = Path(stl_path)

    if not stl_path.is_file():
        raise FileNotFoundError(
            f"File not found: {stl_path}"
        )

    # ========================================================
    # READ GEOMETRY REPRESENTATION FROM CONFIG
    # ========================================================

    representation = "surface"

    if config is not None:

        representation = (
            config
            .get("simulation", {})
            .get("geometry", {})
            .get("representation", "surface")
            .lower()
        )

    if representation not in ("surface", "volume"):

        raise ValueError(
            f"Unsupported geometry representation: "
            f"{representation}"
        )

    require_volume = representation == "volume"

    # ========================================================
    # LOAD STL
    # ========================================================

    if verbose:

        print(f"Loading: {stl_path}")

        print()
        print("=" * 60)
        print("STL REPAIR")
        print("=" * 60)

        print(
            f"Representation : {representation}"
        )

        print(
            f"Requires volume: {require_volume}"
        )

    try:

        mesh = trimesh.load_mesh(
            stl_path,
            process=True
        )

    except Exception as e:

        raise RuntimeError(
            f"Could not load STL: {e}"
        ) from e

    if not isinstance(mesh, trimesh.Trimesh):

        raise ValueError(
            "STL did not load as a triangular mesh."
        )

    # ========================================================
    # ORIGINAL MESH
    # ========================================================

    if verbose:

        print("\n=== ORIGINAL MESH ===")

        print(
            f"Watertight : {mesh.is_watertight}"
        )

        print(
            f"Vertices   : {len(mesh.vertices)}"
        )

        print(
            f"Faces      : {len(mesh.faces)}"
        )

        print(
            f"Euler Num. : {mesh.euler_number}"
        )

        print(
            f"Winding    : "
            f"{mesh.is_winding_consistent}"
        )

        try:

            print(
                f"Bodies     : {mesh.body_count}"
            )

        except Exception:

            print(
                "Bodies     : Unknown"
            )

        try:

            print(
                f"Volume     : {mesh.volume}"
            )

        except Exception:

            print(
                "Volume     : Unknown"
            )

    # ========================================================
    # COMPONENT ANALYSIS
    # ========================================================

    components = mesh.split(
        only_watertight=False
    )

    if verbose:

        print("\n=== COMPONENT ANALYSIS ===")

        print(
            f"Connected components: "
            f"{len(components)}"
        )

        for i, comp in enumerate(components):

            print(
                f"[{i}] "
                f"Faces={len(comp.faces):8d} "
                f"Vertices={len(comp.vertices):8d} "
                f"Watertight={comp.is_watertight}"
            )

    # ========================================================
    # KEEP LARGEST COMPONENT
    # ========================================================

    if len(components) > 1:

        largest = max(
            components,
            key=lambda m: len(m.faces)
        )

        if verbose:

            print(
                f"\nKeeping largest component "
                f"({len(largest.faces)} faces)"
            )

        mesh = largest

    # ========================================================
    # REPAIR
    # ========================================================

    if verbose:

        print("\n=== REPAIRING ===")

    # --------------------------------------------------------
    # Remove unreferenced vertices
    # --------------------------------------------------------

    try:

        mesh.remove_unreferenced_vertices()

    except Exception as e:

        if verbose:
            print(
                f"Warning: remove_unreferenced_vertices: {e}"
            )

    # --------------------------------------------------------
    # Remove degenerate faces
    # --------------------------------------------------------

    try:

        mesh.remove_degenerate_faces()

    except Exception as e:

        if verbose:
            print(
                f"Warning: remove_degenerate_faces: {e}"
            )

    # --------------------------------------------------------
    # Remove duplicate faces
    # --------------------------------------------------------

    try:

        mesh.remove_duplicate_faces()

    except Exception as e:

        if verbose:
            print(
                f"Warning: remove_duplicate_faces: {e}"
            )

    # --------------------------------------------------------
    # Remove infinite values
    # --------------------------------------------------------

    try:

        mesh.remove_infinite_values()

    except Exception as e:

        if verbose:
            print(
                f"Warning: remove_infinite_values: {e}"
            )

    # --------------------------------------------------------
    # Fix orientation
    # --------------------------------------------------------

    try:

        trimesh.repair.fix_normals(mesh)

    except Exception as e:

        if verbose:
            print(
                f"Warning: fix_normals: {e}"
            )

    # ========================================================
    # VOLUME-SPECIFIC REPAIR
    # ========================================================

    if require_volume:

        if verbose:

            print(
                "\nVolume representation detected."
            )

            print(
                "Attempting to close surface holes..."
            )

        try:

            trimesh.repair.fill_holes(mesh)

        except Exception as e:

            if verbose:
                print(
                    f"Warning: fill_holes: {e}"
                )

    # ========================================================
    # FINAL PROCESSING
    # ========================================================

    try:

        mesh.process(
            validate=True
        )

    except Exception as e:

        if verbose:
            print(
                f"Warning: mesh.process: {e}"
            )

    # ========================================================
    # FINAL MESH
    # ========================================================

    if verbose:

        print("\n=== FINAL MESH ===")

        print(
            f"Watertight : {mesh.is_watertight}"
        )

        print(
            f"Vertices   : {len(mesh.vertices)}"
        )

        print(
            f"Faces      : {len(mesh.faces)}"
        )

        print(
            f"Euler Num. : {mesh.euler_number}"
        )

        print(
            f"Winding    : "
            f"{mesh.is_winding_consistent}"
        )

        try:

            print(
                f"Volume     : {mesh.volume}"
            )

        except Exception:

            print(
                "Volume     : Unknown"
            )

        try:

            print(
                f"Bodies     : {mesh.body_count}"
            )

        except Exception:

            print(
                "Bodies     : Unknown"
            )

    # ========================================================
    # OUTPUT
    # ========================================================

    if output_path is None:

        output_path = stl_path.with_name(
            f"{stl_path.stem}_fixed{stl_path.suffix}"
        )

    else:

        output_path = Path(output_path)

    mesh.export(output_path)

    if verbose:

        print("\nSaved repaired STL:")
        print(output_path)

    return output_path


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    stl_path = input(
        "Enter STL path: "
    ).strip()

    fix_stl(
        stl_path=stl_path
    )
