from pathlib import Path
import numpy as np
import trimesh
import pymeshfix


def validate_stl(stl_path, config=None, verbose=True):
    """
    Validate an STL according to the geometry representation
    specified in the simulation config.

    Parameters
    ----------
    stl_path : str or Path
        Path to the STL file.

    config : dict, optional
        Simulation configuration dictionary.

        Expected structure:

            simulation:
              geometry:
                representation: volume

        Supported representations:
            - volume
            - surface

    verbose : bool
        If True, print diagnostic information.

    Returns
    -------
    bool
        True  -> STL passes validation.
        False -> STL is invalid for the requested representation.
    """

    stl_path = Path(stl_path)

    # ========================================================
    # Determine geometry representation
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

    # Validate representation value

    if representation not in ("surface", "volume"):

        if verbose:
            print(
                f"ERROR: Unsupported geometry representation: "
                f"{representation}"
            )

        return False

    require_volume = representation == "volume"

    # ========================================================
    # Check file
    # ========================================================

    if not stl_path.exists():

        if verbose:
            print(f"ERROR: STL does not exist: {stl_path}")

        return False

    # ========================================================
    # Load STL
    # ========================================================

    try:

        mesh = trimesh.load_mesh(
            stl_path,
            process=True
        )

    except Exception as e:

        if verbose:
            print(f"ERROR: Could not load STL: {e}")

        return False

    # ========================================================
    # Check mesh type
    # ========================================================

    if not isinstance(mesh, trimesh.Trimesh):

        if verbose:
            print(
                "ERROR: STL did not load as a triangular mesh."
            )

        return False

    # ========================================================
    # Basic information
    # ========================================================

    vertices = len(mesh.vertices)
    faces = len(mesh.faces)

    watertight = bool(mesh.is_watertight)
    volume = bool(mesh.is_volume)
    winding = bool(mesh.is_winding_consistent)

    # ========================================================
    # Duplicate faces
    # ========================================================

    faces_sorted = np.sort(
        mesh.faces,
        axis=1
    )

    _, counts = np.unique(
        faces_sorted,
        axis=0,
        return_counts=True
    )

    duplicate_faces = int(
        np.sum(counts > 1)
    )

    # ========================================================
    # Degenerate faces
    # ========================================================

    degenerate_faces = int(
        np.sum(mesh.area_faces <= 1e-12)
    )

    # ========================================================
    # Non-manifold edges
    # ========================================================

    edges = np.sort(
        mesh.edges_unique,
        axis=1
    )

    _, edge_counts = np.unique(
        edges,
        axis=0,
        return_counts=True
    )

    non_manifold_edges = int(
        np.sum(edge_counts > 2)
    )

    # ========================================================
    # Diagnostics
    # ========================================================

    if verbose:

        print("=" * 60)
        print("STL VALIDATION")
        print("=" * 60)

        print(f"File                 : {stl_path}")
        print(f"Representation       : {representation}")
        print(f"Requires volume      : {require_volume}")
        print(f"Vertices             : {vertices}")
        print(f"Faces                : {faces}")
        print(f"Watertight           : {watertight}")
        print(f"Volume               : {volume}")
        print(f"Winding consistent   : {winding}")
        print(f"Euler number         : {mesh.euler_number}")

        print()
        print(f"Duplicate faces      : {duplicate_faces}")
        print(f"Degenerate faces     : {degenerate_faces}")
        print(f"Non-manifold edges   : {non_manifold_edges}")

    # ========================================================
    # Determine validity
    # ========================================================

    valid = True

    # --------------------------------------------------------
    # Geometry must exist
    # --------------------------------------------------------

    if vertices == 0 or faces == 0:

        valid = False

        if verbose:
            print(
                "FAIL: Mesh contains no geometry."
            )

    # --------------------------------------------------------
    # Duplicate faces
    # --------------------------------------------------------

    if duplicate_faces > 0:

        valid = False

        if verbose:
            print(
                f"FAIL: {duplicate_faces} duplicate "
                "triangle(s) detected."
            )

    # --------------------------------------------------------
    # Degenerate faces
    # --------------------------------------------------------

    if degenerate_faces > 0:

        valid = False

        if verbose:
            print(
                f"FAIL: {degenerate_faces} degenerate "
                "triangle(s) detected."
            )

    # --------------------------------------------------------
    # Non-manifold geometry
    # --------------------------------------------------------

    if non_manifold_edges > 0:

        valid = False

        if verbose:
            print(
                f"FAIL: {non_manifold_edges} "
                "non-manifold edge(s) detected."
            )

    # --------------------------------------------------------
    # Winding
    # --------------------------------------------------------

    if not winding:

        valid = False

        if verbose:
            print(
                "FAIL: Triangle winding is inconsistent."
            )

    # ========================================================
    # VOLUME-SPECIFIC VALIDATION
    # ========================================================

    if require_volume:

        if not watertight:

            valid = False

            if verbose:
                print(
                    "FAIL: Volume representation requires "
                    "a watertight STL."
                )

        if not volume:

            valid = False

            if verbose:
                print(
                    "FAIL: STL is not recognized as a "
                    "valid closed volume."
                )

    # ========================================================
    # SURFACE-SPECIFIC VALIDATION
    # ========================================================

    else:

        if verbose:

            print()
            print(
                "INFO: Surface representation selected."
            )

            print(
                "      Watertight/volume checks are not required."
            )

    # ========================================================
    # Final result
    # ========================================================

    if verbose:

        print()
        print("=" * 60)

        if valid:
            print("RESULT: STL VALID")
        else:
            print("RESULT: STL INVALID")

        print("=" * 60)

    return valid
