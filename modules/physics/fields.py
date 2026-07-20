# modules/physics/fields.py

from dolfinx import fem


class PhysicalFields:
    """
    Container for all physical fields used in the simulation.
    """

    def __init__(self, k, rho, c, alpha):
        self.k = k
        self.rho = rho
        self.c = c
        self.alpha = alpha


def validate_material(material_cfg: dict):
    """
    Validate material parameters from YAML.
    """

    required = ["k", "rho", "c"]

    for key in required:
        if key not in material_cfg:
            raise ValueError(f"❌ Missing material.{key}")

        if material_cfg[key] <= 0:
            raise ValueError(f"❌ material.{key} must be > 0")

    return True


def create_uniform_fields(mesh, material_cfg: dict, thickness: float = None):
    """
    Create uniform physical fields as FEM constants.

    Args:
        mesh: FEniCSx mesh
        material_cfg: dict with k, rho, c
        thickness: optional thickness for surface physics

    Returns:
        PhysicalFields object
    """

    validate_material(material_cfg)

    k_val = material_cfg["k"]
    rho_val = material_cfg["rho"]
    c_val = material_cfg["c"]

    # Create constants
    k = fem.Constant(mesh, k_val)
    rho = fem.Constant(mesh, rho_val)
    c = fem.Constant(mesh, c_val)

    # Surface physics nuance
    if thickness is not None:
        alpha_val = k_val / (rho_val * c_val * thickness)
    else:
        alpha_val = k_val / (rho_val * c_val)

    alpha = fem.Constant(mesh, alpha_val)

    return PhysicalFields(k=k, rho=rho, c=c, alpha=alpha)


def build_fields(mesh, config: dict):
    """
    Main entry point for creating physical fields.

    Args:
        mesh: FEniCSx mesh
        config: full YAML config

    Returns:
        PhysicalFields
    """

    surface = config["domain"]

    if "material" not in surface:
        raise ValueError("❌ Missing surface.material")

    material_cfg = surface["material"]

    # Optional thickness (future-proof)
    thickness = surface.get("thickness", None)

    fields = create_uniform_fields(
        mesh=mesh,
        material_cfg=material_cfg,
        thickness=thickness
    )

    print("\n🧪 Physical fields created:")
    print(f"   k = {material_cfg['k']}")
    print(f"   rho = {material_cfg['rho']}")
    print(f"   c = {material_cfg['c']}")
    print(f"   alpha = {fields.alpha.value}")

    return fields
