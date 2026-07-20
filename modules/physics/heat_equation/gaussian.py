import numpy as np


def update_gaussian_source(heat_eq, source, t):

    t0 = float(source["t_start"])
    t1 = float(source["t_end"])

    if not (t0 <= t <= t1):

        heat_eq.Q.x.array[:] = 0.0

        if heat_eq.debug:
            print(f"   ❄ Source OFF (t={t:.3f}s)")

        return

    peak = float(source["value"])

    center = np.asarray(
        source["center"],
        dtype=float
    )

    radius = float(source["radius"])

    if radius <= 0.0:
        raise ValueError(
            "physics.source.radius must be > 0"
        )

    sigma = radius / 3.0

    def gaussian(x):

        dx = x[0] - center[0]
        dy = x[1] - center[1]
        dz = x[2] - center[2]

        r2 = dx * dx + dy * dy + dz * dz

        return peak * np.exp(
            -r2 / (2.0 * sigma * sigma)
        )

    heat_eq.Q.interpolate(gaussian)

    if heat_eq.debug:
        print(f"   🔥 Gaussian source ON (t={t:.3f}s)")
