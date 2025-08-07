import numpy as np

def compute_lame_parameters(E: float, nu: float):
    # Compute Lame parameters (mu, lambda) from Young's modulus (E) and Poisson's ratio (nu)
    mu = E / (2 * (1 + nu))
    lambd = E * nu / ((1 + nu) * (1 - 2 * nu))
    return mu, lambd

def directions_to_spherical(vectors: np.ndarray) -> np.ndarray:
    """
    Convert Nx3 unit vectors to spherical coordinates (azimuth, inclination).

    Parameters:
        vectors: A numpy array of shape (N, 3), each row a unit vector.

    Returns:
        spherical_coords: A numpy array of shape (N, 2), where each row is (azimuth, inclination).
                          azimuth ∈ [0, 2π): angle in x-y plane from the x-axis (around the z-axis)
                          inclination ∈ [0, π]: angle from the z-axis
    """
    x = vectors[:, 0]
    y = vectors[:, 1]
    z = vectors[:, 2]

    azimuth = np.arctan2(y, x) % (2 * np.pi)          # azimuth in [0, 2π)
    inclination = np.arccos(np.clip(z, -1.0, 1.0))    # inclination in [0, π]

    return np.stack((azimuth, inclination), axis=1)  # shape: (N, 2)