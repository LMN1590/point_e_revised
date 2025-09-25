import numpy as np
from itertools import product
from pyquaternion import Quaternion

def generate_quaternion_grid(resolution=2):
    """
    Generate all possible quaternions from discretized Euler angles using pyquaternion.
    
    Parameters:
        resolution (int): number of discrete values per axis.
                          e.g. 5 means angles = [0, 90, 180, 270, 360).
    
    Returns:
        quaternions (np.ndarray): shape (resolution^3, 4), each row is (w, x, y, z).
    """
    # Create discretized angles in radians (exclude 360=2π to avoid duplicates)
    angles = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
    all_combos = product(angles, repeat=3)  # (roll, pitch, yaw)

    quaternions = []
    for combo in all_combos:
        q = Quaternion(axis=[1,0,0], angle=combo[0]) \
          * Quaternion(axis=[0,1,0], angle=combo[1]) \
          * Quaternion(axis=[0,0,1], angle=combo[2])
        quaternions.append([q.w, q.x, q.y, q.z])
    
    return np.array(quaternions)

# Example usage
quats = generate_quaternion_grid(2)
print("Number of quaternions:", quats.shape[0])
print("First 5 quaternions:\n", quats)
