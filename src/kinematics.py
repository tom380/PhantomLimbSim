import math, numpy as np
from config import T_CYCLE, LENGTH_FEMUR, LENGTH_TIBIA

def knee_angle_fourier(t: float) -> float:
    gait = t / T_CYCLE
    theta = (
        45
        + 10 * np.sin(2 * np.pi * (gait + 0.03) + np.pi - 0.1)
        + 45
        + 9.7 * np.sin(4 * np.pi * (gait + 0.04) + np.pi + 1.6)
        - 49.9
    )
    return theta


def knee2foot(theta_rad: float) -> float:
    return LENGTH_FEMUR + LENGTH_TIBIA - math.sqrt(
        LENGTH_FEMUR ** 2 + LENGTH_TIBIA ** 2 - 2 * LENGTH_FEMUR * LENGTH_TIBIA * math.cos(math.pi - theta_rad - 0.06) # Add 0.06 rad offset to better match knee angle to profile
    )