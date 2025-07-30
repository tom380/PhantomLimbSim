import math, numpy as np
from config import T_CYCLE, LENGTH_FEMUR, LENGTH_TIBIA
from scipy.interpolate import CubicSpline
import scipy.io

def knee_angle_fourier(t: float) -> float:
    if not hasattr(knee_angle_fourier, "angles"):
        mat = scipy.io.loadmat("src/knee_angles.mat")

        knee_angle_fourier.angles = np.squeeze(mat['angles'])
        knee_angle_fourier.n = len(knee_angle_fourier.angles)

        x = np.linspace(0, 1, knee_angle_fourier.n, endpoint=False)
        knee_angle_fourier.spline = CubicSpline(x, knee_angle_fourier.angles, bc_type='periodic')

    gait = (t / T_CYCLE) % 1.0
    return float(knee_angle_fourier.spline(gait))

def D(theta_rad: float) -> float:
    theta = np.pi - theta_rad
    return np.sqrt(LENGTH_FEMUR ** 2 + LENGTH_TIBIA ** 2 - 2 * LENGTH_FEMUR * LENGTH_TIBIA * np.cos(theta))

def knee2foot(theta_rad: float) -> float:
    return LENGTH_FEMUR + LENGTH_TIBIA - D(theta_rad)

def knee2hip(theta_rad: float) -> float:
    return -np.asin(LENGTH_TIBIA * np.sin(np.pi - theta_rad) / D(theta_rad))

def knee2ankle(theta_rad: float) -> float:
    return -np.asin(LENGTH_FEMUR * np.sin(np.pi - theta_rad) / D(theta_rad))