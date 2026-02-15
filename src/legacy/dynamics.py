"""Legacy dynamics helpers retained for future validation and model cross-checks.

This module is not used by the active runtime path (main -> run).
"""

import numpy as np
from config import (
    LENGTH_FEMUR, LENGTH_TIBIA,
    MASS_FEMUR, MASS_TIBIA,
    COM_FEMUR, COM_TIBIA,
    INERTIA_FEMUR, INERTIA_TIBIA,
    GRAVITY
)

def theoretical_force(theta, theta_dot, theta_ddot):
    """Estimate knee force from a reduced-order 2-link dynamics formulation."""
    # Convert to the internal angle/sign convention used by the derived equations.
    theta = np.pi - theta
    theta_dot = -theta_dot
    theta_ddot = -theta_ddot

    D2 = LENGTH_FEMUR**2 + LENGTH_TIBIA**2 - 2 * LENGTH_FEMUR * LENGTH_TIBIA * np.cos(theta)
    D = np.sqrt(D2)

    k = (LENGTH_FEMUR * LENGTH_TIBIA * np.cos(theta) - LENGTH_TIBIA**2) / D2
    dk = (-LENGTH_FEMUR * LENGTH_TIBIA * np.sin(theta) * (1 + 2 * k)) / D2

    A = MASS_FEMUR * COM_FEMUR**2 + MASS_TIBIA * LENGTH_FEMUR**2 + INERTIA_FEMUR
    B = MASS_TIBIA * COM_TIBIA**2 + INERTIA_TIBIA
    E = -2 * MASS_TIBIA * LENGTH_FEMUR * COM_TIBIA

    M = A * k**2 + B * (k + 1)**2 + E * k * (k+1) * np.cos(theta)

    dM_dtheta = (2 * A * k + 2 * B * (k + 1) + E * (2 * k + 1) * np.cos(theta)) * dk - E * k * (k + 1) * np.sin(theta)

    C = 0.5 * dM_dtheta * theta_dot

    G = ((MASS_FEMUR * COM_FEMUR + MASS_TIBIA * LENGTH_FEMUR) * LENGTH_TIBIA * k + MASS_TIBIA * COM_TIBIA * LENGTH_FEMUR * (k + 1)) * GRAVITY * np.sin(theta) / D

    return (M * theta_ddot + C * theta_dot + G) / (- LENGTH_FEMUR * LENGTH_TIBIA * np.sin(theta) / D)
