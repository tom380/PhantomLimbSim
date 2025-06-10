import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import scipy.io as sio


# Parameters
T_CYCLE = 2.0
TOTAL_SIM_TIME = 2 * T_CYCLE

# Force model constants
LENGTH_FEMUR    = 0.29869           # m
LENGTH_TIBIA    = 0.30433           # m
COM_FEMUR       = 0.1448932918      # m
COM_TIBIA       = 0.13535673954     # m
MASS_FEMUR      = 2.50513539        # kg
MASS_TIBIA      = 1.22768423        # kg
INERTIA_FEMUR   = 0.01452348        # kg·m²
INERTIA_TIBIA   = 0.00751915        # kg·m²
GRAVITY         = 9.81              # m/s²


def theoretical_force(theta, theta_dot, theta_ddot):
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


# Desired knee angle (deg)
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


def main():
    model_path = "phantom_barrutia.xml"
    if not Path(model_path).exists():
        raise FileNotFoundError(model_path)

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Video capture (optional)
    RECORD_VIDEO = False
    if RECORD_VIDEO:
        renderer = mujoco.Renderer(model, 640, 480)
        frames, fps = [], int(round(1 / model.opt.timestep))

    # Logs
    abs_time, gait_pct = [], []
    knee_act, knee_des = [], []
    F_meas, F_theo = [], []
    theta_dot_log, theta_ddot_log = [], []

    mujoco.viewer.launch(model, data)

    # with mujoco.viewer.launch_passive(model, data) as viewer:
    #     data.qpos[model.joint("knee_angle_l").qposadr[0]] = math.radians(knee_angle_fourier(0))  # Set initial knee angle
    #     data.qpos[model.joint("shank_band_knee").qposadr[0]] = math.radians(knee_angle_fourier(0))  # Set initial knee angle
    #     data.qpos[model.joint("hip_flexion_l").qposadr[0]] = math.radians(90 - (180 - knee_angle_fourier(0))/2)  # Set initial knee angle
    #     while viewer.is_running() and data.time < TOTAL_SIM_TIME:
    #         t0 = time.time()

    #         mujoco.mj_step(model, data)

    #         # Desired target
    #         theta_des_rad = math.radians(knee_angle_fourier(data.time))
    #         data.ctrl[model.actuator("platform_act").id] = knee2foot(theta_des_rad)

    #         joint = model.joint("knee_angle_l")
    #         theta_rad = data.qpos[joint.qposadr[0]]
    #         theta_dot = data.qvel[joint.dofadr[0]]
    #         theta_ddot = data.qacc[joint.dofadr[0]]

    #         F_th = theoretical_force(theta_rad, theta_dot, theta_ddot)
    #         F_ms = -data.sensordata[2]

    #         # Log
    #         abs_time.append(data.time)
    #         gait_pct.append((data.time % T_CYCLE) / T_CYCLE)
    #         knee_act.append(math.degrees(theta_rad))
    #         knee_des.append(math.degrees(theta_des_rad))
    #         F_meas.append(F_ms)
    #         F_theo.append(F_th)
    #         theta_dot_log.append(theta_dot)
    #         theta_ddot_log.append(theta_ddot)

    #         # Display
    #         viewer.sync()

    #         # Update camera
    #         if RECORD_VIDEO:
    #             renderer.update_scene(data, camera="fixed_cam")
    #             frames.append(renderer.render().copy())

    #         # Calculate time to next step (0 if frame took longer than realtime)
    #         dt = model.opt.timestep - (time.time() - t0)
    #         if dt > 0:
    #             time.sleep(dt)

    # # ---- Plot last cycle ----
    # abs_np = np.asarray(abs_time)
    # mask = abs_np >= (TOTAL_SIM_TIME - T_CYCLE)

    # gait = np.asarray(gait_pct)[mask]
    # order = np.argsort(gait)
    # gait = gait[order]

    # knee_act_last = np.asarray(knee_act)[mask][order]
    # knee_des_last = np.asarray(knee_des)[mask][order]
    # F_meas_last = np.asarray(F_meas)[mask][order]
    # F_theo_last = np.asarray(F_theo)[mask][order]

    # fig, ax1 = plt.subplots()
    # l1, = ax1.plot(gait, knee_act_last, label="Actual Knee")
    # l2, = ax1.plot(gait, knee_des_last, label="Desired Knee")
    # ax1.set_xlabel("Gait percentage (cycle‑normalized)")
    # ax1.set_ylabel("Knee angle (deg)")
    # ax1.set_xlim(0, 1)

    # ax2 = ax1.twinx()
    # l3, = ax2.plot(gait, F_meas_last, linestyle="--", label="Measured Force")
    # l4, = ax2.plot(gait, F_theo_last, linestyle="--", label="Theoretical Force")
    # ax2.set_ylabel("Force (N)")

    # ax1.legend(handles=[l1, l2, l3, l4], loc="upper center", ncol=4)
    # ax1.grid(True)
    # fig.tight_layout()
    # fig.savefig("simulation_results_last_cycle.png", dpi=300)
    # plt.show()

    # if RECORD_VIDEO:
    #     imageio.mimsave("run.mp4", frames, fps=fps, codec="libx264")
    #     print("Saved run.mp4")

    # print("Saved simulation_results_last_cycle.png")

    # # ---- Save full dataset to MATLAB .mat ----
    # data_dict = {
    #     "time": np.asarray(abs_time),
    #     "gait_pct": np.asarray(gait_pct),
    #     "knee_act": np.asarray(knee_act),
    #     "knee_des": np.asarray(knee_des),
    #     "F_meas": np.asarray(F_meas),
    #     "F_theo": np.asarray(F_theo),
    #     "theta_dot": np.asarray(theta_dot_log),
    #     "theta_ddot": np.asarray(theta_ddot_log),
    # }
    # sio.savemat("simulation_data.mat", data_dict)
    # print("Saved simulation_data.mat (MATLAB‑compatible)")


if __name__ == "__main__":
    main()
