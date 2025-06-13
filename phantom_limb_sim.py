import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import scipy.io as sio

# Simulation parameters
T_CYCLE = 2.0
TOTAL_SIM_TIME = 2 * T_CYCLE

# Force model constants
LENGTH_FEMUR = 0.29869
LENGTH_TIBIA = 0.30433
COM_FEMUR = 0.1448932918
COM_TIBIA = 0.13535673954
MASS_FEMUR = 2.50513539
MASS_TIBIA = 1.22768423
INERTIA_FEMUR = 0.01452348
INERTIA_TIBIA = 0.00751915
GRAVITY = 9.81

@dataclass
class SimConfig:
    record_video: bool = False
    log_data: bool = False
    actuate_platform: bool = True
    passive: bool = False


def theoretical_force(theta, theta_dot, theta_ddot):
    theta = np.pi - theta
    theta_dot = -theta_dot
    theta_ddot = -theta_ddot
    d2 = LENGTH_FEMUR**2 + LENGTH_TIBIA**2 - 2 * LENGTH_FEMUR * LENGTH_TIBIA * np.cos(theta)
    d = np.sqrt(d2)
    k = (LENGTH_FEMUR * LENGTH_TIBIA * np.cos(theta) - LENGTH_TIBIA**2) / d2
    dk = (-LENGTH_FEMUR * LENGTH_TIBIA * np.sin(theta) * (1 + 2 * k)) / d2
    a = MASS_FEMUR * COM_FEMUR**2 + MASS_TIBIA * LENGTH_FEMUR**2 + INERTIA_FEMUR
    b = MASS_TIBIA * COM_TIBIA**2 + INERTIA_TIBIA
    e = -2 * MASS_TIBIA * LENGTH_FEMUR * COM_TIBIA
    m = a * k**2 + b * (k + 1)**2 + e * k * (k + 1) * np.cos(theta)
    dm_dtheta = (2 * a * k + 2 * b * (k + 1) + e * (2 * k + 1) * np.cos(theta)) * dk - e * k * (k + 1) * np.sin(theta)
    c = 0.5 * dm_dtheta * theta_dot
    g = ((MASS_FEMUR * COM_FEMUR + MASS_TIBIA * LENGTH_FEMUR) * LENGTH_TIBIA * k + MASS_TIBIA * COM_TIBIA * LENGTH_FEMUR * (k + 1)) * GRAVITY * np.sin(theta) / d
    return (m * theta_ddot + c * theta_dot + g) / (-LENGTH_FEMUR * LENGTH_TIBIA * np.sin(theta) / d)


def knee_angle_fourier(t: float) -> float:
    gait = t / T_CYCLE
    theta = 45 + 10 * np.sin(2 * np.pi * (gait + 0.03) + np.pi - 0.1)
    theta += 45 + 9.7 * np.sin(4 * np.pi * (gait + 0.04) + np.pi + 1.6)
    theta -= 49.9
    return theta


def knee2foot(theta_rad: float) -> float:
    return LENGTH_FEMUR + LENGTH_TIBIA - math.sqrt(
        LENGTH_FEMUR ** 2 + LENGTH_TIBIA ** 2 - 2 * LENGTH_FEMUR * LENGTH_TIBIA * math.cos(math.pi - theta_rad - 0.06)
    )


def run_simulation(model_path: str, cfg: SimConfig):
    if not Path(model_path).exists():
        raise FileNotFoundError(model_path)
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    viewer_fn = mujoco.viewer.launch_passive if cfg.passive else mujoco.viewer.launch
    renderer = None
    frames = []
    fps = int(round(1 / model.opt.timestep))
    if cfg.record_video:
        renderer = mujoco.Renderer(model, 640, 480)

    logs = {
        "time": [],
        "gait_pct": [],
        "knee_act": [],
        "knee_des": [],
        "F_meas": [],
        "F_theo": [],
        "theta_dot": [],
        "theta_ddot": [],
    }

    with viewer_fn(model, data) as viewer:
        data.qpos[model.joint("knee_angle_l").qposadr[0]] = math.radians(knee_angle_fourier(0))
        data.qpos[model.joint("shank_band_knee").qposadr[0]] = math.radians(knee_angle_fourier(0))
        data.qpos[model.joint("hip_flexion_l").qposadr[0]] = math.radians(90 - (180 - knee_angle_fourier(0)) / 2)

        while viewer.is_running() and data.time < TOTAL_SIM_TIME:
            t0 = time.time()
            mujoco.mj_step(model, data)

            if cfg.actuate_platform:
                theta_des_rad = math.radians(knee_angle_fourier(data.time))
                data.ctrl[model.actuator("platform_act").id] = knee2foot(theta_des_rad)
            else:
                theta_des_rad = math.radians(knee_angle_fourier(data.time))

            joint = model.joint("knee_angle_l")
            theta_rad = data.qpos[joint.qposadr[0]]
            theta_dot = data.qvel[joint.dofadr[0]]
            theta_ddot = data.qacc[joint.dofadr[0]]

            if cfg.log_data:
                logs["time"].append(data.time)
                logs["gait_pct"].append((data.time % T_CYCLE) / T_CYCLE)
                logs["knee_act"].append(math.degrees(theta_rad))
                logs["knee_des"].append(math.degrees(theta_des_rad))
                logs["F_meas"].append(-data.sensordata[2])
                logs["F_theo"].append(theoretical_force(theta_rad, theta_dot, theta_ddot))
                logs["theta_dot"].append(theta_dot)
                logs["theta_ddot"].append(theta_ddot)

            if cfg.record_video:
                renderer.update_scene(data, camera="fixed_cam")
                frames.append(renderer.render().copy())

            viewer.sync()
            dt = model.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

    if cfg.record_video and frames:
        imageio.mimsave("run.mp4", frames, fps=fps, codec="libx264")
        print("Saved run.mp4")

    if cfg.log_data and logs["time"]:
        abs_np = np.asarray(logs["time"])
        mask = abs_np >= (TOTAL_SIM_TIME - T_CYCLE)
        gait = np.asarray(logs["gait_pct"])[mask]
        order = np.argsort(gait)
        gait = gait[order]
        knee_act_last = np.asarray(logs["knee_act"])[mask][order]
        knee_des_last = np.asarray(logs["knee_des"])[mask][order]
        F_meas_last = np.asarray(logs["F_meas"])[mask][order]
        F_theo_last = np.asarray(logs["F_theo"])[mask][order]
        fig, ax1 = plt.subplots()
        l1, = ax1.plot(gait, knee_act_last, label="Actual Knee")
        l2, = ax1.plot(gait, knee_des_last, label="Desired Knee")
        ax1.set_xlabel("Gait percentage (cycle-normalized)")
        ax1.set_ylabel("Knee angle (deg)")
        ax1.set_xlim(0, 1)
        ax2 = ax1.twinx()
        l3, = ax2.plot(gait, F_meas_last, linestyle="--", label="Measured Force")
        l4, = ax2.plot(gait, F_theo_last, linestyle="--", label="Theoretical Force")
        ax2.set_ylabel("Force (N)")
        ax1.legend(handles=[l1, l2, l3, l4], loc="upper center", ncol=4)
        ax1.grid(True)
        fig.tight_layout()
        fig.savefig("simulation_results_last_cycle.png", dpi=300)
        plt.close(fig)

        sio.savemat("simulation_data.mat", {k: np.asarray(v) for k, v in logs.items()})
        print("Saved simulation_results_last_cycle.png and simulation_data.mat")


def parse_args() -> SimConfig:
    parser = argparse.ArgumentParser(description="Phantom limb simulation")
    parser.add_argument("--model", default="phantom_barrutia.xml", help="MJCF model to load")
    parser.add_argument("--video", action="store_true", help="record video")
    parser.add_argument("--log", action="store_true", help="log data and plot results")
    parser.add_argument("--no-actuation", action="store_true", help="do not actuate the platform")
    parser.add_argument("--passive", action="store_true", help="use passive viewer")
    args = parser.parse_args()
    return args.model, SimConfig(record_video=args.video, log_data=args.log, actuate_platform=not args.no_actuation, passive=args.passive)


if __name__ == "__main__":
    model_path, cfg = parse_args()
    run_simulation(model_path, cfg)
