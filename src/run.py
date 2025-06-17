import time
import math
from pathlib import Path
import mujoco, mujoco.viewer
from config import TOTAL_SIM_TIME, T_CYCLE
import kinematics
import dynamics

def sim(model_path, record_video=False, record_force=False):
    if not Path(model_path).exists():
        raise FileNotFoundError(model_path)

    model = mujoco.MjModel.from_xml_path(model_path)
    data  = mujoco.MjData(model)

    if record_video:
        renderer = mujoco.Renderer(model, 640, 480)
        frames, fps = [], int(round(1 / model.opt.timestep))

    if record_force:
        logs = {
        "time": [], "gait": [],
        "knee_act": [], "knee_des": [],
        "F_meas": [], "F_theo": [],
        "theta_dot": [], "theta_ddot": []
        }

    with mujoco.viewer.launch_passive(model, data) as viewer:

        data.qpos[model.joint("knee_angle_l").qposadr[0]] = math.radians(kinematics.knee_angle_fourier(0))  # Set initial knee angle
        data.qpos[model.joint("shank_band_knee").qposadr[0]] = math.radians(kinematics.knee_angle_fourier(0))  # Set initial knee angle
        data.qpos[model.joint("hip_flexion_l").qposadr[0]] = math.radians(90 - (180 - kinematics.knee_angle_fourier(0))/2)  # Set initial knee angle

        while viewer.is_running() and data.time < TOTAL_SIM_TIME:
            t0 = time.time()

            mujoco.mj_step(model, data)

            # Desired target
            theta_des_rad = math.radians(kinematics.knee_angle_fourier(data.time))
            data.ctrl[model.actuator("platform_act").id] = kinematics.knee2foot(theta_des_rad)

            if record_force:
                joint = model.joint("knee_angle_l")
                theta_rad = data.qpos[joint.qposadr[0]]
                theta_dot = data.qvel[joint.dofadr[0]]
                theta_ddot = data.qacc[joint.dofadr[0]]

                F_th = dynamics.theoretical_force(theta_rad, theta_dot, theta_ddot)
                F_ms = -data.sensordata[2]

                # Log
                logs["time"].append(data.time)
                logs["gait"].append((data.time % T_CYCLE) / T_CYCLE)
                logs["knee_act"].append(math.degrees(theta_rad))
                logs["knee_des"].append(math.degrees(theta_des_rad))
                logs["F_meas"].append(F_ms)
                logs["F_theo"].append(F_th)
                logs["theta_dot"].append(theta_dot)
                logs["theta_ddot"].append(theta_ddot)

            # Display
            viewer.sync()

            # Update camera
            if record_video:
                renderer.update_scene(data, camera="fixed_cam")
                frames.append(renderer.render().copy())

            # Calculate time to next step (0 if frame took longer than realtime)
            dt = model.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

    return logs if record_force else None, (frames, fps) if record_video else (None, None)