import time
import math
from pathlib import Path
import mujoco, mujoco.viewer
from config import TOTAL_SIM_TIME, T_CYCLE
import kinematics
import dynamics
import xmlparser
import numpy as np

def sim(model_path, actuated=True, record_video=False, record_force=False):
    if not Path(model_path).exists():
        raise FileNotFoundError(model_path)

    model_string = xmlparser.parse(model_path)
    model = mujoco.MjModel.from_xml_string(model_string)
    data  = mujoco.MjData(model)

    if not actuated:
        mujoco.viewer.launch(model, data)
        return None, (None, None)

    else:
        if record_video:
            renderer = mujoco.Renderer(model, 640, 480)
            target_fps = 120
            tpf = np.floor(1/(target_fps * model.opt.timestep))
            frames, fps = [], 1/(tpf*model.opt.timestep)

        if record_force:
            logs = {
            "time": [], "gait": [],
            "knee_act": [], "knee_des": [],
            "F_meas": [], "F_theo": [],
            "theta_dot": [], "theta_ddot": []
            }

        with mujoco.viewer.launch_passive(model, data) as viewer:
            knee_0 = math.radians(kinematics.knee_angle_fourier(0))
            data.qpos[model.joint("knee_angle_l").qposadr[0]] = knee_0  # Set initial knee angle
            data.qpos[model.joint("shank_band_knee").qposadr[0]] = knee_0  # Set initial knee angle
            data.qpos[model.joint("hip_flexion_l").qposadr[0]] = kinematics.knee2hip(knee_0)  # Set initial knee angle
            data.qpos[model.joint("ankle_angle_l").qposadr[0]] = kinematics.knee2ankle(knee_0)  # Set initial ankle angle

            data.mocap_pos[0] = [0, 0, kinematics.knee2foot(knee_0) + 0.0028]

            clutch_id = model.actuator("clutch_spring").id
            data.ctrl[clutch_id] = math.radians(kinematics.knee_angle_fourier(0))
            clutch_gainprm = model.actuator_gainprm[clutch_id]
            clutch_biasprm = model.actuator_biasprm[clutch_id]
            while viewer.is_running() and data.time < TOTAL_SIM_TIME:
                t0 = time.time()

                mujoco.mj_step(model, data)

                # Desired target
                theta_des_rad = math.radians(kinematics.knee_angle_fourier(data.time))
                # data.ctrl[model.actuator("platform_act").id] = kinematics.knee2foot(theta_des_rad) + 0.0028
                data.mocap_pos[0] = [0, 0, kinematics.knee2foot(theta_des_rad) + 0.0028]


                if not hasattr(sim, "prev_engaged"):
                    sim.prev_engaged = None

                engaged = (data.time % T_CYCLE < T_CYCLE * 0.35)
                if engaged != sim.prev_engaged:
                    if engaged:
                        print("Engage")
                        model.actuator_gainprm[clutch_id] = clutch_gainprm
                        model.actuator_biasprm[clutch_id] = clutch_biasprm
                    else:
                        print("Disengage")
                        model.actuator_gainprm[clutch_id] = 0
                        model.actuator_biasprm[clutch_id] = 0
                    sim.prev_engaged = engaged

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
                    if int(data.time / model.opt.timestep) % tpf == 0:
                        renderer.update_scene(data, camera="fixed_cam")
                        frames.append(renderer.render().copy())

                # Calculate time to next step (0 if frame took longer than realtime)
                dt = model.opt.timestep - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)

        return logs if record_force else None, (frames, fps) if record_video else (None, None)