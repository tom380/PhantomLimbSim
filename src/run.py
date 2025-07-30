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
            opt_scene = mujoco.MjvOption()
            opt_scene.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            target_fps = 120
            tpf = np.floor(1/(target_fps * model.opt.timestep))
            frames, fps = [], 1/(tpf*model.opt.timestep)

        if record_force:
            logs = {
            "time": [], "gait": [],
            "moment": [],
            "phantom_theta": [], "phantom_omega": [], "phantom_alpha": [],
            "exo_theta": [], "exo_omega": [], "exo_alpha": [],
            "spring_moment": [], "spring_length": []
            }

        with mujoco.viewer.launch_passive(model, data) as viewer:
            knee_0 = math.radians(kinematics.knee_angle_fourier(0))
            data.qpos[model.joint("knee_angle").qposadr[0]] = knee_0  # Set initial knee angle
            data.qpos[model.joint("shank_band_knee").qposadr[0]] = knee_0  # Set initial knee angle
            data.qpos[model.joint("hip_flexion").qposadr[0]] = kinematics.knee2hip(knee_0)  # Set initial knee angle
            data.qpos[model.joint("ankle_angle").qposadr[0]] = kinematics.knee2ankle(knee_0)  # Set initial ankle angle

            clutch_id = model.actuator("clutch_spring").id
            data.ctrl[clutch_id] = math.radians(kinematics.knee_angle_fourier(0))
            clutch_gainprm = model.actuator_gainprm[clutch_id].copy()
            clutch_biasprm = model.actuator_biasprm[clutch_id].copy()
            while viewer.is_running() and data.time < TOTAL_SIM_TIME:
                t0 = time.time()

                mujoco.mj_step(model, data)

                # Desired target
                theta_des_rad = math.radians(kinematics.knee_angle_fourier(data.time))
                data.ctrl[model.actuator("knee_actuator").id] = theta_des_rad


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
                    phantom_knee = model.joint("knee_angle")
                    exo_knee = model.joint("shank_band_knee")

                    torque = data.actuator_force[model.actuator("knee_actuator").id]
                    spring_torque = -data.actuator_force[model.actuator("clutch_spring").id]

                    exo_theta = data.qpos[exo_knee.qposadr[0]]
                    spring_length = (exo_theta - data.ctrl[clutch_id]) if engaged else 0

                    # Log
                    logs["time"].append(data.time)
                    logs["gait"].append((data.time % T_CYCLE) / T_CYCLE)
                    logs["moment"].append(torque)
                    logs["phantom_theta"].append(data.qpos[phantom_knee.qposadr[0]])
                    logs["phantom_omega"].append(data.qvel[phantom_knee.dofadr[0]])
                    logs["phantom_alpha"].append(data.qacc[phantom_knee.dofadr[0]])
                    logs["exo_theta"].append(exo_theta)
                    logs["exo_omega"].append(data.qvel[exo_knee.dofadr[0]])
                    logs["exo_alpha"].append(data.qacc[exo_knee.dofadr[0]])
                    logs["spring_moment"].append(spring_torque)
                    logs["spring_length"].append(spring_length)

                # Display
                viewer.sync()

                # Update camera
                if record_video:
                    if int(data.time / model.opt.timestep) % tpf == 0:
                        renderer.update_scene(data, camera="fixed_cam", scene_option=opt_scene)
                        frames.append(renderer.render().copy())

                # Calculate time to next step (0 if frame took longer than realtime)
                dt = model.opt.timestep - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)

        return logs if record_force else None, (frames, fps) if record_video else (None, None)