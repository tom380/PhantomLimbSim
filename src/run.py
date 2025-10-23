import time
import math
from pathlib import Path
import mujoco, mujoco.viewer
from config import TOTAL_SIM_TIME, T_CYCLE
import kinematics
import dynamics
import xmlparser
import numpy as np
from progress_display import SimulationProgress

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
            "exo_theta": [], "exo_omega": [], "exo_alpha": [], "exo_moment": [], "exo_constraint": [],
            "spring_moment": [], "spring_length": [],
            "moment_applied": [], "moment_bias": [], "moment_spring": [], "moment_damper": [], "moment_gravcomp": [], "moment_fluid": [], "moment_passive": [], "moment_actuator": [], "moment_smooth": [], "moment_constraint": [], "moment_inverse": [],
            "exo_applied": [], "exo_bias": [], "exo_spring": [], "exo_damper": [], "exo_gravcomp": [], "exo_fluid": [], "exo_passive": [], "exo_actuator": [], "exo_smooth": [], "exo_constraint": [], "exo_inverse": []
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
            progress = SimulationProgress(TOTAL_SIM_TIME, model.opt.timestep)

            def compute_gait_state(current_time: float):
                gait_phase = (current_time % T_CYCLE) / T_CYCLE if T_CYCLE else 0.0
                phase_label = "stance" if gait_phase < 0.6 else "swing"
                return gait_phase, phase_label

            progress.start()
            step_count = 0
            gait_phase, phase_label = compute_gait_state(data.time)
            progress.update(data.time, step_count, gait_phase, phase_label, None)
            last_step_duration = None
            while viewer.is_running() and data.time < TOTAL_SIM_TIME:
                iteration_start = time.perf_counter()

                mujoco.mj_step(model, data)
                step_count += 1

                # Desired target
                theta_des_rad = math.radians(kinematics.knee_angle_fourier(data.time))
                data.ctrl[model.actuator("knee_actuator").id] = theta_des_rad


                if not hasattr(sim, "prev_engaged"):
                    sim.prev_engaged = None

                engaged = (data.time % T_CYCLE < T_CYCLE * 0.35)
                if engaged != sim.prev_engaged:
                    if engaged:
                        model.actuator_gainprm[clutch_id] = clutch_gainprm
                        model.actuator_biasprm[clutch_id] = clutch_biasprm
                    else:
                        model.actuator_gainprm[clutch_id] = 0
                        model.actuator_biasprm[clutch_id] = 0
                    sim.prev_engaged = engaged

                if record_force:
                    phantom_knee = model.joint("knee_angle")
                    exo_knee = model.joint("shank_band_knee")

                    torque = data.actuator_force[model.actuator("knee_actuator").id]
                    # Control
                    torque_applied = data.qfrc_applied[phantom_knee.dofadr[0]] # Applied generalized force
                    # Computed by mj_fwdVelocity/mj_rne (without acceleration)
                    torque_bias = data.qfrc_bias[phantom_knee.dofadr[0]] # C(qpos,qvel)
                    # Computed by mj_fwdVelocity/mj_passive
                    torque_spring = data.qfrc_spring[phantom_knee.dofadr[0]] # Passive spring force
                    torque_damper = data.qfrc_damper[phantom_knee.dofadr[0]] # Passive damper force
                    torque_gravcomp = data.qfrc_gravcomp[phantom_knee.dofadr[0]] # Passive gravity compensation force
                    torque_fluid = data.qfrc_fluid[phantom_knee.dofadr[0]] # Passive fluid force
                    torque_passive = data.qfrc_passive[phantom_knee.dofadr[0]] # Total passive force
                    # Computed by mj_fwdActuation
                    torque_actuator = data.qfrc_actuator[phantom_knee.dofadr[0]] # Actuator force
                    # Computed by mj_fwdAcceleration
                    torque_smooth = data.qfrc_smooth[phantom_knee.dofadr[0]] # Net unconstrained force
                    # Computed by mj_fwdConstraint/mj_inverse
                    torque_constraint = data.qfrc_constraint[phantom_knee.dofadr[0]] # Constraint force
                    # Computed by mj_inverse
                    torque_inverse = data.qfrc_inverse[phantom_knee.dofadr[0]] # Net external force; should equal: qfrc_applied + J'*xfrc_applied + qfrc_actuator


                    exo_applied = data.qfrc_applied[exo_knee.dofadr[0]]
                    exo_bias = data.qfrc_bias[exo_knee.dofadr[0]]
                    exo_spring = data.qfrc_spring[exo_knee.dofadr[0]]
                    exo_damper = data.qfrc_damper[exo_knee.dofadr[0]]
                    exo_gravcomp = data.qfrc_gravcomp[exo_knee.dofadr[0]]
                    exo_fluid = data.qfrc_fluid[exo_knee.dofadr[0]]
                    exo_passive = data.qfrc_passive[exo_knee.dofadr[0]]
                    exo_actuator = data.qfrc_actuator[exo_knee.dofadr[0]]
                    exo_smooth = data.qfrc_smooth[exo_knee.dofadr[0]]
                    exo_constraint = data.qfrc_constraint[exo_knee.dofadr[0]]
                    exo_inverse = data.qfrc_inverse[exo_knee.dofadr[0]]


                    spring_torque = -data.actuator_force[model.actuator("clutch_spring").id]
                    i = exo_knee.dofadr[0]
                    exo_torque = data.qfrc_constraint[exo_knee.dofadr[0]]

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
                    logs["exo_moment"].append(exo_torque)
                    logs["spring_moment"].append(spring_torque)
                    logs["spring_length"].append(spring_length)
                    logs["moment_applied"].append(torque_applied)
                    logs["moment_bias"].append(torque_bias)
                    logs["moment_spring"].append(torque_spring)
                    logs["moment_damper"].append(torque_damper)
                    logs["moment_gravcomp"].append(torque_gravcomp)
                    logs["moment_fluid"].append(torque_fluid)
                    logs["moment_passive"].append(torque_passive)
                    logs["moment_actuator"].append(torque_actuator)
                    logs["moment_smooth"].append(torque_smooth)
                    logs["moment_constraint"].append(torque_constraint)
                    logs["moment_inverse"].append(torque_inverse)
                    logs["exo_applied"].append(exo_applied)
                    logs["exo_bias"].append(exo_bias)
                    logs["exo_spring"].append(exo_spring)
                    logs["exo_damper"].append(exo_damper)
                    logs["exo_gravcomp"].append(exo_gravcomp)
                    logs["exo_fluid"].append(exo_fluid)
                    logs["exo_passive"].append(exo_passive)
                    logs["exo_actuator"].append(exo_actuator)
                    logs["exo_smooth"].append(exo_smooth)
                    logs["exo_constraint"].append(exo_constraint)
                    logs["exo_inverse"].append(exo_inverse)


                # Display
                viewer.sync()
                last_step_duration = time.perf_counter() - iteration_start
                gait_phase, phase_label = compute_gait_state(data.time)
                progress.update(data.time, step_count, gait_phase, phase_label, last_step_duration)

                # Update camera
                if record_video:
                    if int(data.time / model.opt.timestep) % tpf == 0:
                        renderer.update_scene(data, camera="fixed_cam", scene_option=opt_scene)
                        frames.append(renderer.render().copy())

                # Calculate time to next step (0 if frame took longer than realtime)
                loop_duration = time.perf_counter() - iteration_start
                dt = model.opt.timestep - loop_duration
                if dt > 0:
                    time.sleep(dt)
            gait_phase, phase_label = compute_gait_state(data.time)
            progress.finish(data.time, step_count, gait_phase, phase_label, last_step_duration)

        return logs if record_force else None, (frames, fps) if record_video else (None, None)
