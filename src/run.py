import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from config import TOTAL_SIM_TIME
import kinematics
import xmlparser
from progress_display import SimulationProgress


def sim(
    model_path,
    actuated=True,
    record_video=False,
    record_force=False,
    record_flex_contact=False,
    kinematics_mode="average",
    spring_index=3,
    sim_time=None,
):
    if not Path(model_path).exists():
        raise FileNotFoundError(model_path)

    model_string = xmlparser.parse(model_path)
    model = mujoco.MjModel.from_xml_string(model_string)
    data = mujoco.MjData(model)

    profile = kinematics.get_kinematics_profile(kinematics_mode, spring_index)

    default_time = profile.dataset_duration if profile.mode == "full" else TOTAL_SIM_TIME
    target_time = default_time

    if sim_time is not None:
        if sim_time <= 0:
            target_time = profile.dataset_duration if profile.dataset_duration is not None else default_time
        else:
            target_time = sim_time

    if profile.dataset_duration is not None:
        dataset_time = profile.dataset_duration
        if dataset_time is None or dataset_time <= 0:
            dataset_time = 0.0
        if target_time is None or target_time <= 0:
            target_time = dataset_time
        else:
            target_time = min(target_time, dataset_time) if dataset_time > 0 else target_time

    if target_time is None or target_time <= 0:
        target_time = TOTAL_SIM_TIME

    target_time = float(target_time) if target_time is not None else None

    joint_logs = None
    flex_logs = None

    if not actuated:
        mujoco.viewer.launch(model, data)
        return (None, None), (None, None)

    else:
        if record_video:
            renderer = mujoco.Renderer(model, 640, 480)
            opt_scene = mujoco.MjvOption()
            opt_scene.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            target_fps = 120
            tpf = np.floor(1/(target_fps * model.opt.timestep))
            frames, fps = [], 1/(tpf*model.opt.timestep)

        if record_force:
            joint_logs = {
            "time": [], "gait": [],
            "moment": [],
            "phantom_theta": [], "phantom_omega": [], "phantom_alpha": [],
            "exo_theta": [], "exo_omega": [], "exo_alpha": [], "exo_moment": [], "exo_constraint": [],
            "spring_moment": [], "spring_length": [],
            "moment_applied": [], "moment_bias": [], "moment_spring": [], "moment_damper": [], "moment_gravcomp": [], "moment_fluid": [], "moment_passive": [], "moment_actuator": [], "moment_smooth": [], "moment_constraint": [], "moment_inverse": [],
            "exo_applied": [], "exo_bias": [], "exo_spring": [], "exo_damper": [], "exo_gravcomp": [], "exo_fluid": [], "exo_passive": [], "exo_actuator": [], "exo_smooth": [], "exo_constraint": [], "exo_inverse": []
            }

        contact_force = None
        flex_body_ids = None
        if record_flex_contact:
            flex_logs = {"time": [], "flex_id": [], "pos": [], "pos_local": [], "force_world": [], "normal": []}
            contact_force = np.zeros(6, dtype=float)
            flex_body_ids = []
            for flex_id in range(model.nflex):
                node_start = model.flex_nodeadr[flex_id]
                if model.flex_nodenum[flex_id] <= 0:
                    flex_body_ids.append(-1)
                    continue
                body_id = int(model.flex_nodebodyid[node_start])
                flex_body_ids.append(body_id)

        with mujoco.viewer.launch_passive(model, data) as viewer:
            knee_joint = model.joint("knee_angle")
            exo_joint = model.joint("shank_band_knee")
            hip_joint = model.joint("hip_flexion")
            ankle_joint = model.joint("ankle_angle")

            knee_actuator_id = model.actuator("knee_actuator").id
            clutch_id = model.actuator("clutch_spring").id

            knee_0 = profile.initial_knee_angle_rad
            data.qpos[knee_joint.qposadr[0]] = knee_0
            data.qpos[exo_joint.qposadr[0]] = knee_0
            data.qpos[hip_joint.qposadr[0]] = kinematics.knee2hip(knee_0)
            data.qpos[ankle_joint.qposadr[0]] = kinematics.knee2ankle(knee_0)

            data.ctrl[knee_actuator_id] = knee_0
            data.ctrl[clutch_id] = knee_0

            clutch_gainprm = model.actuator_gainprm[clutch_id].copy()
            clutch_biasprm = model.actuator_biasprm[clutch_id].copy()
            clutch_gainprm[0] = profile.spring_stiffness
            if clutch_biasprm.size > 1:
                clutch_biasprm[1] = -profile.spring_stiffness

            engaged = profile.clutch_engaged(data.time)
            if engaged:
                data.ctrl[clutch_id] = data.qpos[exo_joint.qposadr[0]]
                model.actuator_gainprm[clutch_id] = clutch_gainprm
                model.actuator_biasprm[clutch_id] = clutch_biasprm
            else:
                model.actuator_gainprm[clutch_id] = 0
                model.actuator_biasprm[clutch_id] = 0

            progress = SimulationProgress(max(target_time, 0.0), model.opt.timestep)
            progress.start()

            step_count = 0
            prev_engaged = engaged
            last_step_duration = None

            initial_phase = profile.gait_phase(data.time)
            initial_label = profile.phase_label(data.time)
            progress.update(data.time, step_count, initial_phase, initial_label, None)

            while viewer.is_running() and data.time < target_time:
                iteration_start = time.perf_counter()

                mujoco.mj_step(model, data)
                step_count += 1

                theta_des_rad = profile.knee_angle_rad(data.time)
                data.ctrl[knee_actuator_id] = theta_des_rad

                engaged_now = profile.clutch_engaged(data.time)
                if engaged_now and not prev_engaged:
                    data.ctrl[clutch_id] = data.qpos[exo_joint.qposadr[0]]
                    model.actuator_gainprm[clutch_id] = clutch_gainprm
                    model.actuator_biasprm[clutch_id] = clutch_biasprm
                elif prev_engaged and not engaged_now:
                    model.actuator_gainprm[clutch_id] = 0
                    model.actuator_biasprm[clutch_id] = 0
                prev_engaged = engaged_now

                if record_force:
                    torque = data.actuator_force[knee_actuator_id]
                    # Control
                    torque_applied = data.qfrc_applied[knee_joint.dofadr[0]] # Applied generalized force
                    # Computed by mj_fwdVelocity/mj_rne (without acceleration)
                    torque_bias = data.qfrc_bias[knee_joint.dofadr[0]] # C(qpos,qvel)
                    # Computed by mj_fwdVelocity/mj_passive
                    torque_spring = data.qfrc_spring[knee_joint.dofadr[0]] # Passive spring force
                    torque_damper = data.qfrc_damper[knee_joint.dofadr[0]] # Passive damper force
                    torque_gravcomp = data.qfrc_gravcomp[knee_joint.dofadr[0]] # Passive gravity compensation force
                    torque_fluid = data.qfrc_fluid[knee_joint.dofadr[0]] # Passive fluid force
                    torque_passive = data.qfrc_passive[knee_joint.dofadr[0]] # Total passive force
                    # Computed by mj_fwdActuation
                    torque_actuator = data.qfrc_actuator[knee_joint.dofadr[0]] # Actuator force
                    # Computed by mj_fwdAcceleration
                    torque_smooth = data.qfrc_smooth[knee_joint.dofadr[0]] # Net unconstrained force
                    # Computed by mj_fwdConstraint/mj_inverse
                    torque_constraint = data.qfrc_constraint[knee_joint.dofadr[0]] # Constraint force
                    # Computed by mj_inverse
                    torque_inverse = data.qfrc_inverse[knee_joint.dofadr[0]] # Net external force; should equal: qfrc_applied + J'*xfrc_applied + qfrc_actuator

                    exo_applied = data.qfrc_applied[exo_joint.dofadr[0]]
                    exo_bias = data.qfrc_bias[exo_joint.dofadr[0]]
                    exo_spring = data.qfrc_spring[exo_joint.dofadr[0]]
                    exo_damper = data.qfrc_damper[exo_joint.dofadr[0]]
                    exo_gravcomp = data.qfrc_gravcomp[exo_joint.dofadr[0]]
                    exo_fluid = data.qfrc_fluid[exo_joint.dofadr[0]]
                    exo_passive = data.qfrc_passive[exo_joint.dofadr[0]]
                    exo_actuator = data.qfrc_actuator[exo_joint.dofadr[0]]
                    exo_smooth = data.qfrc_smooth[exo_joint.dofadr[0]]
                    exo_constraint = data.qfrc_constraint[exo_joint.dofadr[0]]
                    exo_inverse = data.qfrc_inverse[exo_joint.dofadr[0]]

                    spring_torque = -data.actuator_force[clutch_id]
                    exo_torque = data.qfrc_constraint[exo_joint.dofadr[0]]

                    exo_theta = data.qpos[exo_joint.qposadr[0]]
                    spring_length = (exo_theta - data.ctrl[clutch_id]) if engaged_now else 0

                    # Log
                    joint_logs["time"].append(data.time)
                    joint_logs["gait"].append(profile.gait_phase(data.time))
                    joint_logs["moment"].append(torque)
                    joint_logs["phantom_theta"].append(data.qpos[knee_joint.qposadr[0]])
                    joint_logs["phantom_omega"].append(data.qvel[knee_joint.dofadr[0]])
                    joint_logs["phantom_alpha"].append(data.qacc[knee_joint.dofadr[0]])
                    joint_logs["exo_theta"].append(exo_theta)
                    joint_logs["exo_omega"].append(data.qvel[exo_joint.dofadr[0]])
                    joint_logs["exo_alpha"].append(data.qacc[exo_joint.dofadr[0]])
                    joint_logs["exo_moment"].append(exo_torque)
                    joint_logs["spring_moment"].append(spring_torque)
                    joint_logs["spring_length"].append(spring_length)
                    joint_logs["moment_applied"].append(torque_applied)
                    joint_logs["moment_bias"].append(torque_bias)
                    joint_logs["moment_spring"].append(torque_spring)
                    joint_logs["moment_damper"].append(torque_damper)
                    joint_logs["moment_gravcomp"].append(torque_gravcomp)
                    joint_logs["moment_fluid"].append(torque_fluid)
                    joint_logs["moment_passive"].append(torque_passive)
                    joint_logs["moment_actuator"].append(torque_actuator)
                    joint_logs["moment_smooth"].append(torque_smooth)
                    joint_logs["moment_constraint"].append(torque_constraint)
                    joint_logs["moment_inverse"].append(torque_inverse)
                    joint_logs["exo_applied"].append(exo_applied)
                    joint_logs["exo_bias"].append(exo_bias)
                    joint_logs["exo_spring"].append(exo_spring)
                    joint_logs["exo_damper"].append(exo_damper)
                    joint_logs["exo_gravcomp"].append(exo_gravcomp)
                    joint_logs["exo_fluid"].append(exo_fluid)
                    joint_logs["exo_passive"].append(exo_passive)
                    joint_logs["exo_actuator"].append(exo_actuator)
                    joint_logs["exo_smooth"].append(exo_smooth)
                    joint_logs["exo_constraint"].append(exo_constraint)
                    joint_logs["exo_inverse"].append(exo_inverse)

                if record_flex_contact:
                    for contact_id in range(data.ncon):
                        contact = data.contact[contact_id]
                        contact_force[:] = 0.0
                        mujoco.mj_contactForce(model, data, contact_id, contact_force)
                        frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
                        base_force_world = frame.T @ contact_force[:3]
                        contact_pos = np.asarray(contact.pos, dtype=float)

                        for side in (0, 1):
                            flex_id = int(contact.flex[side])
                            if flex_id < 0:
                                continue
                            applied_force = base_force_world if side == 0 else -base_force_world
                            normal_component = contact_force[0] if side == 0 else -contact_force[0]

                            pos_local = contact_pos
                            if flex_body_ids is not None:
                                body_id = flex_body_ids[flex_id] if flex_id < len(flex_body_ids) else -1
                                if body_id >= 0:
                                    body_pos = np.asarray(data.xpos[body_id], dtype=float)
                                    body_rot = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
                                    pos_local = body_rot.T @ (contact_pos - body_pos)

                            flex_logs["time"].append(data.time)
                            flex_logs["flex_id"].append(flex_id)
                            flex_logs["pos"].append(contact_pos.copy())
                            flex_logs["pos_local"].append(np.asarray(pos_local, dtype=float))
                            flex_logs["force_world"].append(applied_force.copy())
                            flex_logs["normal"].append(float(normal_component))


                # Display
                viewer.sync()
                last_step_duration = time.perf_counter() - iteration_start
                gait_phase = profile.gait_phase(data.time)
                phase_label = profile.phase_label(data.time)
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
            gait_phase = profile.gait_phase(data.time)
            phase_label = profile.phase_label(data.time)
            progress.finish(data.time, step_count, gait_phase, phase_label, last_step_duration)

        return (joint_logs, flex_logs), (frames, fps) if record_video else (None, None)
