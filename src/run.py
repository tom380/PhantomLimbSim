"""Core MuJoCo simulation loop and optional logging/recording outputs."""

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from config import TOTAL_SIM_TIME
import kinematics
import xmlparser
from progress_display import SimulationProgress


def _setup_video_capture(model, video_fps, video_cameras):
    """Create camera-specific renderers and frame-sampling settings."""
    if video_fps <= 0:
        raise ValueError(f"Video FPS must be positive (received {video_fps}).")

    camera_names = video_cameras if video_cameras else ["fixed_cam"]
    unique_cameras = []
    seen = set()
    for camera_name in camera_names:
        if camera_name in seen:
            continue
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id < 0:
            raise ValueError(f"Unknown camera '{camera_name}' in model.")
        seen.add(camera_name)
        unique_cameras.append(camera_name)

    opt_scene = mujoco.MjvOption()
    opt_scene.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    timestep = float(model.opt.timestep)
    timesteps_per_frame = max(1, int(np.floor(1.0 / (video_fps * timestep))))
    recording_fps = 1.0 / (timesteps_per_frame * timestep)
    renderers = {name: mujoco.Renderer(model, 640, 480) for name in unique_cameras}
    frames_by_camera = {name: [] for name in unique_cameras}

    return renderers, frames_by_camera, opt_scene, timesteps_per_frame, recording_fps


def sim(
    model_path,
    actuated=True,
    record_video=False,
    video_fps=120.0,
    video_cameras=None,
    record_force=False,
    record_flex_contact=False,
    kinematics_mode="average",
    spring_index=3,
    sim_time=None,
    kinematics_data_dir="data/kinematics",
):
    """Run the simulation and optionally collect force, contact, and video outputs."""
    if not Path(model_path).exists():
        raise FileNotFoundError(model_path)

    # Parse custom XML tags (<meshfinder>, <pinmesh>, etc.) before building the MuJoCo model.
    model_string = xmlparser.parse(model_path)
    model = mujoco.MjModel.from_xml_string(model_string)
    data = mujoco.MjData(model)

    profile = kinematics.get_kinematics_profile(
        mode=kinematics_mode,
        spring_index=spring_index,
        data_dir=kinematics_data_dir,
    )

    default_time = profile.dataset_duration if profile.mode == "full" else TOTAL_SIM_TIME
    target_time = default_time

    # Runtime override rules:
    # - sim_time > 0: use that value
    # - sim_time <= 0: use full dataset duration (if available)
    # - when in full mode, always clamp to available dataset length
    if sim_time is not None:
        if sim_time <= 0:
            target_time = profile.dataset_duration if profile.dataset_duration is not None else default_time
        else:
            target_time = sim_time

    if profile.dataset_duration is not None:
        dataset_time = profile.dataset_duration
        if dataset_time <= 0:
            dataset_time = 0.0
        if target_time is None or target_time <= 0:
            target_time = dataset_time
        elif dataset_time > 0:
            target_time = min(target_time, dataset_time)

    if target_time is None or target_time <= 0:
        target_time = TOTAL_SIM_TIME

    target_time = float(target_time)

    joint_logs = None
    flex_logs = None
    frames_by_camera = None
    recording_fps = None

    if not actuated:
        mujoco.viewer.launch(model, data)
        return (None, None), (None, None)

    if record_video:
        (
            video_renderers,
            frames_by_camera,
            video_scene_option,
            timesteps_per_frame,
            recording_fps,
        ) = _setup_video_capture(model, video_fps, video_cameras)

    if record_force:
        # Fixed schema so downstream MATLAB/post-processing can rely on consistent keys.
        joint_logs = {
            "time": [],
            "gait": [],
            "moment": [],
            "phantom_theta": [],
            "phantom_omega": [],
            "phantom_alpha": [],
            "exo_theta": [],
            "exo_omega": [],
            "exo_alpha": [],
            "exo_moment": [],
            "spring_moment": [],
            "spring_length": [],
            "moment_applied": [],
            "moment_bias": [],
            "moment_spring": [],
            "moment_damper": [],
            "moment_gravcomp": [],
            "moment_fluid": [],
            "moment_passive": [],
            "moment_actuator": [],
            "moment_smooth": [],
            "moment_constraint": [],
            "moment_inverse": [],
            "exo_applied": [],
            "exo_bias": [],
            "exo_spring": [],
            "exo_damper": [],
            "exo_gravcomp": [],
            "exo_fluid": [],
            "exo_passive": [],
            "exo_actuator": [],
            "exo_smooth": [],
            "exo_constraint": [],
            "exo_inverse": [],
        }

    contact_force = None
    flex_body_ids = None
    if record_flex_contact:
        flex_logs = {
            "time": [],
            "flex_id": [],
            "pos": [],
            "pos_local": [],
            "force_world": [],
            "normal": [],
            "body_id": [],
            "geom_id": [],
        }
        contact_force = np.zeros(6, dtype=float)

        # Hard-map flex_id -> body_id for known flexes, then fallback to inferred node ownership.
        flex_body_ids = [-1] * model.nflex
        flex_name_to_body = {"thigh": "thigh", "l_shank": "shank"}
        flex_body_lookup = {}
        for flex_name, body_name in flex_name_to_body.items():
            try:
                flex_body_lookup[flex_name] = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, body_name
                )
            except Exception:
                flex_body_lookup[flex_name] = -1

        for flex_id in range(model.nflex):
            body_id = -1
            if flex_id == 0 and flex_body_lookup.get("thigh", -1) >= 0:
                body_id = flex_body_lookup["thigh"]
            elif flex_id == 1 and flex_body_lookup.get("l_shank", -1) >= 0:
                body_id = flex_body_lookup["l_shank"]

            if body_id < 0:
                node_start = model.flex_nodeadr[flex_id]
                node_num = model.flex_nodenum[flex_id]
                if node_num > 0:
                    nodes = model.flex_nodebodyid[node_start : node_start + node_num]
                    nodes = [int(x) for x in nodes if int(x) >= 0]
                    if nodes:
                        vals, counts = np.unique(nodes, return_counts=True)
                        body_id = int(vals[np.argmax(counts)])

            flex_body_ids[flex_id] = body_id

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

        # Initialize clutch state at t=0 so the first step starts with correct spring behavior.
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

            # Advance simulation, then apply desired knee position for the new state.
            mujoco.mj_step(model, data)
            step_count += 1

            theta_des_rad = profile.knee_angle_rad(data.time)
            data.ctrl[knee_actuator_id] = theta_des_rad

            # Apply clutch stiffness only while the gait profile reports engagement.
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
                # Platform/knee-side generalized force breakdown.
                torque = data.actuator_force[knee_actuator_id]
                torque_applied = data.qfrc_applied[knee_joint.dofadr[0]]
                torque_bias = data.qfrc_bias[knee_joint.dofadr[0]]
                torque_spring = data.qfrc_spring[knee_joint.dofadr[0]]
                torque_damper = data.qfrc_damper[knee_joint.dofadr[0]]
                torque_gravcomp = data.qfrc_gravcomp[knee_joint.dofadr[0]]
                torque_fluid = data.qfrc_fluid[knee_joint.dofadr[0]]
                torque_passive = data.qfrc_passive[knee_joint.dofadr[0]]
                torque_actuator = data.qfrc_actuator[knee_joint.dofadr[0]]
                torque_smooth = data.qfrc_smooth[knee_joint.dofadr[0]]
                torque_constraint = data.qfrc_constraint[knee_joint.dofadr[0]]
                torque_inverse = data.qfrc_inverse[knee_joint.dofadr[0]]

                # Exosuit-side generalized force breakdown (same components on exo joint dof).
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

                # Spring torque comes directly from clutch actuator force sign convention.
                spring_torque = -data.actuator_force[clutch_id]
                exo_torque = data.qfrc_constraint[exo_joint.dofadr[0]]

                exo_theta = data.qpos[exo_joint.qposadr[0]]
                spring_length = (exo_theta - data.ctrl[clutch_id]) if engaged_now else 0

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

                    # contact.frame stores contact basis vectors; rotate local normal/tangent force to world frame.
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
                        geom_id = int(contact.geom[side])
                        body_id = flex_body_ids[flex_id] if flex_id < len(flex_body_ids) else -1
                        if body_id < 0 and geom_id >= 0:
                            body_id = int(model.geom_bodyid[geom_id])

                        # Also store body-local contact position when a body frame is available.
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
                        flex_logs["body_id"].append(body_id)
                        flex_logs["geom_id"].append(geom_id)

            viewer.sync()
            last_step_duration = time.perf_counter() - iteration_start
            gait_phase = profile.gait_phase(data.time)
            phase_label = profile.phase_label(data.time)
            progress.update(data.time, step_count, gait_phase, phase_label, last_step_duration)

            # Downsample render calls to approximate target video FPS.
            if record_video and int(data.time / model.opt.timestep) % timesteps_per_frame == 0:
                for camera_name, renderer in video_renderers.items():
                    renderer.update_scene(data, camera=camera_name, scene_option=video_scene_option)
                    frames_by_camera[camera_name].append(renderer.render().copy())

            # Keep wall-clock loop close to simulation timestep when compute time permits.
            loop_duration = time.perf_counter() - iteration_start
            dt = model.opt.timestep - loop_duration
            if dt > 0:
                time.sleep(dt)

        gait_phase = profile.gait_phase(data.time)
        phase_label = profile.phase_label(data.time)
        progress.finish(data.time, step_count, gait_phase, phase_label, last_step_duration)

    return (joint_logs, flex_logs), (frames_by_camera, recording_fps) if record_video else (None, None)
