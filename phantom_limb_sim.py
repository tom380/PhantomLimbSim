import mujoco
import mujoco.viewer
import time
import math

import numpy as np
import matplotlib.pyplot as plt

import imageio.v2 as imageio


# Parameters for the Fourier series (these values are illustrative)
T = 2.0  # duration of one gait cycle (normalized)
a0 = 40  # mean knee angle in degrees
a1 = 20  # amplitude for the first harmonic (degrees)
phi1 = 0  # phase shift for the first harmonic
a2 = 10  # amplitude for the second harmonic (degrees)
phi2 = np.pi/4  # phase shift for the second harmonic

def knee_angle_fourier(t, T):
    gait = t / T
    theta = 45 + 10 * np.sin(2 * np.pi * (gait + 0.03) + np.pi - 0.1) + 45 + 9.7 * np.sin(4 *np.pi * (gait + 0.04) + np.pi + 1.6) - 49.9
    return theta

def knee2foot(knee_angle):
    femur = 0.29869 #+ 0.02  # Length from hip to knee (in meters)
    tibia = 0.30433 #+ 0.02  # Length from knee to ankle (in meters)
    d = femur + tibia - math.sqrt(femur**2 + tibia**2 - 2 * femur * tibia * math.cos(math.pi - knee_angle))
    return d

model = mujoco.MjModel.from_xml_path("phantom_barrutia.xml")
data = mujoco.MjData(model)
# mujoco.viewer.launch(model, data)

renderer = mujoco.Renderer(model, 640, 480)
frames   = []
fps      = int(round(1 / model.opt.timestep))

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = True
    start = time.time()

    # Initialize the plot
    plt.ion()
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    line_actual, = ax.plot([], [], label="Actual Knee Angle")
    line_desired, = ax.plot([], [], label="Desired Knee Angle")
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    line_force, = ax2.plot([], [], label="Ankle Force", color=default_colors[2])
    ax.set_xlim(0, 1)
    ax.set_ylim(10, 80)
    ax2.set_ylim(-30, 30)
    ax.set_xlabel("Gait Percentage")
    ax.set_ylabel("Knee Angle (degrees)")
    ax2.set_ylabel("Force (N)")

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    ax.legend(handles + handles2, labels + labels2)
    # ax.legend()
    ax.grid(True)

    # Data storage for plotting
    gait_percentage = []
    actual_knee_angles = []
    desired_knee_angles = []
    ankle_force = []

    step_count = 0
    while viewer.is_running():
        step_start = time.time()

        # Perform simulation step
        mujoco.mj_step(model, data)

        
        renderer.update_scene(data, camera='fixed_cam')
        frame_rgb = renderer.render()
        frames.append(frame_rgb.copy())

        if data.time % T < model.opt.timestep:
            time.sleep(2)
            gait_percentage.clear()
            actual_knee_angles.clear()
            desired_knee_angles.clear()
            ankle_force.clear()


        desired_knee_angle = knee_angle_fourier(data.time, T) * math.pi / 180
        platform_set = knee2foot(desired_knee_angle)

        data.mocap_pos[0] = [0, 0, platform_set]
        # data.mocap_quat[0] = [0, 0, 0, 1]
        actual_knee_angle = data.qpos[model.joint("knee_angle_l").qposadr[0]]


        if step_count % 10 == 0:
            gait_percentage.append((data.time % T) / T)
            actual_knee_angles.append(actual_knee_angle * 180 / math.pi)
            desired_knee_angles.append(desired_knee_angle * 180 / math.pi)
            ankle_force.append(data.sensordata[2])

            line_actual.set_data(gait_percentage, actual_knee_angles)
            line_desired.set_data(gait_percentage, desired_knee_angles)
            line_force.set_data(gait_percentage, ankle_force)
            fig.canvas.draw()
            fig.canvas.flush_events()

        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        step_count += 1

    imageio.mimsave("run.mp4", frames, fps=fps, codec="libx264")

    # Wait until the figure is closed
    plt.ioff()
    plt.show()