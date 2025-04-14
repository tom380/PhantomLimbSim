import mujoco
import mujoco.viewer
import time
import math

import numpy as np
import matplotlib.pyplot as plt

# Parameters for the Fourier series (these values are illustrative)
T = 2.0  # duration of one gait cycle (normalized)
a0 = 40  # mean knee angle in degrees
a1 = 20  # amplitude for the first harmonic (degrees)
phi1 = 0  # phase shift for the first harmonic
a2 = 10  # amplitude for the second harmonic (degrees)
phi2 = np.pi/4  # phase shift for the second harmonic

def knee_angle_fourier(t, T, a0, a1, phi1, a2, phi2):
    """
    Approximate knee angle as a Fourier series with two harmonics.

    t : time or normalized gait cycle (0 <= t <= T)
    """
    theta = a0 + a1 * np.cos(2 * np.pi * t/T + phi1) + a2 * np.cos(4 * np.pi * t/T + phi2)
    return t / T * 90

def knee2foot(knee_angle):
    femur = 0.40445  # Length from hip to knee (in meters)
    tibia = 0.40012  # Length from knee to ankle (in meters)
    d = femur + tibia - math.sqrt(femur**2 + tibia**2 - 2 * femur * tibia * math.cos(math.pi - knee_angle))
    return d

model = mujoco.MjModel.from_xml_path("phantom_limb.xml")
viewer = mujoco.viewer.launch(model)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()

    # Initialize the plot
    plt.ion()
    fig, ax = plt.subplots()
    line_actual, = ax.plot([], [], label="Actual Knee Angle")
    line_desired, = ax.plot([], [], label="Desired Knee Angle")
    line_platform_set, = ax.plot([], [], label="Platform Set Height")
    line_platform_actual, = ax.plot([], [], label="Platform Actual Height")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 180)
    ax.set_xlabel("Gait Percentage")
    ax.set_ylabel("Knee Angle / Platform Height (degrees / meters)")
    ax.legend()
    ax.grid(True)

    # Data storage for plotting
    gait_percentage = []
    actual_knee_angles = []
    desired_knee_angles = []
    platform_set_heights = []
    platform_actual_heights = []

    while viewer.is_running():
        step_start = time.time()

        # Perform simulation step
        mujoco.mj_step(model, data)

        # Reset the plot for each gait cycle
        if data.time % T < model.opt.timestep:
            gait_percentage.clear()
            actual_knee_angles.clear()
            desired_knee_angles.clear()
            platform_set_heights.clear()
            platform_actual_heights.clear()

        # Calculate desired and actual knee angles
        desired_knee_angle = knee_angle_fourier(data.time, T, a0, a1, phi1, a2, phi2) * math.pi / 180
        platform_set = knee2foot(desired_knee_angle)
        data.ctrl[model.actuator("platform_act").id] = platform_set
        actual_knee_angle = data.qpos[model.joint("knee_angle_l").qposadr[0]]
        platform_actual = data.qpos[model.joint("platform_slide").qposadr[0]]

        # Update data for plotting
        gait_percentage.append((data.time % T) / T)
        actual_knee_angles.append(actual_knee_angle * 180 / math.pi)
        desired_knee_angles.append(desired_knee_angle * 180 / math.pi)
        platform_set_heights.append(platform_set * 1000)
        platform_actual_heights.append(platform_actual * 1000)

        # Update the plot
        line_actual.set_data(gait_percentage, actual_knee_angles)
        line_desired.set_data(gait_percentage, desired_knee_angles)
        line_platform_set.set_data(gait_percentage, platform_set_heights)
        line_platform_actual.set_data(gait_percentage, platform_actual_heights)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(max(actual_knee_angles), max(desired_knee_angles), max(platform_set_heights), max(platform_actual_heights)) + 0.1)
        plt.draw()
        plt.pause(0.001)

        # Sync the viewer
        viewer.sync()

        # Rudimentary time keeping
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    # Wait until the figure is closed
    plt.ioff()
    plt.show()