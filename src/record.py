import imageio.v2 as imageio
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

from config import T_CYCLE, TOTAL_SIM_TIME

def plot_last_cycle(logs, output_fn="simulation_results_last_cycle.png"):
    t    = np.array(logs["time"])
    mask = t >= (TOTAL_SIM_TIME - T_CYCLE)
    gait = np.array(logs["gait"])[mask]
    idx  = np.argsort(gait)

    knee_a = np.array(logs["knee_act"])[mask][idx]
    knee_d = np.array(logs["knee_des"])[mask][idx]
    Fm     = np.array(logs["F_meas"])[mask][idx]
    Ft     = np.array(logs["F_theo"])[mask][idx]

    fig, ax1 = plt.subplots()
    ax1.plot(gait[idx], knee_a, label="Actual Knee")
    ax1.plot(gait[idx], knee_d, label="Desired Knee")
    ax1.set_xlabel("Gait %")
    ax1.set_ylabel("Knee angle (°)")
    ax1.set_xlim(0,1)

    ax2 = ax1.twinx()
    ax2.plot(gait[idx], Fm, "--", label="Measured F")
    ax2.plot(gait[idx], Ft, "--", label="Theoretical F")
    ax2.set_ylabel("Force (N)")

    ax1.legend(loc="upper right")
    ax1.grid(True)
    fig.tight_layout()
    fig.savefig(output_fn, dpi=300)
    plt.show()
    print("Saved" + output_fn)

def save_video(frames, fps, fn="run.mp4"):
    imageio.mimsave(fn, frames, fps=fps, codec="libx264")
    print("Saved " + fn)

def save_mat(logs, fn="simulation_data.mat"):
    data = {k: np.asarray(v) for k,v in logs.items()}
    sio.savemat(fn, data)
    print("Saved"+ fn + " (MATLAB‑compatible)")