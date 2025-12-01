import imageio.v2 as imageio
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

def plot_last_cycle(logs, output_fn="simulation_results_last_cycle"):
    gait = np.array(logs["gait"])
    if gait.size == 0:
        raise ValueError("No gait data available in logs.")

    resets = np.where(np.diff(gait) < -0.5)[0]
    start_idx = int(resets[-1] + 1) if resets.size else 0
    gait_segment = gait[start_idx:]
    idx = np.argsort(gait_segment)

    knee_a = np.array(logs["phantom_theta"])[start_idx:][idx]
    knee_e = np.array(logs["exo_theta"])[start_idx:][idx]
    moment = np.array(logs["moment"])[start_idx:][idx]
    moment_exo = np.array(logs["exo_moment"])[start_idx:][idx]

    fig, ax1 = plt.subplots()
    ax1.plot(gait_segment[idx], knee_a, label="Knee angle")
    ax1.plot(gait_segment[idx], knee_e, label="Exo angle")
    ax1.set_xlabel("Gait %")
    ax1.set_ylabel("Knee angle (rad)")
    ax1.set_xlim(0,1)

    ax2 = ax1.twinx()
    ax2.plot(gait_segment[idx], moment, label="Moment Platform", color='green')
    ax2.plot(gait_segment[idx], moment_exo, label="Moment Exo", color='red')
    ax2.set_ylabel("Moment (Nm)")

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid(True)
    fig.tight_layout()
    fig.savefig(output_fn + ".png", dpi=300)
    plt.show()
    print("Saved " + output_fn + ".png")

def save_video(frames, fps, fn="run"):
    imageio.mimsave(fn + ".mp4", frames, fps=fps, codec="libx264")
    print("Saved " + fn + ".mp4")

def save_mat(logs, fn="simulation_data"):
    data = {k: np.asarray(v) for k,v in logs.items()}
    sio.savemat(fn + ".mat", data)
    print("Saved " + fn + ".mat" + " (MATLAB-compatible)")

def save_flex_contacts(flex_logs, fn=None):
    if flex_logs is None:
        raise ValueError("No flex contact data provided.")

    name = fn + "_flex_contacts" if fn else "flex_contacts_simple"
    mat_payload = {
        "time": np.asarray(flex_logs["time"], dtype=float),
        "flex_id": np.asarray(flex_logs["flex_id"], dtype=int),
        "pos": np.asarray(flex_logs["pos"], dtype=float),
        "pos_local": np.asarray(flex_logs.get("pos_local", []), dtype=float),
        "force_world": np.asarray(flex_logs["force_world"], dtype=float),
        "normal": np.asarray(flex_logs["normal"], dtype=float),
        "body_id": np.asarray(flex_logs.get("body_id", []), dtype=int),
        "geom_id": np.asarray(flex_logs.get("geom_id", []), dtype=int),
        "body_pos_world": np.asarray(flex_logs.get("body_pos_world", []), dtype=float),
        "body_rot_world": np.asarray(flex_logs.get("body_rot_world", []), dtype=float),
    }
    sio.savemat(name + ".mat", mat_payload)
    print("Saved " + name + ".mat" + " (flex contacts)")
