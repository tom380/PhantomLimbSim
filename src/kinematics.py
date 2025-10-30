import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import scipy.io
from scipy.interpolate import CubicSpline

from config import T_CYCLE, LENGTH_FEMUR, LENGTH_TIBIA

SPRING_STIFFNESSES = (
    59.604965380516845,
    89.692223816246951,
    144.5135624936535,
    195.8747359748776,
    283.1155274990882,
    351.8861634135672,
)

KINEMATICS_SAMPLE_RATE = 1000.0
KINEMATICS_DT = 1.0 / KINEMATICS_SAMPLE_RATE
AVERAGE_KNEE_PATH = "src/knee_angles.mat"
FULL_KINEMATICS_PATH = "src/kinematics.mat"


@dataclass(frozen=True)
class KinematicsProfile:
    mode: str
    spring_index: int
    spring_stiffness: float
    dt: Optional[float]
    dataset_duration: Optional[float]
    cycle_duration: float
    initial_knee_angle_rad: float
    _knee_angle_fn: Callable[[float], float]
    _clutch_fn: Callable[[float], bool]
    _gait_phase_fn: Callable[[float], float]
    _phase_label_fn: Callable[[float], str]

    def knee_angle_rad(self, t: float) -> float:
        return self._knee_angle_fn(t)

    def clutch_engaged(self, t: float) -> bool:
        return self._clutch_fn(t)

    def gait_phase(self, t: float) -> float:
        return self._gait_phase_fn(t)

    def phase_label(self, t: float) -> str:
        return self._phase_label_fn(t)


def get_spring_stiffness(index: int) -> float:
    if not 1 <= index <= len(SPRING_STIFFNESSES):
        raise ValueError(
            f"Spring index must be between 1 and {len(SPRING_STIFFNESSES)} (received {index})."
        )
    return SPRING_STIFFNESSES[index - 1]


def knee_angle_fourier(t: float) -> float:
    if not hasattr(knee_angle_fourier, "_spline"):
        mat = scipy.io.loadmat(AVERAGE_KNEE_PATH)
        angles = np.squeeze(mat["angles"])

        x = np.linspace(0, 1, len(angles), endpoint=False)
        knee_angle_fourier._spline = CubicSpline(x, angles, bc_type="periodic")

    gait = (t / T_CYCLE) % 1.0
    return float(knee_angle_fourier._spline(gait))


def _find_cycle_starts(clutch: np.ndarray) -> np.ndarray:
    clutch_int = clutch.astype(int)
    rising = np.where(np.diff(clutch_int) > 0)[0] + 1
    if clutch[0]:
        rising = np.insert(rising, 0, 0)
    if rising.size == 0:
        return np.array([0], dtype=int)
    return rising.astype(int)


def _compute_cycle_durations(start_indices: np.ndarray, total_samples: int, dt: float) -> np.ndarray:
    if start_indices.size == 0:
        return np.array([total_samples * dt], dtype=float)

    end_indices = np.append(start_indices[1:], total_samples)
    durations = (end_indices - start_indices) * dt
    return durations.astype(float)


def _load_full_dataset():
    if hasattr(_load_full_dataset, "_cache"):
        return _load_full_dataset._cache

    mat = scipy.io.loadmat(FULL_KINEMATICS_PATH, squeeze_me=True)
    phantom = np.asarray(mat["phantom_theta"])
    clutch = np.asarray(mat["clutch"])

    dataset = []
    for idx in range(len(SPRING_STIFFNESSES)):
        knee_deg = np.asarray(phantom[idx], dtype=float).reshape(-1)
        clutch_raw = np.asarray(clutch[idx], dtype=int).reshape(-1)
        clutch_bool = clutch_raw.astype(bool)

        sample_count = knee_deg.size
        time = np.arange(sample_count, dtype=float) * KINEMATICS_DT
        cycle_start_indices = _find_cycle_starts(clutch_bool)
        cycle_durations = _compute_cycle_durations(cycle_start_indices, sample_count, KINEMATICS_DT)
        avg_cycle_duration = (
            float(cycle_durations.mean()) if cycle_durations.size else float(sample_count * KINEMATICS_DT)
        )

        if cycle_start_indices.size:
            cycle_start_times = time[cycle_start_indices]
        else:
            cycle_start_times = np.array([0.0], dtype=float)

        dataset.append(
            {
                "time": time,
                "angle_rad": np.radians(knee_deg),
                "clutch": clutch_bool,
                "dt": KINEMATICS_DT,
                "cycle_start_indices": cycle_start_indices,
                "cycle_start_times": cycle_start_times,
                "cycle_durations": cycle_durations,
                "avg_cycle_duration": avg_cycle_duration,
            }
        )

    _load_full_dataset._cache = dataset
    return dataset


def _create_average_profile(spring_index: int) -> KinematicsProfile:
    stiffness = get_spring_stiffness(spring_index)

    def knee_angle_fn(t: float) -> float:
        return math.radians(knee_angle_fourier(t))

    def gait_phase_fn(t: float) -> float:
        if T_CYCLE == 0:
            return 0.0
        return (t % T_CYCLE) / T_CYCLE

    def clutch_fn(t: float) -> bool:
        return (t % T_CYCLE) < (0.35 * T_CYCLE)

    def phase_label_fn(t: float) -> str:
        return "stance" if gait_phase_fn(t) < 0.6 else "swing"

    initial_angle = knee_angle_fn(0.0)

    return KinematicsProfile(
        mode="average",
        spring_index=spring_index,
        spring_stiffness=stiffness,
        dt=None,
        dataset_duration=None,
        cycle_duration=T_CYCLE,
        initial_knee_angle_rad=initial_angle,
        _knee_angle_fn=knee_angle_fn,
        _clutch_fn=clutch_fn,
        _gait_phase_fn=gait_phase_fn,
        _phase_label_fn=phase_label_fn,
    )


def _create_full_profile(spring_index: int) -> KinematicsProfile:
    data = _load_full_dataset()[spring_index - 1]
    time = data["time"]
    angle_rad = data["angle_rad"]
    clutch = data["clutch"]
    dt = data["dt"]
    cycle_start_times = data["cycle_start_times"]
    cycle_durations = data["cycle_durations"]
    avg_cycle_duration = data["avg_cycle_duration"]

    sample_count = angle_rad.size
    total_duration = float(time[-1]) if sample_count > 1 else 0.0

    def knee_angle_fn(t: float) -> float:
        return float(np.interp(t, time, angle_rad, left=angle_rad[0], right=angle_rad[-1]))

    def clutch_fn(t: float) -> bool:
        if t <= 0.0:
            return bool(clutch[0])
        if t >= total_duration:
            return bool(clutch[-1])
        idx = int(np.clip(t / dt, 0, sample_count - 1))
        return bool(clutch[idx])

    def gait_phase_fn(t: float) -> float:
        if cycle_start_times.size == 0:
            return 0.0

        idx = np.searchsorted(cycle_start_times, t, side="right") - 1
        if idx < 0:
            idx = 0

        if idx >= len(cycle_durations):
            duration = cycle_durations[-1] if cycle_durations.size else avg_cycle_duration
        else:
            duration = cycle_durations[idx]

        if duration <= 0:
            duration = avg_cycle_duration if avg_cycle_duration > 0 else 1.0

        start = cycle_start_times[idx]
        phase = (t - start) / duration
        return float(np.clip(phase, 0.0, 1.0))

    def phase_label_fn(t: float) -> str:
        return "stance" if clutch_fn(t) else "swing"

    return KinematicsProfile(
        mode="full",
        spring_index=spring_index,
        spring_stiffness=get_spring_stiffness(spring_index),
        dt=dt,
        dataset_duration=total_duration,
        cycle_duration=avg_cycle_duration,
        initial_knee_angle_rad=float(angle_rad[0]),
        _knee_angle_fn=knee_angle_fn,
        _clutch_fn=clutch_fn,
        _gait_phase_fn=gait_phase_fn,
        _phase_label_fn=phase_label_fn,
    )


def get_kinematics_profile(mode: str, spring_index: int) -> KinematicsProfile:
    mode = mode.lower()
    if mode == "average":
        return _create_average_profile(spring_index)
    if mode == "full":
        return _create_full_profile(spring_index)
    raise ValueError(f"Unsupported kinematics mode '{mode}'. Expected 'average' or 'full'.")


def D(theta_rad: float) -> float:
    theta = np.pi - theta_rad
    return np.sqrt(LENGTH_FEMUR ** 2 + LENGTH_TIBIA ** 2 - 2 * LENGTH_FEMUR * LENGTH_TIBIA * np.cos(theta))


def knee2foot(theta_rad: float) -> float:
    return LENGTH_FEMUR + LENGTH_TIBIA - D(theta_rad)


def knee2hip(theta_rad: float) -> float:
    return -np.asin(LENGTH_TIBIA * np.sin(np.pi - theta_rad) / D(theta_rad))


def knee2ankle(theta_rad: float) -> float:
    return -np.asin(LENGTH_FEMUR * np.sin(np.pi - theta_rad) / D(theta_rad))

