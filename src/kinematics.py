"""Kinematic profile loading and geometric helper transforms for the simulator."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import scipy.io
from scipy.interpolate import CubicSpline

from config import LENGTH_FEMUR, LENGTH_TIBIA, T_CYCLE

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
AVERAGE_KNEE_FILENAME = "knee_angles.mat"
FULL_KINEMATICS_FILENAME = "kinematics.mat"
DEFAULT_KINEMATICS_DATA_DIR = "data/kinematics"

# These thresholds are intentionally fixed model assumptions for average-mode gait synthesis.
AVERAGE_CLUTCH_DUTY = 0.35
AVERAGE_STANCE_PHASE_SPLIT = 0.6

_AVERAGE_SPLINE_CACHE = {}
_FULL_DATASET_CACHE = {}


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


def _resolve_data_paths(data_dir):
    """Resolve average/full MAT file paths from a data directory setting."""
    base_dir = Path(data_dir or DEFAULT_KINEMATICS_DATA_DIR).expanduser()
    candidate_dirs = [base_dir]
    if not base_dir.is_absolute():
        candidate_dirs.append(Path(__file__).resolve().parent.parent / base_dir)

    for candidate_dir in candidate_dirs:
        average_path = candidate_dir / AVERAGE_KNEE_FILENAME
        full_path = candidate_dir / FULL_KINEMATICS_FILENAME
        if average_path.exists() and full_path.exists():
            return average_path, full_path

    raise FileNotFoundError(
        f"Could not find {AVERAGE_KNEE_FILENAME} and {FULL_KINEMATICS_FILENAME} in any of: "
        + ", ".join(str(p) for p in candidate_dirs)
    )


def get_spring_stiffness(index: int) -> float:
    if not 1 <= index <= len(SPRING_STIFFNESSES):
        raise ValueError(
            f"Spring index must be between 1 and {len(SPRING_STIFFNESSES)} (received {index})."
        )
    return SPRING_STIFFNESSES[index - 1]


def _average_spline(average_path):
    """Load and cache the periodic spline used by average-mode knee angle lookup."""
    cache_key = str(Path(average_path).resolve())
    spline = _AVERAGE_SPLINE_CACHE.get(cache_key)
    if spline is None:
        mat = scipy.io.loadmat(average_path)
        angles = np.squeeze(mat["angles"])
        x = np.linspace(0, 1, len(angles), endpoint=False)
        spline = CubicSpline(x, angles, bc_type="periodic")
        _AVERAGE_SPLINE_CACHE[cache_key] = spline
    return spline


def knee_angle_fourier(t: float) -> float:
    """Backward-compatible default average-profile angle lookup in degrees."""
    average_path, _ = _resolve_data_paths(DEFAULT_KINEMATICS_DATA_DIR)
    gait = (t / T_CYCLE) % 1.0
    return float(_average_spline(average_path)(gait))


def _find_cycle_starts(clutch):
    """Return sample indices where clutch transitions from open to engaged."""
    clutch_int = clutch.astype(int)
    rising = np.where(np.diff(clutch_int) > 0)[0] + 1
    if clutch[0]:
        rising = np.insert(rising, 0, 0)
    if rising.size == 0:
        return np.array([0], dtype=int)
    return rising.astype(int)


def _compute_cycle_durations(start_indices, total_samples, dt):
    """Compute per-cycle durations from cycle start indices."""
    if start_indices.size == 0:
        return np.array([total_samples * dt], dtype=float)

    end_indices = np.append(start_indices[1:], total_samples)
    durations = (end_indices - start_indices) * dt
    return durations.astype(float)


def _load_full_dataset(full_path):
    """Load and cache full recorded kinematics for all spring indices."""
    cache_key = str(Path(full_path).resolve())
    dataset = _FULL_DATASET_CACHE.get(cache_key)
    if dataset is not None:
        return dataset

    mat = scipy.io.loadmat(full_path, squeeze_me=True)
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
        cycle_start_times = time[cycle_start_indices] if cycle_start_indices.size else np.array([0.0], dtype=float)

        dataset.append(
            {
                "time": time,
                "angle_rad": np.radians(knee_deg),
                "clutch": clutch_bool,
                "dt": KINEMATICS_DT,
                "cycle_start_times": cycle_start_times,
                "cycle_durations": cycle_durations,
                "avg_cycle_duration": avg_cycle_duration,
            }
        )

    _FULL_DATASET_CACHE[cache_key] = dataset
    return dataset


def _create_average_profile(spring_index, average_path):
    """Create a periodic profile driven by one averaged gait cycle."""
    stiffness = get_spring_stiffness(spring_index)
    spline = _average_spline(average_path)

    def knee_angle_fn(t):
        gait = (t / T_CYCLE) % 1.0
        return math.radians(float(spline(gait)))

    def gait_phase_fn(t):
        if T_CYCLE == 0:
            return 0.0
        return (t % T_CYCLE) / T_CYCLE

    def clutch_fn(t):
        return (t % T_CYCLE) < (AVERAGE_CLUTCH_DUTY * T_CYCLE)

    def phase_label_fn(t):
        return "stance" if gait_phase_fn(t) < AVERAGE_STANCE_PHASE_SPLIT else "swing"

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


def _create_full_profile(spring_index, full_path):
    """Create a profile that replays the recorded full dataset in time."""
    data = _load_full_dataset(full_path)[spring_index - 1]
    time = data["time"]
    angle_rad = data["angle_rad"]
    clutch = data["clutch"]
    dt = data["dt"]
    cycle_start_times = data["cycle_start_times"]
    cycle_durations = data["cycle_durations"]
    avg_cycle_duration = data["avg_cycle_duration"]

    sample_count = angle_rad.size
    total_duration = float(time[-1]) if sample_count > 1 else 0.0

    def knee_angle_fn(t):
        return float(np.interp(t, time, angle_rad, left=angle_rad[0], right=angle_rad[-1]))

    def clutch_fn(t):
        if t <= 0.0:
            return bool(clutch[0])
        if t >= total_duration:
            return bool(clutch[-1])
        idx = int(np.clip(t / dt, 0, sample_count - 1))
        return bool(clutch[idx])

    def gait_phase_fn(t):
        if cycle_start_times.size == 0:
            return 0.0

        # Reconstruct local cycle phase by locating the latest cycle boundary, then normalizing by
        # that cycle's measured duration from the recorded clutch transitions.
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

    def phase_label_fn(t):
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


def get_kinematics_profile(mode: str, spring_index: int, data_dir=DEFAULT_KINEMATICS_DATA_DIR) -> KinematicsProfile:
    """Return the configured kinematics profile using average or full dataset mode.

    `mode="average"` builds a periodic spline from one gait cycle.
    `mode="full"` replays the recorded dataset and derives cycle phase from clutch transitions.
    """
    average_path, full_path = _resolve_data_paths(data_dir)

    mode = mode.lower()
    if mode == "average":
        return _create_average_profile(spring_index, average_path)
    if mode == "full":
        return _create_full_profile(spring_index, full_path)
    raise ValueError(f"Unsupported kinematics mode '{mode}'. Expected 'average' or 'full'.")


def D(theta_rad):
    theta = np.pi - theta_rad
    return np.sqrt(LENGTH_FEMUR ** 2 + LENGTH_TIBIA ** 2 - 2 * LENGTH_FEMUR * LENGTH_TIBIA * np.cos(theta))


def knee2foot(theta_rad):
    return LENGTH_FEMUR + LENGTH_TIBIA - D(theta_rad)


def knee2hip(theta_rad):
    return -np.asin(LENGTH_TIBIA * np.sin(np.pi - theta_rad) / D(theta_rad))


def knee2ankle(theta_rad):
    return -np.asin(LENGTH_FEMUR * np.sin(np.pi - theta_rad) / D(theta_rad))
