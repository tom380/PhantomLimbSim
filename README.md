# PhantomLimbSim

This repository provides the simulation code and tooling for a MuJoCo phantom knee-exoskeleton model with deformable soft-tissue interface mechanics. It includes model preprocessing, simulation control, force/contact logging, video export, and post-processing utilities.

In the submitted study, simulated knee moments showed strong agreement with experiments (reported correlations 0.98-0.99), while also enabling spatial interface-force analysis (e.g., pressure hotspot maps) that is difficult to measure directly in physical experiments.

## Thesis Context
- This repository supports the thesis work *Simulating the soft-tissue interface for a knee exoskeleton for improved interaction dynamics during assisted gait*.
- It focuses on modeling soft-tissue interface dynamics in a MuJoCo phantom knee-exoskeleton setup and validating interaction behavior against experiments.
- Link: `https://www.techrxiv.org/users/872607/articles/1380700-simulating-the-soft-tissue-interface-for-a-knee-exoskeleton-for-improved-interaction-dynamics-during-assisted-gait`
- DOI: `10.36227/techrxiv.176972148.89375705/v1`

## What This Repository Does
- Simulates a phantom limb / exosuit setup in MuJoCo using XML models and mesh assets.
- Drives joint motion with configurable kinematics profiles (`average` or `full` dataset mode).
- Supports optional logging of:
  - knee/exo moments and force components (`.mat` + plot)
  - flex contact samples (`.mat`)
  - simulation video (single or multi-camera `.mp4`)
- Includes MATLAB scripts in `matlab/` for post-processing and visualization.

## Repository Layout
- `src/`: simulation code and CLI entrypoint.
- `models/`: MuJoCo model XML files.
- `meshes/`: mesh assets used by models.
- `data/kinematics/`: kinematics datasets (`kinematics.mat`, `knee_angles.mat`).
- `matlab/`: analysis and plotting scripts.
- `post/`: helper scripts for composing views/videos.
- `src/legacy/`: older modules kept for reference, not in active runtime path.

## Requirements
- Python:  3.12.9
- MuJoCo Python bindings: `mujoco==3.3.2` plus manual binary replacement from custom fork `https://github.com/tom380/mujoco`
- Core Python packages:
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `imageio`
  - `meshio`
  - `trimesh`
- Optional: MATLAB for scripts in `matlab/`.

## Important Compatibility Note
- This project depends on a custom MuJoCo binary from `https://github.com/tom380/mujoco` for correct flex/exosuit contact behavior.
- Upstream/default MuJoCo in this setup treats strap collision too coarsely (effectively as a solid/convex body), so interface mechanics are not physically representative.
- The published custom binary is currently available for **Windows x64** via GitHub Releases: `https://github.com/tom380/mujoco/releases`.
- Install the standard Python package (`mujoco==3.3.2`) and then replace the MuJoCo binary in that installation with the released custom build.

## Setup
```bash
# from repository root
# Option A (conda)
conda create -n phantom-limb-sim python=3.12.9 -y
conda activate phantom-limb-sim

pip install --upgrade pip
pip install -r requirements.txt

# print installed mujoco package directory
python3 -c "import mujoco, pathlib; print(pathlib.Path(mujoco.__file__).resolve().parent)"
```

For Windows x64, follow the instructions in the release notes:
- `https://github.com/tom380/mujoco/releases`
- Replace `mujoco.dll` in the printed package directory with the released custom binary.

## Quick Start
Run default simulation:
```bash
python3 src/main.py
```

Run with force logging:
```bash
python3 src/main.py --record-force --output outputs/run_force
```

Run full recorded kinematics:
```bash
python3 src/main.py --kinematics full --sim-time 0 --record-force --output outputs/run_full
```

Run video recording from multiple cameras:
```bash
python3 src/main.py \
  --record-video \
  --video-fps 120 \
  --video-camera fixed_cam \
  --video-camera side_cam \
  --output outputs/run_video
```

Record flex contacts:
```bash
python3 src/main.py --record-flex-contact --output outputs/run_contacts
```

## CLI Reference
Primary options (see `src/args.py`):
- `--model`: MuJoCo XML model path (default: `models/phantom_barrutia.xml`)
- `--output`: output path/prefix
- `--unactuated`: open active viewer without actuation
- `--record-video`: enable video recording
- `--video-fps`: target recording FPS (default `120`)
- `--video-camera`: camera name; repeat flag to record multiple cameras
- `--record-force`: log force/moment quantities
- `--record-flex-contact`: log flex contacts
- `--kinematics`: `average` or `full`
- `--spring-index`: spring dataset index `1..6`
- `--sim-time`: simulation time in seconds (`0` with `--kinematics full` uses dataset length)
- `--kinematics-data-dir`: directory containing kinematics `.mat` files

## Outputs
- Force logs:
  - `<output>.mat`
  - `<output>.png`
- Flex contact logs:
  - `<output>_flex_contacts.mat`
- Video:
  - single camera: `<output>_<camera>.mp4`
  - multiple cameras: one file per selected camera

## Reproducibility Notes
- Default model: `models/phantom_barrutia.xml`
- Default kinematics data: `data/kinematics/`
- Paper uses `--kinematics average`
- Go over each `--spring-index`, 1 through 6

## Known Limitations
- Some parser/surface assumptions are model-specific.
- Flex pinning in `src/xmlparser.py` has a TODO regarding mesh translation handling.
- Plots are broken, use output data to plot instead.

## Contact
- `twjfransen@gmail.com`
