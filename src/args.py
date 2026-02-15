"""Command-line argument parsing for phantom limb simulation runs."""

import argparse


def parse_args():
    """Build and parse CLI arguments for simulation and output configuration."""
    parser = argparse.ArgumentParser(
        description="Simulate a phantom limb with a knee angle profile and measure forces."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="models/phantom_barrutia.xml",
        help="Path to the MuJoCo model XML file (default: phantom_barrutia.xml)",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output path and name",
    )

    parser.add_argument(
        "--unactuated",
        action="store_true",
        help="Use active viewer without actuation",
    )

    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record the simulation as a video (default: False)",
    )

    parser.add_argument(
        "--video-fps",
        type=float,
        default=120.0,
        help="Target recording framerate for video capture (default: 120).",
    )

    parser.add_argument(
        "--video-camera",
        action="append",
        default=None,
        help=(
            "Camera name to record. Repeat to capture multiple cameras. "
            "Default: fixed_cam."
        ),
    )

    parser.add_argument(
        "--record-force",
        action="store_true",
        help="Record the force measurements during the simulation (default: False)",
    )

    parser.add_argument(
        "--record-flex-contact",
        action="store_true",
        help="Record raw flex contact locations and forces (default: False)",
    )

    parser.add_argument(
        "--kinematics",
        choices=("average", "full"),
        default="average",
        help="Select the kinematics source: average gait cycle or full recorded dataset (default: average).",
    )

    parser.add_argument(
        "--spring-index",
        type=int,
        choices=range(1, 7),
        default=3,
        help="Choose the exosuit spring dataset (1-6) and corresponding stiffness (default: 3).",
    )

    parser.add_argument(
        "--sim-time",
        type=float,
        default=None,
        help="Simulation time in seconds (default: configuration value). Use 0 with '--kinematics full' to run the entire dataset.",
    )

    parser.add_argument(
        "--kinematics-data-dir",
        type=str,
        default="data/kinematics",
        help="Directory containing kinematics.mat and knee_angles.mat (default: data/kinematics).",
    )
    args = parser.parse_args()
    # Keep legacy behavior when no explicit camera is passed.
    if args.video_camera is None:
        args.video_camera = ["fixed_cam"]
    return args
