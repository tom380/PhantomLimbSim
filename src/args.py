import argparse

def parse_args():
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
        "--record-force",
        action="store_true",
        help="Record the force measurements during the simulation (default: False)",
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

    return parser.parse_args()
