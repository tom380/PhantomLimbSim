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
        "--record-video",
        action="store_true",
        help="Record the simulation as a video (default: False)",
    )

    parser.add_argument(
        "--record-force",
        action="store_true",
        help="Record the force measurements during the simulation (default: False)",
    )

    return parser.parse_args()