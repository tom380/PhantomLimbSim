from args import parse_args
import record
import run


def main():
    # Parse command line arguments
    args = parse_args()

    logs, (frames, fps) = run.sim(
        model_path=args.model,
        record_video=args.record_video,
        record_force=args.record_force
    )

    ## TODO: File save location and names should be configurable
    if args.record_force:
        # ---- Plot last cycle ----
        record.plot_last_cycle(logs, "../outputs/simulation_results_last_cycle.png")
        # ---- Save full dataset to MATLAB .mat ----
        record.save_mat(logs, "../outputs/simulation_data.mat")

    if args.record_video:
        record.save_video(frames, fps, "../outputs/run.mp4")


if __name__ == "__main__":
    main()
