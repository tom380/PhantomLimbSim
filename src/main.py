from args import parse_args
import record
import run


def main():
    # Parse command line arguments
    args = parse_args()

    logs, (frames, fps) = run.sim(
        model_path=args.model,
        actuated= not args.unactuated,
        record_video=args.record_video,
        record_force=args.record_force,
        kinematics_mode=args.kinematics,
        spring_index=args.spring_index,
        sim_time=args.sim_time,
    )

    ## TODO: File save location and names should be configurable
    if args.record_force:
        if args.output is None:
            # ---- Plot last cycle ----
            record.plot_last_cycle(logs)
            # ---- Save full dataset to MATLAB .mat ----
            record.save_mat(logs)
        else:
            # ---- Plot last cycle ----
            record.plot_last_cycle(logs, args.output)
            # ---- Save full dataset to MATLAB .mat ----
            record.save_mat(logs, args.output)

    if args.record_video:
        if args.output is None:
            record.save_video(frames, fps)
        else:
            record.save_video(frames, fps, args.output)


if __name__ == "__main__":
    main()
