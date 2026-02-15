"""Entry point for running simulations and persisting requested outputs."""

from args import parse_args
import record
import run


def main():
    """Parse CLI options, run the simulation, and write outputs."""
    args = parse_args()

    logs, (frames, fps) = run.sim(
        model_path=args.model,
        actuated= not args.unactuated,
        record_video=args.record_video,
        video_fps=args.video_fps,
        video_cameras=args.video_camera,
        record_force=args.record_force,
        record_flex_contact=args.record_flex_contact,
        kinematics_mode=args.kinematics,
        spring_index=args.spring_index,
        sim_time=args.sim_time,
        kinematics_data_dir=args.kinematics_data_dir,
    )

    joint_logs, flex_logs = logs

    # TODO: File save location and names should be configurable.
    if args.record_force:
        output_prefix = args.output
        if joint_logs is None:
            raise RuntimeError("Joint force logging was requested but no joint logs were returned.")
        if output_prefix is None:
            record.plot_last_cycle(joint_logs)
            record.save_mat(joint_logs)
        else:
            record.plot_last_cycle(joint_logs, output_prefix)
            record.save_mat(joint_logs, output_prefix)

    if args.record_flex_contact:
        if flex_logs is None:
            raise RuntimeError("Flex contact logging was requested but no flex logs were returned.")
        record.save_flex_contacts(flex_logs, args.output)

    if args.record_video:
        if args.output is None:
            record.save_video(frames, fps)
        else:
            record.save_video(frames, fps, args.output)


if __name__ == "__main__":
    main()
