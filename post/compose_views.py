#!/usr/bin/env python3
"""
Compose three videos side-by-side.

Defaults:
  left: view2.mp4
  middle: view1.mp4
  right: view3.mp4
  tail: last 2 seconds
  speed: 0.5x (doubles duration)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose three videos side-by-side (left, middle, right)."
    )
    parser.add_argument("--left", default="view2.mp4", help="Left video path.")
    parser.add_argument("--middle", default="view1.mp4", help="Middle video path.")
    parser.add_argument("--right", default="view3.mp4", help="Right video path.")
    parser.add_argument(
        "--output",
        default="views_side_by_side.mp4",
        help="Output video path.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Target height for all clips (default: max input height).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output FPS (default: first available input FPS, else 30).",
    )
    parser.add_argument(
        "--tail-seconds",
        type=float,
        default=2.0,
        help="Seconds from the end of each clip to keep.",
    )
    parser.add_argument(
        "--speed-factor",
        type=float,
        default=0.5,
        help="Playback speed multiplier (0.5 slows to 2x duration).",
    )
    return parser.parse_args()


def main() -> int:
    try:
        from moviepy.editor import VideoFileClip, clips_array
    except Exception:
        try:
            from moviepy import VideoFileClip, clips_array
        except Exception as exc:  # pragma: no cover - import error path
            print(
                "moviepy is required. Install it with: pip install moviepy",
                file=sys.stderr,
            )
            print(f"Import error: {exc}", file=sys.stderr)
            return 2

    args = parse_args()
    inputs = [Path(args.left), Path(args.middle), Path(args.right)]
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        print(f"Missing input files: {', '.join(missing)}", file=sys.stderr)
        return 3

    clips = []
    try:
        clips = [VideoFileClip(str(p)) for p in inputs]
        if hasattr(clips[0], "resize"):
            resize_clip = lambda c, h: c.resize(height=h)
        else:
            resize_clip = lambda c, h: c.resized(height=h)

        if hasattr(clips[0], "subclip"):
            subclip_clip = lambda c, start, end: c.subclip(start, end)
        else:
            subclip_clip = lambda c, start, end: c.subclipped(start, end)

        if hasattr(clips[0], "with_speed_scaled"):
            speed_clip = lambda c, factor: c.with_speed_scaled(factor=factor)
        else:
            try:
                from moviepy import vfx
            except Exception:
                from moviepy.editor import vfx
            speed_clip = lambda c, factor: c.fx(vfx.speedx, factor)

        target_height = args.height or max(c.h for c in clips)
        resized = [resize_clip(c, target_height) for c in clips]

        too_short = [p for p, c in zip(inputs, resized) if c.duration < args.tail_seconds]
        if too_short:
            print(
                "These clips are shorter than --tail-seconds: "
                + ", ".join(str(p) for p in too_short),
                file=sys.stderr,
            )
            return 4

        tailed = [
            subclip_clip(c, c.duration - args.tail_seconds, c.duration)
            for c in resized
        ]
        slowed = [speed_clip(c, args.speed_factor) for c in tailed]
        min_duration = min(c.duration for c in slowed)
        trimmed = [subclip_clip(c, 0, min_duration) for c in slowed]

        final = clips_array([trimmed])
        fps = args.fps
        if fps is None:
            for c in trimmed:
                if getattr(c, "fps", None):
                    fps = c.fps
                    break
        if fps is None:
            fps = 30

        final.write_videofile(
            args.output,
            fps=fps,
            codec="libx264",
            audio=False,
        )
        final.close()
    finally:
        for clip in clips:
            try:
                clip.close()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
