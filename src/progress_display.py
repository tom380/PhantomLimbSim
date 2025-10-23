import math
import sys
import time


class SimulationProgress:
    def __init__(self, total_time, timestep, bar_length=30):
        self.total_time = max(total_time, 0)
        self.timestep = timestep if timestep and timestep > 0 else None
        self.bar_length = bar_length
        self.total_steps = self._compute_total_steps()
        self.start_time = None
        self.last_line_len = 0

    def _compute_total_steps(self):
        if not self.timestep or not self.total_time:
            return 1
        return max(1, math.ceil(self.total_time / self.timestep))

    def start(self):
        if self.start_time is not None:
            return
        self.start_time = time.perf_counter()
        # legend_lines = [
        #     "Progress bar fields:",
        #     "  |####| -> simulation time progress",
        #     "  [curr/total] -> completed timesteps",
        #     "  t= -> wall-clock runtime since start",
        #     "  gait= -> gait cycle percentage and phase",
        #     "  step= -> compute time per iteration (excl. sleep)",
        # ]
        # print("\n".join(legend_lines))

    def _format_elapsed(self, seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        if minutes:
            return f"{minutes}m{secs:02d}s"
        return f"{secs}s"

    def _write_line(self, line):
        padding = ""
        if self.last_line_len > len(line):
            padding = " " * (self.last_line_len - len(line))
        sys.stdout.write("\r" + line + padding)
        sys.stdout.flush()
        self.last_line_len = len(line)

    def update(self, sim_time, step_count, gait_phase, phase_label, step_duration):
        if self.start_time is None:
            self.start()

        progress_fraction = 0.0
        if self.total_time:
            progress_fraction = min(max(sim_time / self.total_time, 0.0), 1.0)

        filled = int(round(progress_fraction * self.bar_length))
        filled = min(filled, self.bar_length)
        bar = "#" * filled + "-" * (self.bar_length - filled)
        progress_percent = progress_fraction * 100.0

        steps_display = min(step_count, self.total_steps)
        elapsed_seconds = time.perf_counter() - self.start_time
        elapsed_str = self._format_elapsed(elapsed_seconds)

        if gait_phase is not None:
            gait_percent = f"{gait_phase * 100:.2f}%"
        else:
            gait_percent = "--.--%"

        phase_info = phase_label or "-----"

        if step_duration is None:
            step_str = "--.--ms"
        else:
            step_str = f"{step_duration * 1000:.2f}ms"

        line = (
            f"|{bar}| {progress_percent:6.2f}% "
            f"[{steps_display}/{self.total_steps}] "
            f"t={elapsed_str} "
            f"gait={gait_percent} ({phase_info}) "
            f"step={step_str}"
        )
        self._write_line(line)

    def finish(self, sim_time, step_count, gait_phase, phase_label, step_duration):
        self.update(sim_time, step_count, gait_phase, phase_label, step_duration)
        sys.stdout.write("\n")
        sys.stdout.flush()
