import time
import threading
from typing import Callable


def fmt_hhmmss(seconds: float) -> str:
    seconds_i = int(max(0.0, float(seconds)))
    h = seconds_i // 3600
    m = (seconds_i % 3600) // 60
    s = seconds_i % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def start_heartbeat(
    logger,
    *,
    phase: str,
    interval_sec: float,
    describe: Callable[[], str] | None = None,
):
    """Emit a periodic INFO heartbeat while a long-running phase is active.

    Intended to keep visibility without bloating logs: use a large interval
    (e.g., 3600s). Returns (stop_event, thread) or (None, None) if disabled.
    """
    if interval_sec <= 0:
        return None, None

    stop = threading.Event()
    started = time.time()

    def _run():
        # Wait first so fast phases don't print.
        while not stop.wait(interval_sec):
            try:
                now = time.strftime("%Y-%m-%d %H:%M:%S")
                elapsed = time.time() - started
                extra = f" {describe()}" if describe is not None else ""
                logger.info("[hb] ts=%s phase=%s elapsed=%s%s", now, phase, fmt_hhmmss(elapsed), extra)
            except Exception:
                # Best-effort; never break the solver.
                pass

    t = threading.Thread(target=_run, name=f"heartbeat:{phase}", daemon=True)
    t.start()
    return stop, t

