import os
import time
import threading
from pathlib import Path
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
    status_path: str | Path | None = None,
    log_every_beat: bool = False,
):
    """Emit a periodic heartbeat while a long-running phase is active.

    By default, heartbeats overwrite a status file instead of appending to the
    main log on every interval. Returns (stop_event, thread) or (None, None) if
    disabled.
    """
    if interval_sec <= 0:
        return None, None

    stop = threading.Event()
    started = time.time()
    started_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started))
    status_path_obj = Path(status_path) if status_path is not None else None

    def _snapshot_text() -> str:
        elapsed = time.time() - started
        extra = f" {describe()}" if describe is not None else ""
        return f"[hb] st={started_str} phase={phase} elapsed={fmt_hhmmss(elapsed)}{extra}"

    def _write_status_snapshot() -> None:
        if status_path_obj is None:
            return
        tmp_path = status_path_obj.with_suffix(status_path_obj.suffix + ".tmp")
        status_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "w") as f:
            f.write(_snapshot_text())
            f.write("\n")
        os.replace(tmp_path, status_path_obj)

    try:
        _write_status_snapshot()
    except Exception:
        pass

    def _run():
        # Wait first so fast phases don't print.
        while not stop.wait(interval_sec):
            try:
                _write_status_snapshot()
                if log_every_beat:
                    logger.info("%s", _snapshot_text())
            except Exception:
                # Best-effort; never break the solver.
                pass

    t = threading.Thread(target=_run, name=f"heartbeat:{phase}", daemon=True)
    t.start()
    return stop, t
