from __future__ import annotations

import os
import resource
from typing import Optional


def _read_proc_meminfo_value_kib(key: str) -> Optional[int]:
    """Read a KiB-valued field from /proc/meminfo (Linux)."""
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if not line.startswith(key + ":"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1])
                return None
    except Exception:
        return None


def _read_proc_status_value_kib(key: str) -> Optional[int]:
    """Read a KiB-valued field from /proc/self/status (Linux)."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if not line.startswith(key + ":"):
                    continue
                # e.g. "VmRSS:\t  123456 kB"
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1])
                return None
    except Exception:
        return None


def rss_kib() -> Optional[int]:
    """Current resident set size (VmRSS) in KiB."""
    return _read_proc_status_value_kib("VmRSS")


def vmsize_kib() -> Optional[int]:
    """Current virtual size (VmSize) in KiB."""
    return _read_proc_status_value_kib("VmSize")


def data_kib() -> Optional[int]:
    """Current data segment size (VmData) in KiB."""
    return _read_proc_status_value_kib("VmData")


def maxrss_kib() -> Optional[int]:
    """Peak RSS (ru_maxrss) in KiB on Linux, None on failure."""
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        v = getattr(ru, "ru_maxrss", None)
        if v is None:
            return None
        # On Linux ru_maxrss is KiB.
        return int(v)
    except Exception:
        return None


def memtotal_kib() -> Optional[int]:
    """Total system memory in KiB."""
    return _read_proc_meminfo_value_kib("MemTotal")


def memavailable_kib() -> Optional[int]:
    """Estimated currently available system memory in KiB."""
    return _read_proc_meminfo_value_kib("MemAvailable")


def fmt_kib(kib: Optional[int]) -> str:
    if kib is None:
        return "n/a"
    gib = kib / (1024.0 * 1024.0)
    return f"{gib:.1f}GiB"


def is_linux_procfs_available() -> bool:
    return os.path.exists("/proc/self/status")
