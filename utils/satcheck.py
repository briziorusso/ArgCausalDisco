from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Literal, cast

try:
    from . import mem as _mem
except ImportError:  # pragma: no cover
    from utils import mem as _mem


SolveStatus = Literal["sat", "unsat", "unknown"]
_GAP_AWARE_UNKNOWN_THRESHOLD = 2
_FRONTIER_UNKNOWN_THRESHOLD = 4


def solve_with_timeout(
    ctl,
    *,
    solve_timeout: float | None,
    on_model,
    assumptions: Any = None,
) -> tuple[bool, Any | None]:
    """Run clingo solve with an optional hard wall-time limit."""
    if solve_timeout is None:
        with ctl.solve(yield_=True, assumptions=assumptions or []) as handle:
            for model in handle:
                on_model(model)
            try:
                return True, handle.get()
            except Exception:
                return True, None

    handle = ctl.solve(async_=True, on_model=on_model, assumptions=assumptions or [])
    finished = handle.wait(solve_timeout)
    if not finished:
        try:
            handle.cancel()
        except Exception:
            pass
        result = None
        try:
            if bool(handle.wait(1.0)):
                result = handle.get()
        except TypeError:
            pass
        except Exception:
            result = None
        return False, result
    try:
        return True, handle.get()
    except Exception:
        return True, None


def classify_solve_result(*, found_sat: bool, finished: bool, result: Any | None) -> SolveStatus:
    if found_sat:
        return "sat"
    if not finished:
        if bool(getattr(result, "satisfiable", False)):
            return "sat"
        return "unknown"

    satisfiable = getattr(result, "satisfiable", None)
    if satisfiable is True:
        return "sat"
    if bool(getattr(result, "unsatisfiable", False)):
        return "unsat"
    if bool(getattr(result, "unknown", False)):
        return "unknown"
    if bool(getattr(result, "interrupted", False)):
        return "unknown"
    if satisfiable is False:
        return "unsat"
    return "unknown"


def build_probe_candidates(lo: int, hi: int, mid: int, limit: int) -> list[int]:
    """Generate alternate removal counts after a timed-out midpoint satcheck."""
    limit = max(0, int(limit))
    if limit == 0 or lo > hi:
        return []

    candidates: list[int] = []
    seen: set[int] = {mid}

    def _add(value: int) -> None:
        if value < lo or value > hi or value in seen:
            return
        seen.add(value)
        candidates.append(value)

    # Prefer interior probes that can shrink the bracket materially before
    # falling back to boundary probes. This avoids sequences where repeated
    # UNKNOWN midpoints only advance `lo` one step at a time.
    upper_lo, upper_hi = mid + 1, hi - 1
    lower_lo, lower_hi = lo + 1, mid - 1
    while len(candidates) < limit and (upper_lo <= upper_hi or lower_lo <= lower_hi):
        if upper_lo <= upper_hi:
            cand = (upper_lo + upper_hi + 1) // 2
            _add(cand)
            upper_lo = cand + 1
        if len(candidates) >= limit:
            break
        if lower_lo <= lower_hi:
            cand = (lower_lo + lower_hi) // 2
            _add(cand)
            lower_hi = cand - 1

    _add(hi)
    _add(lo)

    return candidates[:limit]


def _gap_candidates_from_tested_points(
    *,
    lo: int,
    hi: int,
    tested_points: set[int],
) -> list[tuple[int, int, int, int]]:
    if lo > hi:
        return []
    anchors = sorted({lo - 1, hi, *[p for p in tested_points if lo <= p <= hi]})
    candidates: list[tuple[int, int, int, int]] = []
    for left, right in zip(anchors, anchors[1:]):
        width = int(right - left)
        if width <= 1:
            continue
        cand = (left + right) // 2
        if cand < lo or cand > hi or cand in tested_points:
            continue
        candidates.append((width, cand, left, right))
    return candidates


def satcheck_peak_kib(
    *,
    rss_before_kib: int | None,
    rss_after_kib: int | None,
    hwm_before_kib: int | None,
    hwm_after_kib: int | None,
) -> int | None:
    peak = None
    for value in (rss_before_kib, rss_after_kib):
        if value is None:
            continue
        peak = value if peak is None else max(peak, value)
    if hwm_after_kib is not None and (hwm_before_kib is None or hwm_after_kib > hwm_before_kib):
        peak = hwm_after_kib if peak is None else max(peak, hwm_after_kib)
    return peak


def build_search_fingerprint(
    *,
    n_nodes: int,
    fact_keys: list[str],
    max_path_length: int | None,
    max_conditioning_size: int | None,
    collider_tree_depth: int | None,
    cycle_length: int | None,
    prior_forbidden: list[tuple[int, int]] | None = None,
    prior_required: list[tuple[int, int]] | None = None,
) -> str:
    payload = {
        "n_nodes": int(n_nodes),
        "facts": list(fact_keys),
        "max_path_length": max_path_length,
        "max_conditioning_size": max_conditioning_size,
        "collider_tree_depth": collider_tree_depth,
        "cycle_length": cycle_length,
        "prior_forbidden": sorted(list(prior_forbidden or [])),
        "prior_required": sorted(list(prior_required or [])),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _clamp_int(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


def _autotune_state_path(facts_location: str) -> Path | None:
    if not facts_location:
        return None
    try:
        return Path(facts_location).resolve().parent / "satcheck_autotune.json"
    except Exception:
        return None


def _search_state_path(facts_location: str) -> Path | None:
    if not facts_location:
        return None
    try:
        return Path(facts_location).resolve().parent / "satcheck_search.json"
    except Exception:
        return None


def _load_autotune_state(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_autotune_state(path: Path | None, state: dict[str, Any], logger: logging.Logger) -> None:
    if path is None:
        return
    tmp_path = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(state, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        logger.debug("Failed to persist satcheck autotune state to %s", path, exc_info=True)
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


def _recommend_threads(
    current_threads: int,
    *,
    timed_out: bool,
    peak_kib: int | None,
    available_mem_kib: int | None,
    total_mem_kib: int | None,
    min_threads: int,
    max_threads: int,
    increase_step: int,
) -> tuple[int, str]:
    current_threads = _clamp_int(current_threads, min_threads, max_threads)
    if min_threads >= max_threads:
        return current_threads, "fixed"

    peak_ratio = None
    avail_ratio = None
    if total_mem_kib and total_mem_kib > 0:
        if peak_kib is not None:
            peak_ratio = float(peak_kib) / float(total_mem_kib)
        if available_mem_kib is not None:
            avail_ratio = float(available_mem_kib) / float(total_mem_kib)

    if timed_out:
        if (
            peak_ratio is not None
            and peak_ratio <= 0.35
            and (avail_ratio is None or avail_ratio >= 0.40)
            and current_threads < max_threads
        ):
            return _clamp_int(current_threads + max(1, int(increase_step)), min_threads, max_threads), "timeout_low_mem_increase"
        if (
            (peak_ratio is not None and peak_ratio >= 0.80)
            or (avail_ratio is not None and avail_ratio <= 0.15)
        ):
            reduced = max(min_threads, int(current_threads * 0.75))
            if reduced < current_threads:
                return reduced, "timeout_high_mem_reduce"
        return current_threads, "timeout_keep"

    if (
        (peak_ratio is not None and peak_ratio >= 0.90)
        or (avail_ratio is not None and avail_ratio <= 0.10)
    ):
        reduced = max(min_threads, int(current_threads * 0.75))
        if reduced < current_threads:
            return reduced, "high_mem_reduce"
    return current_threads, "keep"


class SatcheckThreadTuner:
    def __init__(
        self,
        *,
        facts_location: str,
        max_threads: int,
        initial_threads: int | None,
        adaptive: bool,
        min_threads: int = 1,
        increase_step: int = 2,
        logger: logging.Logger | None = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self.max_threads = max(1, int(max_threads))
        self.min_threads = max(1, min(self.max_threads, int(min_threads or 1)))
        base_threads = self.max_threads if initial_threads is None else int(initial_threads)
        self.current_threads = _clamp_int(base_threads, self.min_threads, self.max_threads)
        self.increase_step = max(1, int(increase_step or 1))
        self.adaptive = bool(adaptive)
        self.total_mem_kib = _mem.memtotal_kib()
        self.state_path = _autotune_state_path(facts_location)
        self.state: dict[str, Any] = _load_autotune_state(self.state_path) if self.adaptive else {}
        if self.adaptive:
            self._restore()

    @property
    def state_path_str(self) -> str | None:
        return str(self.state_path) if self.state_path is not None else None

    def _restore(self) -> None:
        if not self.state:
            return
        try:
            if bool(self.state.get("inflight")):
                prev_threads = _clamp_int(
                    int(self.state.get("last_started_threads", self.current_threads)),
                    self.min_threads,
                    self.max_threads,
                )
                recovered_threads = max(self.min_threads, prev_threads // 2)
                self.current_threads = min(self.current_threads, recovered_threads)
                self._logger.warning(
                    "[autotune] previous run ended during a satcheck; reducing satcheck threads %d->%d",
                    prev_threads,
                    self.current_threads,
                )
                self.state.update(
                    {
                        "inflight": False,
                        "recommended_threads": self.current_threads,
                        "last_event": "recover_inflight_reduce",
                        "updated_at": time.time(),
                    }
                )
                _write_autotune_state(self.state_path, self.state, self._logger)
                return

            prev_recommended = self.state.get("recommended_threads")
            if prev_recommended is not None:
                self.current_threads = _clamp_int(int(prev_recommended), self.min_threads, self.max_threads)
                self._logger.info(
                    "[autotune] restored satcheck thread recommendation=%d from %s",
                    self.current_threads,
                    self.state_path,
                )
        except Exception:
            self._logger.debug("Failed to restore satcheck autotune state", exc_info=True)

    def _persist(
        self,
        *,
        inflight: bool,
        current_threads: int,
        recommended_threads: int | None = None,
        removed: int | None = None,
        status: str | None = None,
        timed_out: bool | None = None,
        rss_kib: int | None = None,
        peak_kib: int | None = None,
        available_mem_kib: int | None = None,
        event: str | None = None,
    ) -> None:
        if not self.adaptive or self.state_path is None:
            return
        state = dict(self.state or {})
        state["version"] = 1
        state["updated_at"] = time.time()
        state["inflight"] = bool(inflight)
        state["last_started_threads"] = int(current_threads)
        state["max_threads"] = int(self.max_threads)
        state["min_threads"] = int(self.min_threads)
        if recommended_threads is not None:
            state["recommended_threads"] = int(recommended_threads)
        elif "recommended_threads" not in state:
            state["recommended_threads"] = int(current_threads)
        if removed is not None:
            state["last_removed"] = int(removed)
        if status is not None:
            state["last_status"] = str(status)
        if timed_out is not None:
            state["last_timed_out"] = bool(timed_out)
        if rss_kib is not None:
            state["last_rss_kib"] = int(rss_kib)
        if peak_kib is not None:
            state["last_peak_kib"] = int(peak_kib)
        if available_mem_kib is not None:
            state["last_available_mem_kib"] = int(available_mem_kib)
        if self.total_mem_kib is not None:
            state["total_mem_kib"] = int(self.total_mem_kib)
        if event is not None:
            state["last_event"] = str(event)
        self.state = state
        _write_autotune_state(self.state_path, state, self._logger)

    def mark_start(self, *, removed: int) -> None:
        self._persist(
            inflight=True,
            current_threads=self.current_threads,
            recommended_threads=self.current_threads,
            removed=removed,
            event="start",
        )

    def finish(
        self,
        *,
        removed: int,
        status: SolveStatus,
        timed_out: bool,
        rss_kib: int | None,
        peak_kib: int | None,
        available_mem_kib: int | None,
    ) -> dict[str, Any]:
        previous_threads = int(self.current_threads)
        event = "done"
        if self.adaptive:
            next_threads, event = _recommend_threads(
                previous_threads,
                timed_out=timed_out,
                peak_kib=peak_kib,
                available_mem_kib=available_mem_kib,
                total_mem_kib=self.total_mem_kib,
                min_threads=self.min_threads,
                max_threads=self.max_threads,
                increase_step=self.increase_step,
            )
            self.current_threads = int(next_threads)

        self._persist(
            inflight=False,
            current_threads=previous_threads,
            recommended_threads=self.current_threads,
            removed=removed,
            status=status,
            timed_out=timed_out,
            rss_kib=rss_kib,
            peak_kib=peak_kib,
            available_mem_kib=available_mem_kib,
            event=event,
        )
        return {
            "previous_threads": previous_threads,
            "current_threads": int(self.current_threads),
            "changed": int(self.current_threads) != previous_threads,
            "event": event,
        }


class SatcheckSearchCheckpoint:
    def __init__(
        self,
        *,
        facts_location: str,
        fingerprint: str,
        satcheck_timeout: float | None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self.path = _search_state_path(facts_location)
        self.fingerprint = str(fingerprint)
        self.satcheck_timeout = None if satcheck_timeout is None else float(satcheck_timeout)
        self.cache: dict[int, dict[str, Any]] = {}
        self.records: list[dict[str, Any]] = []
        self.best_sat_removed: int | None = None
        self.best_unsat_removed: int | None = None
        self.approximate_remove_search = False
        self.search_complete = False
        self.remove_n: int | None = None
        self.loaded = False
        self.timeout_mismatch = False
        self.dropped_unknown_records = 0
        self._load()

    def _normalize_timeout(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _recompute_bounds(self) -> None:
        sat_vals = [removed for removed, entry in self.cache.items() if entry.get("status") == "sat"]
        unsat_vals = [removed for removed, entry in self.cache.items() if entry.get("status") == "unsat"]
        self.best_sat_removed = min(sat_vals) if sat_vals else None
        self.best_unsat_removed = max(unsat_vals) if unsat_vals else None

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
        except Exception:
            self._logger.debug("Failed to load satcheck search checkpoint from %s", self.path, exc_info=True)
            return

        if not isinstance(data, dict):
            return
        stored_fingerprint = str(data.get("fingerprint", ""))
        if stored_fingerprint != self.fingerprint:
            self._logger.warning(
                "[remove-search] checkpoint fingerprint mismatch; ignoring stored search state from %s (stored=%s current=%s)",
                self.path,
                stored_fingerprint[:12],
                self.fingerprint[:12],
            )
            return

        self.loaded = True
        self.timeout_mismatch = self._normalize_timeout(data.get("satcheck_timeout")) != self.satcheck_timeout

        raw_cache = data.get("cache", {})
        if isinstance(raw_cache, dict):
            for removed_key, entry in raw_cache.items():
                if not isinstance(entry, dict):
                    continue
                try:
                    removed = int(removed_key)
                except Exception:
                    continue
                status = str(entry.get("status", "")).lower()
                if status not in ("sat", "unsat", "unknown"):
                    continue
                if self.timeout_mismatch and status == "unknown":
                    self.dropped_unknown_records += 1
                    continue
                self.cache[removed] = {
                    "status": status,
                    "sec": float(entry.get("sec", 0.0) or 0.0),
                    "label": str(entry.get("label", "")),
                }

        raw_records = data.get("satcheck_records", [])
        if isinstance(raw_records, list):
            for record in raw_records:
                if not isinstance(record, dict):
                    continue
                status = str(record.get("status", "")).lower()
                if status not in ("sat", "unsat", "unknown"):
                    continue
                if self.timeout_mismatch and status == "unknown":
                    self.dropped_unknown_records += 1
                    continue
                try:
                    cleaned = dict(record)
                    cleaned["removed"] = int(cleaned.get("removed"))
                    cleaned["lo"] = int(cleaned.get("lo"))
                    cleaned["hi"] = int(cleaned.get("hi"))
                    cleaned["mid"] = int(cleaned.get("mid"))
                    cleaned["call"] = int(cleaned.get("call"))
                    cleaned["sec"] = float(cleaned.get("sec", 0.0) or 0.0)
                    cleaned["status"] = status
                    cleaned["label"] = str(cleaned.get("label", ""))
                except Exception:
                    continue
                self.records.append(cleaned)

        restored_approximate = bool(data.get("approximate_remove_search", False))
        self.approximate_remove_search = restored_approximate
        self.search_complete = bool(data.get("search_complete", False))
        try:
            self.remove_n = int(data["remove_n"]) if data.get("remove_n") is not None else None
        except Exception:
            self.remove_n = None

        self._recompute_bounds()

        if self.timeout_mismatch:
            if restored_approximate:
                self.search_complete = False
                self.remove_n = None
            self.approximate_remove_search = any(entry.get("status") == "unknown" for entry in self.cache.values())

    def summary(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "cache_entries": len(self.cache),
            "best_sat_removed": self.best_sat_removed,
            "best_unsat_removed": self.best_unsat_removed,
            "search_complete": self.search_complete,
            "remove_n": self.remove_n,
            "approximate_remove_search": self.approximate_remove_search,
            "timeout_mismatch": self.timeout_mismatch,
            "dropped_unknown_records": self.dropped_unknown_records,
            "path": str(self.path) if self.path is not None else None,
        }

    def has_resume_state(self) -> bool:
        return bool(
            self.loaded
            and (
                self.search_complete
                or self.records
                or self.best_sat_removed is not None
                or self.best_unsat_removed is not None
            )
        )

    def _persist(self) -> None:
        if self.path is None:
            return
        data = {
            "version": 1,
            "updated_at": time.time(),
            "fingerprint": self.fingerprint,
            "satcheck_timeout": self.satcheck_timeout,
            "cache": {str(k): v for k, v in sorted(self.cache.items())},
            "satcheck_records": self.records,
            "best_sat_removed": self.best_sat_removed,
            "best_unsat_removed": self.best_unsat_removed,
            "approximate_remove_search": bool(self.approximate_remove_search),
            "search_complete": bool(self.search_complete),
            "remove_n": self.remove_n,
        }
        _write_autotune_state(self.path, data, self._logger)

    def record(
        self,
        *,
        removed: int,
        status: SolveStatus,
        sec: float,
        label: str,
        lo: int,
        hi: int,
        mid: int,
        call: int,
    ) -> None:
        removed = int(removed)
        status = status.lower()
        self.cache[removed] = {
            "status": status,
            "sec": float(sec),
            "label": str(label),
        }
        self.records.append(
            {
                "call": int(call),
                "lo": int(lo),
                "hi": int(hi),
                "mid": int(mid),
                "removed": int(removed),
                "status": status,
                "sec": float(sec),
                "label": str(label),
            }
        )
        if status == "unknown":
            self.approximate_remove_search = True
        self._recompute_bounds()
        self._persist()

    def mark_search_complete(self, *, remove_n: int, approximate_remove_search: bool) -> None:
        self.search_complete = True
        self.remove_n = int(remove_n)
        self.approximate_remove_search = bool(approximate_remove_search)
        self._persist()


@dataclass
class RemoveSearchResult:
    remove_n: int
    satcheck_records: list[dict[str, Any]]
    best_sat_removed: int | None
    best_unsat_removed: int | None
    approximate_remove_search: bool


def run_remove_search(
    *,
    total_facts: int,
    checkpoint: SatcheckSearchCheckpoint,
    satcheck_probe_limit: int,
    logger: logging.Logger,
    solve_removed: Callable[[int], SolveStatus],
) -> RemoveSearchResult:
    total_facts = max(0, int(total_facts))
    probe_limit = max(0, int(satcheck_probe_limit or 0))
    restored_summary = checkpoint.summary()
    satcheck_records: list[dict[str, Any]] = list(checkpoint.records)
    best_sat_removed: int | None = checkpoint.best_sat_removed
    best_unsat_removed: int | None = checkpoint.best_unsat_removed
    approximate_remove_search = bool(checkpoint.approximate_remove_search)
    lo = max(0, int(best_unsat_removed) + 1) if best_unsat_removed is not None else 0
    hi = min(total_facts, int(best_sat_removed)) if best_sat_removed is not None else total_facts

    try:
        import math

        satcheck_baseline = 1 + int(math.ceil(math.log2(max(1, hi) + 1)))
    except Exception:
        satcheck_baseline = None

    if restored_summary["loaded"]:
        logger.info(
            "[remove-search] restored checkpoint cache=%d best_unsat=%s best_sat=%s complete=%s remove_n=%s approximate=%s timeout_mismatch=%s dropped_unknown=%s",
            restored_summary["cache_entries"],
            restored_summary["best_unsat_removed"],
            restored_summary["best_sat_removed"],
            restored_summary["search_complete"],
            restored_summary["remove_n"],
            restored_summary["approximate_remove_search"],
            restored_summary["timeout_mismatch"],
            restored_summary["dropped_unknown_records"],
        )

    satcheck_cache: dict[int, dict[str, Any]] = dict(checkpoint.cache)
    satcheck_call_count = 0

    def _unknown_points(lo_now: int, hi_now: int) -> list[int]:
        return sorted(
            removed
            for removed, entry in satcheck_cache.items()
            if lo_now <= removed <= hi_now and entry.get("status") == "unknown"
        )

    def _choose_next_candidate(lo_now: int, hi_now: int) -> tuple[int, str]:
        default_mid = (lo_now + hi_now) // 2
        if default_mid not in satcheck_cache:
            fallback_candidate = default_mid
        else:
            fallback_candidate = -1

        unknowns = _unknown_points(lo_now, hi_now)
        tested_points = {removed for removed in satcheck_cache if lo_now <= removed <= hi_now}
        gap_infos = _gap_candidates_from_tested_points(
            lo=lo_now,
            hi=hi_now,
            tested_points=tested_points,
        )
        gap_infos.sort(key=lambda item: (item[0], item[1]), reverse=True)

        if len(unknowns) >= _FRONTIER_UNKNOWN_THRESHOLD and gap_infos:
            upper_half = [info for info in gap_infos if info[1] > default_mid]
            frontier_infos = upper_half if upper_half else gap_infos
            frontier_infos.sort(key=lambda item: (item[0], item[1]), reverse=True)
            frontier_candidate = frontier_infos[0][1]
            logger.info(
                "[remove-search] frontier-aware candidate=%d unknowns=%d lo=%d hi=%d",
                frontier_candidate,
                len(unknowns),
                lo_now,
                hi_now,
            )
            return frontier_candidate, "mid-frontier"

        if len(unknowns) >= _GAP_AWARE_UNKNOWN_THRESHOLD and gap_infos:
            gap_candidate = gap_infos[0][1]
            logger.info(
                "[remove-search] gap-aware candidate=%d unknowns=%d lo=%d hi=%d",
                gap_candidate,
                len(unknowns),
                lo_now,
                hi_now,
            )
            return gap_candidate, "mid-gap"

        if fallback_candidate >= 0:
            return fallback_candidate, "mid"

        for _, cand, _, _ in gap_infos:
            return cand, "mid-gap"

        return default_mid, "mid"

    def _run_satcheck(removed: int, *, lo_now: int, hi_now: int, mid_now: int, label: str) -> SolveStatus:
        nonlocal satcheck_call_count, best_sat_removed, best_unsat_removed, approximate_remove_search
        removed = max(0, min(total_facts, int(removed)))
        cached = satcheck_cache.get(removed)
        if cached is not None:
            return cast(SolveStatus, cached["status"])

        satcheck_call_count += 1
        logger.info(
            "[satcheck] call=%d baseline=%s lo=%d hi=%d mid=%d removed=%d label=%s",
            satcheck_call_count,
            satcheck_baseline if satcheck_baseline is not None else "?",
            lo_now,
            hi_now,
            mid_now,
            removed,
            label,
        )
        t0 = time.perf_counter()
        status = solve_removed(removed)
        t1 = time.perf_counter()
        satcheck_cache[removed] = {
            "status": status,
            "sec": float(t1 - t0),
            "label": label,
        }
        satcheck_records.append(
            {
                "call": int(satcheck_call_count),
                "lo": int(lo_now),
                "hi": int(hi_now),
                "mid": int(mid_now),
                "removed": int(removed),
                "status": status,
                "sec": float(t1 - t0),
                "label": label,
            }
        )
        checkpoint.record(
            removed=removed,
            status=status,
            sec=float(t1 - t0),
            label=label,
            lo=lo_now,
            hi=hi_now,
            mid=mid_now,
            call=int(satcheck_call_count),
        )
        logger.info(
            "[satcheck] result=%s sec=%.3f lo=%d hi=%d mid=%d removed=%d",
            status.upper(),
            t1 - t0,
            lo_now,
            hi_now,
            mid_now,
            removed,
        )
        if status == "sat":
            best_sat_removed = removed if best_sat_removed is None else min(best_sat_removed, removed)
        elif status == "unsat":
            best_unsat_removed = removed if best_unsat_removed is None else max(best_unsat_removed, removed)
        else:
            approximate_remove_search = True
        return status

    def _probe_unknown(lo_now: int, hi_now: int, mid_now: int) -> tuple[int, SolveStatus] | None:
        for cand in build_probe_candidates(lo_now, hi_now, mid_now, probe_limit):
            status = _run_satcheck(
                cand,
                lo_now=lo_now,
                hi_now=hi_now,
                mid_now=mid_now,
                label="probe",
            )
            if status == "sat" and cand < hi_now:
                return cand, status
            if status == "unsat" and cand >= lo_now:
                return cand, status
        return None

    if checkpoint.search_complete and checkpoint.remove_n is not None:
        remove_n = int(checkpoint.remove_n)
        logger.info("[remove-search] reusing completed checkpoint remove_n=%d", remove_n)
        return RemoveSearchResult(
            remove_n=remove_n,
            satcheck_records=satcheck_records,
            best_sat_removed=best_sat_removed,
            best_unsat_removed=best_unsat_removed,
            approximate_remove_search=approximate_remove_search,
        )

    if best_sat_removed == 0:
        remove_n = 0
    else:
        if 0 not in satcheck_cache:
            sat0 = _run_satcheck(0, lo_now=lo, hi_now=hi, mid_now=0, label="initial")
            if sat0 == "sat":
                remove_n = 0
            else:
                if sat0 == "unsat":
                    lo = max(lo, 1)
                remove_n = None
        else:
            remove_n = None

        while remove_n is None and lo < hi:
            mid, mid_label = _choose_next_candidate(lo, hi)
            status = _run_satcheck(mid, lo_now=lo, hi_now=hi, mid_now=mid, label=mid_label)
            if status == "sat":
                hi = mid
                best_sat_removed = hi if best_sat_removed is None else min(best_sat_removed, hi)
                continue
            if status == "unsat":
                lo = mid + 1
                best_unsat_removed = lo - 1 if best_unsat_removed is None else max(best_unsat_removed, lo - 1)
                continue

            probe_result = _probe_unknown(lo, hi, mid)
            if probe_result is not None:
                probe_removed, probe_status = probe_result
                if probe_status == "sat":
                    hi = min(hi, probe_removed)
                    best_sat_removed = hi if best_sat_removed is None else min(best_sat_removed, hi)
                else:
                    lo = max(lo, probe_removed + 1)
                    best_unsat_removed = lo - 1 if best_unsat_removed is None else max(best_unsat_removed, lo - 1)
                continue

            logger.warning(
                "[remove-search] unresolved satcheck window lo=%d hi=%d mid=%d; "
                "falling back to best-known bracket (best_sat=%s best_unsat=%s)",
                lo,
                hi,
                mid,
                best_sat_removed,
                best_unsat_removed,
            )
            approximate_remove_search = True
            break

        if remove_n is None:
            if best_sat_removed is not None:
                remove_n = int(best_sat_removed)
            elif approximate_remove_search:
                remove_n = int(hi)
            elif best_unsat_removed is not None:
                remove_n = min(total_facts, int(best_unsat_removed) + 1)
            else:
                remove_n = int(hi)

    checkpoint.best_sat_removed = best_sat_removed
    checkpoint.best_unsat_removed = best_unsat_removed
    checkpoint.mark_search_complete(
        remove_n=int(remove_n),
        approximate_remove_search=bool(approximate_remove_search),
    )
    return RemoveSearchResult(
        remove_n=int(remove_n),
        satcheck_records=satcheck_records,
        best_sat_removed=best_sat_removed,
        best_unsat_removed=best_unsat_removed,
        approximate_remove_search=bool(approximate_remove_search),
    )
