from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


def pair_key(x: int, y: int) -> str:
    return f"{int(x)},{int(y)}"


def _kind(dep_type: Any) -> str:
    text = str(dep_type or '')
    return 'indep' if 'indep' in text else 'dep'


def _truth_label(value: Any) -> str:
    text = str(value or 'unknown').strip().lower()
    if text in {'true', 'false', 'unknown'}:
        return text
    return 'unknown'


def _clean_counter(counter: Counter[str]) -> dict[str, int]:
    return {k: int(v) for k, v in sorted(counter.items())}


def build_fact_profile(facts: Iterable[tuple], remove_n: int) -> dict[str, Any]:
    facts_list = list(facts)
    total = len(facts_list)
    remove_n = max(0, min(int(remove_n or 0), total))
    removed = facts_list[-remove_n:] if remove_n else []
    kept = facts_list[:-remove_n] if remove_n else list(facts_list)

    removed_kind = Counter(_kind(f[3]) for f in removed)
    kept_kind = Counter(_kind(f[3]) for f in kept)
    removed_truth = Counter(_truth_label(f[6] if len(f) > 6 else 'unknown') for f in removed)
    kept_truth = Counter(_truth_label(f[6] if len(f) > 6 else 'unknown') for f in kept)

    total_pair_counts: Counter[str] = Counter()
    removed_pair_counts: Counter[str] = Counter()
    for fact in facts_list:
        total_pair_counts[pair_key(fact[0], fact[2])] += 1
    for fact in removed:
        removed_pair_counts[pair_key(fact[0], fact[2])] += 1

    fully_released_pairs = sorted(
        pair
        for pair, count in removed_pair_counts.items()
        if count >= int(total_pair_counts.get(pair, 0) or 0) and int(total_pair_counts.get(pair, 0) or 0) > 0
    )

    return {
        'facts_total': int(total),
        'removed_fact_count': int(len(removed)),
        'kept_fact_count': int(len(kept)),
        'removed_fact_keys': [str(f[4]).strip() for f in removed if str(f[4]).strip()],
        'removed_true_fact_keys': [str(f[4]).strip() for f in removed if _truth_label(f[6] if len(f) > 6 else 'unknown') == 'true'],
        'removed_false_fact_keys': [str(f[4]).strip() for f in removed if _truth_label(f[6] if len(f) > 6 else 'unknown') == 'false'],
        'removed_unknown_fact_keys': [str(f[4]).strip() for f in removed if _truth_label(f[6] if len(f) > 6 else 'unknown') == 'unknown'],
        'removed_indep_count': int(removed_kind.get('indep', 0)),
        'removed_dep_count': int(removed_kind.get('dep', 0)),
        'kept_indep_count': int(kept_kind.get('indep', 0)),
        'kept_dep_count': int(kept_kind.get('dep', 0)),
        'removed_truth_counts': _clean_counter(removed_truth),
        'kept_truth_counts': _clean_counter(kept_truth),
        'fully_released_pair_count': int(len(fully_released_pairs)),
        'fully_released_pairs': fully_released_pairs,
        'removed_pair_counts': _clean_counter(removed_pair_counts),
        'pair_fact_counts_total': _clean_counter(total_pair_counts),
    }
