"""Module 6: Evaluation — pure metric computation functions.

All three functions are stateless: same inputs → same outputs, no I/O.

Seeded recall
-------------
recall_at_k(proposals, seeded_ids, labels, k) counts what fraction of
seeded windows appear in the motivating_scene_ids of the top-K accepted
proposals from a single arm. K is pre-registered (default 30).

Rating statistics
-----------------
compute_rating_stats(ratings) returns per-arm mean and bootstrap 95% CI.
CI is suppressed (None) when n_ratings_for_arm < MIN_N_FOR_CI (30).

Krippendorff's alpha
--------------------
krippendorff_alpha(data) computes ordinal Krippendorff's alpha from a
(n_items × n_raters) numpy array with np.nan for missing ratings.
Returns None when fewer than 2 raters provided any overlapping ratings.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from pipeline.interfaces.proposal import ScoredProposal
from pipeline.interfaces.rating import Rating
from pipeline.interfaces.window import WindowKey

# Minimum number of ratings per arm to compute bootstrap CI.
# Below this threshold CI is suppressed and reported as None.
MIN_N_FOR_CI = 30

# Bootstrap samples for CI estimation.
_BOOTSTRAP_SAMPLES = 2000

# Random seed for reproducible bootstrap (does not affect point estimates).
_BOOTSTRAP_SEED = 42


# ---------------------------------------------------------------------------
# Seeded recall
# ---------------------------------------------------------------------------

def compute_seeded_recall(
    proposals: list[ScoredProposal],
    seeded_window_ids: list[WindowKey],
    seeded_subset_labels: dict[WindowKey, Literal["familiar", "unfamiliar"]],
    k: int = 30,
) -> dict[str, dict[str, float]]:
    """Compute recall@k, recall@10, recall@all for overall and both subsets.

    Parameters
    ----------
    proposals
        Accepted proposals from one arm, already ranked (index 0 = rank 1).
        Must have accepted=True to be included; non-accepted proposals are
        skipped regardless of their position.
    seeded_window_ids
        The pre-registered seeded evaluation set.
    seeded_subset_labels
        Maps each seeded WindowKey to "familiar" or "unfamiliar".
        Must cover every key in seeded_window_ids — raises if any are missing.
    k
        Primary recall threshold (pre-registered). Default 30.

    Returns
    -------
    dict with keys "@10", "@{k}", "@30", "@all" nested under each subset.
    Subset keys "overall", "familiar", "unfamiliar" are the outer keys.

    Note: when k == 30 (the default), f"@{k}" and "@30" are the same key.
    Python dict construction silently deduplicates them; the value is the
    same for both, so the result is correct. The "@30" key is always present
    regardless of k to give consumers a stable look-up key.

    Raises
    ------
    ValueError
        If any seeded_window_id is missing from seeded_subset_labels.
    """
    if not seeded_window_ids:
        raise ValueError("seeded_window_ids is empty — cannot compute recall.")

    missing = [w for w in seeded_window_ids if w not in seeded_subset_labels]
    if missing:
        raise ValueError(
            f"seeded_subset_labels missing {len(missing)} keys: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    seeded_set = set(seeded_window_ids)
    familiar_set = {w for w, lbl in seeded_subset_labels.items() if lbl == "familiar"}
    unfamiliar_set = {w for w, lbl in seeded_subset_labels.items() if lbl == "unfamiliar"}

    accepted = [p for p in proposals if p.accepted]

    def _covered(top_proposals: list[ScoredProposal], target: set[WindowKey]) -> set[WindowKey]:
        covered: set[WindowKey] = set()
        for prop in top_proposals:
            for scene_id in prop.motivating_scene_ids:
                if scene_id in target:
                    covered.add(scene_id)
        return covered

    def _recall(covered: set[WindowKey], target: set[WindowKey]) -> float:
        if not target:
            return 0.0
        return len(covered & target) / len(target)

    def _at(cutoff: int | None) -> tuple[float, float, float]:
        top = accepted if cutoff is None else accepted[:cutoff]
        return (
            _recall(_covered(top, seeded_set),     seeded_set),
            _recall(_covered(top, familiar_set),   familiar_set),
            _recall(_covered(top, unfamiliar_set), unfamiliar_set),
        )

    # Compute recall at each K
    at_10 = _at(10)
    at_primary = _at(k)
    at_all = _at(None)
    at_30 = at_primary if k == 30 else _at(30)

    def _pack(triple: tuple[float, float, float]) -> dict[str, float]:
        return {"overall": triple[0], "familiar": triple[1], "unfamiliar": triple[2]}

    # Return structure: {subset: {k_str: recall}}
    # This matches EvaluationReport.seeded_recall nesting: arm -> subset -> k -> recall
    result: dict[str, dict[str, float]] = {
        "overall":    {"@10": at_10[0],    f"@{k}": at_primary[0], "@30": at_30[0],    "@all": at_all[0]},
        "familiar":   {"@10": at_10[1],    f"@{k}": at_primary[1], "@30": at_30[1],    "@all": at_all[1]},
        "unfamiliar": {"@10": at_10[2],    f"@{k}": at_primary[2], "@30": at_30[2],    "@all": at_all[2]},
    }

    return result


# ---------------------------------------------------------------------------
# Rating statistics
# ---------------------------------------------------------------------------

def compute_rating_stats(
    ratings: list[Rating],
) -> dict[str, dict[str, float | tuple[float, float] | None | int]]:
    """Compute per-arm mean coherence/usefulness and 95% bootstrap CI.

    Parameters
    ----------
    ratings
        All Rating objects from all raters. Rating.arm must match proposal arm.

    Returns
    -------
    dict keyed by arm, each value being:
        {
            "mean_coherence": float,
            "mean_usefulness": float,
            "coherence_ci_95": (lo, hi) | None,
            "usefulness_ci_95": (lo, hi) | None,
            "n": int,
        }
    Arms with zero ratings are omitted from the result.
    """
    from collections import defaultdict

    by_arm: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for r in ratings:
        by_arm[r.arm].append((r.coherence_score, r.usefulness_score))

    result: dict[str, dict] = {}
    rng = np.random.default_rng(_BOOTSTRAP_SEED)

    for arm, scores in by_arm.items():
        coh = np.array([s[0] for s in scores], dtype=float)
        use = np.array([s[1] for s in scores], dtype=float)
        n = len(scores)

        ci_coh: tuple[float, float] | None = None
        ci_use: tuple[float, float] | None = None

        if n >= MIN_N_FOR_CI:
            ci_coh = _bootstrap_ci(coh, rng)
            ci_use = _bootstrap_ci(use, rng)

        result[arm] = {
            "mean_coherence": float(np.mean(coh)),
            "mean_usefulness": float(np.mean(use)),
            "coherence_ci_95": ci_coh,
            "usefulness_ci_95": ci_use,
            "n": n,
        }

    return result


def _bootstrap_ci(
    data: np.ndarray,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for the mean."""
    boot_means = np.array([
        np.mean(rng.choice(data, size=len(data), replace=True))
        for _ in range(_BOOTSTRAP_SAMPLES)
    ])
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (lo, hi)


# ---------------------------------------------------------------------------
# Krippendorff's alpha (ordinal metric)
# ---------------------------------------------------------------------------

def krippendorff_alpha(
    data: np.ndarray,
) -> float | None:
    """Ordinal Krippendorff's alpha from an (items × raters) array.

    Uses d(c, k)^2 = (c - k)^2 (interval/squared-difference metric), which
    is standard for Likert-scale ordinal data.

    Parameters
    ----------
    data
        Shape (n_items, n_raters). Use np.nan for missing ratings.

    Returns
    -------
    float in [-1, 1], or None if fewer than 2 raters have overlapping ratings
    on any common item.

    Notes
    -----
    Returns None rather than raising when data is insufficient, so callers
    can handle the absent-metric case explicitly (float | None in the report).
    """
    data = np.array(data, dtype=float)
    n_items, n_raters = data.shape

    # Collect coincidences: for each item, generate all (v_a, v_b) pairs
    # where both values are non-missing.
    coincidences: list[tuple[float, float]] = []
    for i in range(n_items):
        row = data[i]
        valid = row[~np.isnan(row)]
        m = len(valid)
        if m < 2:
            continue
        for a_idx in range(m):
            for b_idx in range(m):
                if a_idx != b_idx:
                    coincidences.append((valid[a_idx], valid[b_idx]))

    if not coincidences:
        return None

    # Check that at least 2 distinct raters contributed
    rater_counts = (~np.isnan(data)).sum(axis=0)  # raters with ≥1 non-nan rating
    active_raters = int((rater_counts > 0).sum())
    if active_raters < 2:
        return None

    pairs = np.array(coincidences)
    c_vals = pairs[:, 0]
    k_vals = pairs[:, 1]

    # Observed disagreement: mean d(c, k) over all coincidences
    d_observed = float(np.mean((c_vals - k_vals) ** 2))

    # Expected disagreement: based on marginal distribution of all values
    all_values = data[~np.isnan(data)]
    n_total = len(all_values)
    if n_total < 2:
        return None

    # D_e = (1 / (n*(n-1))) * sum_{c,k} n_c * n_k * d(c,k)
    # where n_c, n_k are counts of each value in the full flattened array.
    unique_vals, counts = np.unique(all_values, return_counts=True)
    d_expected = 0.0
    for i, (vi, ni) in enumerate(zip(unique_vals, counts)):
        for j, (vj, nj) in enumerate(zip(unique_vals, counts)):
            if i != j:
                d_expected += ni * nj * (vi - vj) ** 2
    d_expected /= n_total * (n_total - 1)

    if d_expected == 0.0:
        # All raters gave identical values → perfect agreement
        return 1.0

    return float(1.0 - d_observed / d_expected)


# ---------------------------------------------------------------------------
# Differential examples
# ---------------------------------------------------------------------------

def compute_differential_examples(
    proposals_by_arm: dict[str, list[ScoredProposal]],
    ratings: list[Rating],
    top_n: int = 10,
) -> list:
    """Find top_n compositions where arms diverged most by rank.

    Returns a list of DifferentialExample objects, sorted by rank delta
    descending (largest divergence first). Requires at least 2 arms.
    Returns empty list for single-arm runs.
    """
    from pipeline.interfaces.report import DifferentialExample

    arms = list(proposals_by_arm.keys())
    if len(arms) < 2:
        return []

    # Build lookup: composition_id -> {arm: (rank, score)}
    by_id: dict[str, dict[str, tuple[int, float]]] = {}
    for arm, proposals in proposals_by_arm.items():
        accepted = [p for p in proposals if p.accepted]
        for rank_0, prop in enumerate(accepted):
            entry = by_id.setdefault(prop.composition_id, {})
            entry[arm] = (rank_0 + 1, prop.final_rank_score)

    # Only compositions present in ≥2 arms
    multi_arm = {cid: d for cid, d in by_id.items() if len(d) >= 2}
    if not multi_arm:
        return []

    # Per-arm per-proposal mean ratings
    from collections import defaultdict
    coherence_sums: dict[tuple[str, str], list[float]] = defaultdict(list)
    usefulness_sums: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in ratings:
        coherence_sums[(r.proposal_id, r.arm)].append(float(r.coherence_score))
        usefulness_sums[(r.proposal_id, r.arm)].append(float(r.usefulness_score))

    def _mean_ratings(cid: str, arm: str, store: dict) -> float | None:
        vals = store.get((cid, arm))
        return float(np.mean(vals)) if vals else None

    # Build DifferentialExample for each multi-arm composition
    examples = []
    for cid, arm_data in multi_arm.items():
        ranks = {a: arm_data[a][0] for a in arm_data}
        scores = {a: arm_data[a][1] for a in arm_data}
        rank_vals = list(ranks.values())
        delta = max(rank_vals) - min(rank_vals)

        # Retrieve constituents from any arm's proposal
        constituents: list[str] = []
        for arm, proposals in proposals_by_arm.items():
            for p in proposals:
                if p.composition_id == cid:
                    constituents = p.constituents
                    break
            if constituents:
                break

        coh_ratings = {
            a: _mean_ratings(cid, a, coherence_sums)  # type: ignore[arg-type]
            for a in arm_data
            if _mean_ratings(cid, a, coherence_sums) is not None
        }
        use_ratings = {
            a: _mean_ratings(cid, a, usefulness_sums)  # type: ignore[arg-type]
            for a in arm_data
            if _mean_ratings(cid, a, usefulness_sums) is not None
        }

        examples.append((delta, DifferentialExample(
            proposal_id=cid,
            constituents=constituents,
            arm_scores=scores,
            arm_ranks=ranks,
            coherence_ratings=coh_ratings,
            usefulness_ratings=use_ratings,
        )))

    examples.sort(key=lambda x: -x[0])
    return [ex for _, ex in examples[:top_n]]
