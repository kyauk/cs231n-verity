"""Pure-function sampler for one discrimination-test round.

Produces three pools of windows from the same dataset:

  - **Verity**:     top-`pool_size` accepted proposals by `final_rank_score`,
                    each represented by its first motivating scene.
  - **Random**:     uniform without replacement from every succeeded
                    `SchemaRecord` in the dataset.
  - **Naive-rare**: union sample from windows containing any of the
                    `top_k_rare_atoms` rarest atoms by marginal frequency.

All randomness keyed off the per-round `seed` so a round is exactly
reproducible. The Random and Naive-rare pools use distinct sub-seeds so the
operator can change one pool's sampling without disturbing the other.

This module reuses `pipeline.modules.hypothesizer.frequency.compute_frequencies`
for the naive-rare baseline — same code path used in production discovery,
so the "frequency" measure is identical between Hypothesizer and baseline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from pipeline.interfaces.proposal import ScoredProposal
from pipeline.interfaces.schema_record import SchemaRecord
from pipeline.interfaces.window import WindowKey
from pipeline.modules.hypothesizer.frequency import (
    compute_frequencies,
    extract_atoms,
)


class SamplingError(Exception):
    """Raised when the inputs cannot produce three valid pools.

    Always actionable: tells the operator which pool fell short and how to
    fix it (more accepted proposals, more succeeded records, etc.).
    """


@dataclass
class SampleResult:
    verity: list[WindowKey]
    random: list[WindowKey]
    naive_rare: list[WindowKey]
    naive_rare_atoms: list[str]   # the top-K rarest atoms actually used


def sample_three_pools(
    scored: list[ScoredProposal],
    schema_records: list[SchemaRecord],
    pool_size: int = 30,
    seed: int = 0,
    top_k_rare_atoms: int = 5,
) -> SampleResult:
    """Sample three pools for one discrimination-test round.

    Parameters
    ----------
    scored
        ScoredProposals from a `pipeline.run analyze` run. Used only for
        the Verity pool. Rejected proposals and proposals with zero
        motivating scenes are skipped.
    schema_records
        SchemaRecords from the same analyze run. Used for the Random pool
        (uniform over succeeded records) and for the Naive-rare pool's
        atom-frequency computation. Failed records are skipped.
    pool_size
        Number of windows per pool. Default 30. Same N for all three pools
        so downstream non-parametric stats compare like with like.
    seed
        Master RNG seed. The Verity pool is deterministic in `scored`
        (sorted by score, no randomness needed). The Random and Naive-rare
        pools use `seed` and `seed+1` respectively.
    top_k_rare_atoms
        How many of the rarest atoms (by marginal frequency) define the
        Naive-rare pool. Default 5 — broader than single-rarest, but still
        defensibly a "frequency-only" baseline.

    Raises
    ------
    SamplingError
        If any pool cannot reach `pool_size`.
    """
    if pool_size <= 0:
        raise SamplingError(f"pool_size must be positive, got {pool_size}")

    # --- Verity pool: deterministic top-K by final_rank_score -----------
    accepted = [
        s for s in scored
        if s.accepted and s.motivating_scene_ids
    ]
    if len(accepted) < pool_size:
        raise SamplingError(
            f"Verity pool needs {pool_size} accepted proposals with at "
            f"least one motivating scene; only {len(accepted)} qualify. "
            f"Run `analyze` on a larger dataset, or lower pool_size."
        )
    # Sort by score DESC; break ties deterministically on composition_id so
    # the Verity pool is bit-exactly reproducible across re-runs regardless
    # of the order the operator's scored.json was constructed in.
    accepted.sort(key=lambda s: (-s.final_rank_score, s.composition_id))
    verity_windows = [s.motivating_scene_ids[0] for s in accepted[:pool_size]]

    # --- Random pool: uniform over succeeded records --------------------
    eligible = [r for r in schema_records if r.succeeded]
    if len(eligible) < pool_size:
        raise SamplingError(
            f"Random pool needs {pool_size} succeeded schema_records; "
            f"only {len(eligible)} available. Re-run `analyze` on more windows."
        )
    rng_random = random.Random(seed)
    random_windows = [
        r.window_id for r in rng_random.sample(eligible, pool_size)
    ]

    # --- Naive-rare pool: top-K rarest atoms, union sample --------------
    atoms_per_window = [
        extract_atoms(
            fields=r.fields,
            compose_over=None,
            valid_atoms=None,
            window_id=str(r.window_id),
        )
        for r in eligible
    ]
    marginals, _ = compute_frequencies(atoms_per_window)
    if not marginals:
        raise SamplingError(
            "Naive-rare pool: no atoms extracted from schema_records. "
            "The encoder may have returned empty `fields` dicts."
        )
    sorted_atoms = sorted(marginals.items(), key=lambda kv: kv[1])
    rare_atoms_list = [a for a, _ in sorted_atoms[:top_k_rare_atoms]]
    rare_set = set(rare_atoms_list)

    candidates = [
        r.window_id
        for r, atoms in zip(eligible, atoms_per_window)
        if atoms & rare_set
    ]
    if len(candidates) < pool_size:
        raise SamplingError(
            f"Naive-rare pool needs {pool_size} windows containing any of "
            f"the {top_k_rare_atoms} rarest atoms ({rare_atoms_list}); only "
            f"{len(candidates)} qualify. Increase top_k_rare_atoms, lower "
            f"pool_size, or run `analyze` on more windows."
        )
    rng_rare = random.Random(seed + 1)
    naive_rare_windows = rng_rare.sample(candidates, pool_size)

    return SampleResult(
        verity=verity_windows,
        random=random_windows,
        naive_rare=naive_rare_windows,
        naive_rare_atoms=rare_atoms_list,
    )
