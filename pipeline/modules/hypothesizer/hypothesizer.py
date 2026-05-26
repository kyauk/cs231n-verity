"""Module 3: Hypothesizer — public interface.

Accepts a list of SchemaRecord objects, extracts qualified atoms from each
succeeded record, computes marginal and pairwise frequencies, then enumerates
and ranks compositionally novel scenario proposals.

Two counts are reported to stderr:
  - Records skipped because failure_mode is set (encoder-side failure)
  - Records skipped because required fields are null (succeeded but incomplete)

These are separate because they have different root causes and different fixes.
"""

from __future__ import annotations

import sys
from typing import Any

from pipeline.interfaces.proposal import CompositionProposal
from pipeline.interfaces.schema_record import SchemaRecord
from pipeline.modules.hypothesizer.composition import build_proposals
from pipeline.modules.hypothesizer.config import (
    HypothesizerConfig,
    HypothesizerEmptyInputError,
)
from pipeline.modules.hypothesizer.frequency import compute_frequencies, extract_atoms


class Hypothesizer:
    """Discovers compositionally novel scenario hypotheses from SchemaRecords.

    Usage
    -----
        hyp = Hypothesizer()
        proposals = hyp.propose(records, arm="reasoning")

    The returned list is ranked by novelty_score DESC (composition_id ASC as
    tie-breaker) and has at most config.top_k entries.

    Thread safety
    -------------
    propose() is stateless and re-entrant. Multiple threads may call it on the
    same Hypothesizer instance concurrently without issue.
    """

    def __init__(self, config: HypothesizerConfig = HypothesizerConfig()) -> None:
        self._config = config

    def propose(
        self,
        records: list[SchemaRecord],
        arm: str = "reasoning",
    ) -> list[CompositionProposal]:
        """Propose compositionally novel scenarios from a set of SchemaRecords.

        Parameters
        ----------
        records
            All SchemaRecord outputs from Module 2 (Encoder) for one arm.
            Records with failure_mode set are counted and skipped.
            Records with succeeded=True but null fields are counted and skipped.
        arm
            The encoder arm that produced these records. Passed through to
            CompositionProposal.arm.

        Returns
        -------
        list[CompositionProposal]
            Ranked novelty proposals, length ≤ config.top_k.

        Raises
        ------
        HypothesizerEmptyInputError
            If records is empty or all records were skipped.
        """
        n_failure_skipped = 0
        n_null_skipped = 0

        atom_sets = []
        keys = []

        for record in records:
            if not record.succeeded:
                n_failure_skipped += 1
                continue

            atom_set = extract_atoms(
                fields=record.fields,
                compose_over=self._config.compose_over,
                valid_atoms=self._config.valid_atoms,
                window_id=str(record.window_id),
            )

            if not atom_set:
                n_null_skipped += 1
                continue

            atom_sets.append(atom_set)
            keys.append(record.window_id)

        if n_failure_skipped:
            print(
                f"[Hypothesizer] Skipped {n_failure_skipped} records "
                f"(failure_mode set — encoder-side failure)",
                file=sys.stderr,
            )
        if n_null_skipped:
            print(
                f"[Hypothesizer] Skipped {n_null_skipped} records "
                f"(null fields — succeeded but no atoms extracted)",
                file=sys.stderr,
            )

        if not atom_sets:
            raise HypothesizerEmptyInputError(
                f"No usable records after filtering "
                f"({n_failure_skipped} failure_mode, {n_null_skipped} null fields, "
                f"{len(records)} total)."
            )

        marginal, pairwise = compute_frequencies(atom_sets)

        return build_proposals(
            atom_sets=atom_sets,
            keys=keys,
            marginal=marginal,
            pairwise=pairwise,
            config=self._config,
            arm=arm,
        )
