# drivers/ — composition-root scripts

These are the runnable entry points for the discovery pipeline. Each script is a
**composition root**: it imports several `pipeline/modules/` packages and wires
them together for one task (ingest, annotate, build the taxonomy, rank, evaluate).
They are intentionally *not* modules themselves — modules may not import one
another (the lego-block rule), but a driver may import many.

Run them from the repository root as modules:

```bash
python -m drivers.verity_taxonomy --stub      # offline wiring check
python -m drivers.verity_scale                # ingest -> annotate -> grow taxonomy
python -m drivers.verity_hardnovel            # difficulty-led selection -> Judge feed
python -m drivers.verity_retype               # interpretation-layer label correction
python -m drivers.verity_hardnovel_retyped    # selection over corrected atoms
```

| Driver | Role |
|---|---|
| `verity_run`, `verity_e2e_ingest`, `verity_e2e_extract` | ingest driving data → per-scene clips → evidence |
| `verity_taxonomy`, `verity_scale`, `verity_salience` | build / grow the emergent taxonomy from evidence |
| `verity_retype` | interpretation-layer label-typing correction (pure recompute) |
| `verity_hardnovel`, `verity_hardnovel_retyped` | difficulty-led selection → synthesized Judge feed |
| `verity_format_descriptions`, `salience_describe`, `salience_to_judge` | scenario formatting + feed helpers |
| `verity_eval_author`, `verity_eval_combine`, `verity_eval_recall` | build the blinded 3-arm evaluation + held-out recall |
| `verity_cluster`, `verity_rehypothesize`, `verity_judge`, `verity_agents_gt` | embed/cluster + compositional-baseline drivers |
