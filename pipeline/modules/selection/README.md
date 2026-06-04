# Module: Selection

Surfaces edge-case scenes from the curated evidence and turns each into a **novel,
generation-ready scenario**. Scenes are ranked by **model-judged difficulty
(primary)** and **behavior-novelty (refining)**; the surfaced scene is described
by **synthesizing** a new scenario from its composition, not by re-captioning a
clip.

## Components

| File | Responsibility | Form |
|---|---|---|
| `ranking.py` | `behavior_novelty()` + `combined_score()` = `difficulty·w_d + novelty·w_n` | pure functions |
| `difficulty.py` | `score_difficulty(client, video)` — independent difficulty judgment | injected VLM client |
| `synthesis.py` | `synthesize_scenario(atoms, grounding)` — novel scene from a composition | text-model call |
| `config.py` | weights, `novelty_axes`, prompts | configuration |

## Design

- **Difficulty leads.** A model-judged operational-hardness score drives the
  ranking. It is granularity-agnostic — it evaluates the whole scene, so it
  captures hard configurations without pre-naming which attributes matter. The
  difficulty client is an injected seam and can be replaced with a stack-native
  reasoning signal.
- **Behavior-novelty refines.** Computed over `{interactions, conditions,
  ego_maneuver}` only — never agent attributes — so incidental attributes (e.g.
  vehicle colour) cannot drive the ranking.
- **Synthesis, not reconstruction.** A composition may have no single source clip;
  the scenario is generated to *embody* the atoms (grounded by, not copied from,
  the evidence), producing a generation-ready spec.
- **Independent difficulty as a cross-check.** Difficulty is judged from a separate
  viewing than salience; disagreement flags possible over-reports.

## Lego-block rule

Imports only `pipeline.interfaces` and external SDKs. Drivers (e.g.
`verity_hardnovel.py`) project the curator's atoms and compose these components;
the module never reaches into another module (enforced by `tests/test_selection.py`).

## Tests

```bash
python -m pytest pipeline/modules/selection/tests/ -v
```
