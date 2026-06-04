# Module: Extractor

The annotation stage. Produces **immutable evidence** (`RawDescriptor`) from a
driving clip via a **reason-first, open-vocabulary** pass — the only place the
vision-language model is invoked on the discovery path.

## How it works

Two steps, by design:

1. **Reason** — the VLM writes an *unconstrained* free-form analysis of the clip.
   No schema to fight, so the model never has to drop an observation that doesn't
   fit a fixed vocabulary.
2. **Structure** — a structuring pass extracts typed descriptors from that text.
   Each `RawDescriptor` carries:
   - a **typed axis** (`agents`, `interactions`, `conditions`, `road`,
     `ego_maneuver`, `weather`, `time`),
   - a **salience** in `[0,1]` — the model's learned judgment of operational
     criticality (not a hardcoded danger list, so it generalizes),
   - a **span pointer** to the sentence that justified the descriptor (every atom
     is auditable to its source),
   - a **text embedding** of the descriptor phrase (used by the curator to cluster).

Output is append-only and never edited downstream.

## Components

| File | Responsibility |
|---|---|
| `clients.py` | Injected protocols — `ReasonClient`, `StructureClient`, `Embedder` — each with an offline **stub** and a self-contained production client (Cosmos-Reason video, NIM text, hosted text-embeddings). |
| `extractor.py` | Orchestrates reason → structure → embed → `RawDescriptor`, with axis-synonym normalization and span verification. |
| `config.py` | Axes, prompt ids, limits. |
| `prompts/` | The reason + structuring prompt templates. |

## Design

- **Open vocabulary** — annotation is free text; structure emerges later (in the
  curator), not from a fixed schema here.
- **Salience at the source** — operational relevance is judged where the full
  scene context exists, per descriptor.
- **Auditability** — every structured atom points back to the reasoning that
  produced it, so a questionable label can be traced to a sentence and a clip.
- **Swappable model** — the reason/structure/embed clients are injected protocols;
  the reasoning model is one constructor argument away from replacement.

## Lego-block rule

Imports only `pipeline.interfaces`; production clients depend only on external SDKs
(no cross-module imports).

## Tests

```bash
python -m pytest pipeline/modules/extractor/tests/ -v
```
