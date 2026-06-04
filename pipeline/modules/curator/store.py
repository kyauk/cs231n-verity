"""Persistence: append-only evidence + numbered taxonomy versions.

Layout under `root`:
    evidence.jsonl                # append-only RawDescriptors (the immutable spine)
    taxonomy/v{NNNN}.json         # one file per version; never overwritten
    projections/v{NNNN}.json      # optional cache; always reconstructible

Invariants this store protects:
  * evidence.jsonl is append-only — descriptors are added, never edited. Re-adding
    identical evidence is idempotent (deterministic descriptor_id).
  * taxonomy versions are immutable once written.
Nothing here interprets the data; it only reads/writes the interface types.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from pipeline.interfaces.taxonomy import Projection, RawDescriptor, Taxonomy


class TaxonomyStore:
    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._evidence = self._root / "evidence.jsonl"
        self._tax_dir = self._root / "taxonomy"
        self._proj_dir = self._root / "projections"
        self._root.mkdir(parents=True, exist_ok=True)
        self._tax_dir.mkdir(parents=True, exist_ok=True)
        self._proj_dir.mkdir(parents=True, exist_ok=True)

    # -- evidence (append-only) --------------------------------------------
    def append_descriptors(self, descriptors: Iterable[RawDescriptor]) -> int:
        """Append descriptors, skipping ids already present (idempotent)."""
        existing = {d.descriptor_id for d in self.load_descriptors()}
        added = 0
        with self._evidence.open("a", encoding="utf-8") as f:
            for d in descriptors:
                if d.descriptor_id in existing:
                    continue
                f.write(json.dumps(d.to_json()) + "\n")
                existing.add(d.descriptor_id)
                added += 1
        return added

    def load_descriptors(self) -> list[RawDescriptor]:
        if not self._evidence.exists():
            return []
        out: list[RawDescriptor] = []
        for line in self._evidence.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(RawDescriptor.from_json(json.loads(line)))
        return out

    # -- taxonomy (versioned, immutable) -----------------------------------
    def save_taxonomy(self, taxonomy: Taxonomy) -> Path:
        path = self._tax_dir / f"v{taxonomy.version:04d}.json"
        if path.exists():
            raise FileExistsError(
                f"taxonomy version {taxonomy.version} already written at {path} — "
                f"versions are immutable; bump the version instead."
            )
        path.write_text(json.dumps(taxonomy.to_json(), indent=2), encoding="utf-8")
        return path

    def latest_version(self) -> int | None:
        versions = self._all_versions()
        return versions[-1] if versions else None

    def load_taxonomy(self, version: int | None = None) -> Taxonomy | None:
        if version is None:
            version = self.latest_version()
            if version is None:
                return None
        path = self._tax_dir / f"v{version:04d}.json"
        if not path.exists():
            return None
        return Taxonomy.from_json(json.loads(path.read_text(encoding="utf-8")))

    def _all_versions(self) -> list[int]:
        return sorted(
            int(p.stem[1:]) for p in self._tax_dir.glob("v*.json") if p.stem[1:].isdigit()
        )

    # -- projection (cache; reconstructible) -------------------------------
    def save_projection(self, projection: Projection) -> Path:
        path = self._proj_dir / f"v{projection.taxonomy_version:04d}.json"
        path.write_text(json.dumps(projection.to_json(), indent=2), encoding="utf-8")
        return path
