"""Pluggable seams for the extractor: Reason, Structure, Embed.

Three protocols, each with a deterministic Stub (no network — used in tests) and a
self-contained production client (external deps only; no cross-module imports, per
the lego-block rule). The composition root injects whichever it wants.

  ReasonClient   video clip  -> free-form reasoning text
  StructureClient reasoning  -> JSON {descriptors:[{axis,text,span}]}
  Embedder       list[str]   -> list[vector]
"""

from __future__ import annotations

import base64
import hashlib
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class ReasonClient(Protocol):
    def describe(self, video_ref: str, prompt: str) -> str: ...


@runtime_checkable
class StructureClient(Protocol):
    def structure(self, reasoning: str, prompt: str) -> str: ...


@runtime_checkable
class Embedder(Protocol):
    dim: int
    def embed(self, texts: list[str]) -> list[list[float]]: ...


# ---------------------------------------------------------------------------
# Stubs — deterministic, offline. Same input -> same output.
# ---------------------------------------------------------------------------

class StubReasonClient:
    """Returns a fixed, plausible reasoning paragraph (no network)."""
    def describe(self, video_ref: str, prompt: str) -> str:
        return (
            "The ego vehicle is cruising on a multi-lane road in daylight under clear skies. "
            "A stopped bus occupies the right lane ahead. A pedestrian steps off the curb from "
            "behind the stopped bus, partially occluded. The ego maintains lane and slows."
        )


class StubStructureClient:
    """Extracts a fixed set of typed descriptors from any reasoning text."""
    def structure(self, reasoning: str, prompt: str) -> str:
        import json
        return json.dumps({"descriptors": [
            {"axis": "ego_maneuver", "text": "cruising then slowing",
             "span": "The ego maintains lane and slows.", "salience": 0.2},
            {"axis": "agents", "text": "stopped bus",
             "span": "A stopped bus occupies the right lane ahead.", "salience": 0.4},
            {"axis": "agents", "text": "pedestrian",
             "span": "A pedestrian steps off the curb from behind the stopped bus, partially occluded.", "salience": 0.6},
            {"axis": "interactions", "text": "occluded pedestrian emergence",
             "span": "A pedestrian steps off the curb from behind the stopped bus, partially occluded.", "salience": 0.9},
            {"axis": "road", "text": "multi-lane road",
             "span": "cruising on a multi-lane road", "salience": 0.1},
        ]})


class StubEmbedder:
    """Deterministic pseudo-embedding from text hash. Stable, offline."""
    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            h = hashlib.sha256(t.lower().encode()).digest()
            vec = [((h[i % len(h)] / 255.0) - 0.5) for i in range(self.dim)]
            n = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / n for x in vec])
        return out


# ---------------------------------------------------------------------------
# Production — self-contained (openai / requests / google only; no pipeline.modules)
# ---------------------------------------------------------------------------

def _gcs_to_data_uri(video_ref: str, max_seconds: float | None, width: int) -> str:
    """Download a gs:// clip via ADC, transcode small, return a base64 data URI."""
    from google.cloud import storage  # noqa: PLC0415
    no_scheme = video_ref[len("gs://"):]
    bucket, _, blob = no_scheme.partition("/")
    project = os.environ.get("GCS_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    with tempfile.TemporaryDirectory() as td:
        raw = os.path.join(td, "raw.mp4"); small = os.path.join(td, "s.mp4")
        storage.Client(project=project).bucket(bucket).blob(blob).download_to_filename(raw)
        cmd = ["ffmpeg", "-y", "-loglevel", "error"]
        if max_seconds:
            cmd += ["-t", f"{max_seconds:.2f}"]
        cmd += ["-i", raw, "-vf", f"scale={width}:-2", "-an", "-c:v", "libx264", "-crf", "30", small]
        subprocess.run(cmd, capture_output=True, timeout=180)
        return "data:video/mp4;base64," + base64.b64encode(Path(small).read_bytes()).decode()


class CosmosReasonClient:
    """Free-form scene reasoning via a (local) Cosmos-Reason NIM.

    video_ref may be a gs:// URI (downloaded+inlined via ADC), a local path, or any
    URL/data-URI the NIM accepts.
    """
    def __init__(self, model_id: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, max_seconds: float | None = 8.0,
                 width: int = 448, max_tokens: int = 1024) -> None:
        self.model_id = model_id or os.environ.get("COSMOS_REASON2_MODEL_ID", "nvidia/cosmos-reason1-7b")
        self._base_url = base_url or os.environ.get("NVIDIA_BASE_URL", "http://localhost:8081/v1")
        self._api_key = api_key or os.environ.get("NVIDIA_API_KEY", "local")
        self._max_seconds = max_seconds
        self._width = width
        self._max_tokens = max_tokens

    def describe(self, video_ref: str, prompt: str) -> str:
        from openai import OpenAI  # noqa: PLC0415
        ref = video_ref
        if video_ref.startswith("gs://"):
            ref = _gcs_to_data_uri(video_ref, self._max_seconds, self._width)
        client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        r = client.chat.completions.create(
            model=self.model_id, max_tokens=self._max_tokens, temperature=0.0,
            messages=[{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": ref}},
                {"type": "text", "text": prompt}]}])
        return r.choices[0].message.content or ""


class NIMStructureClient:
    """Structuring pass via a NIM text model (reads NVIDIA_BASE_URL / a text model)."""
    def __init__(self, model_id: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, max_tokens: int = 1024) -> None:
        self.model_id = model_id or os.environ.get(
            "STRUCTURE_NIM_MODEL_ID", os.environ.get("COSMOS_REASON2_MODEL_ID", "nvidia/cosmos-reason1-7b"))
        self._base_url = base_url or os.environ.get("NVIDIA_BASE_URL", "http://localhost:8081/v1")
        self._api_key = api_key or os.environ.get("NVIDIA_API_KEY", "local")
        self._max_tokens = max_tokens

    def structure(self, reasoning: str, prompt: str) -> str:
        from openai import OpenAI  # noqa: PLC0415
        client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        r = client.chat.completions.create(
            model=self.model_id, max_tokens=self._max_tokens, temperature=0.0,
            messages=[{"role": "user", "content": prompt.replace("{{REASONING}}", reasoning)}])
        return r.choices[0].message.content or ""


class NIMEmbedder:
    """Text embeddings via an OpenAI-compatible /embeddings endpoint.

    Points at EMBED_TEXT_BASE_URL + EMBED_TEXT_MODEL_ID (e.g. an NVIDIA text-embed
    NIM or the hosted catalog). Falls back to a deterministic hash embedding if the
    endpoint is unreachable, so the pipeline degrades instead of crashing.
    """
    def __init__(self, model_id: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, dim: int = 1024) -> None:
        self.model_id = model_id or os.environ.get("EMBED_TEXT_MODEL_ID", "nvidia/nv-embedqa-e5-v5")
        self._base_url = base_url or os.environ.get(
            "EMBED_TEXT_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self._api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self.dim = dim
        self._fallback = StubEmbedder(dim=dim)

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            import requests  # noqa: PLC0415
            r = requests.post(
                f"{self._base_url.rstrip('/')}/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"input": texts, "model": self.model_id,
                      "input_type": "query", "encoding_format": "float"},
                timeout=60)
            r.raise_for_status()
            data = r.json()["data"]
            vecs = [[float(x) for x in d["embedding"]] for d in data]
            self.dim = len(vecs[0]) if vecs else self.dim
            return vecs
        except Exception:  # noqa: BLE001 — degrade, never crash the run
            return self._fallback.embed(texts)
