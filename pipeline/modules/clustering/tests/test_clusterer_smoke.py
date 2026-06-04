"""Offline smoke test for the clustering module — no NIM/GPU.

Uses StubEmbedClient + a fake WindowStorageBase so the full
embed -> UMAP -> HDBSCAN -> ClusterReport path runs deterministically.
"""

from pipeline.interfaces.cluster import ClusterReport
from pipeline.interfaces.window import WindowKey
from pipeline.modules.clustering import Clusterer, ClustererConfig, StubEmbedClient


class _FakeStorage:
    """Minimal WindowStorageBase: returns a unique URL per (segment, window, camera)."""
    def get_window_video_url(self, segment_id, window_idx, camera="FRONT", ttl_seconds=3600):
        return f"https://fake/{segment_id}/{window_idx:04d}/{camera}.mp4"


def _windows(n: int) -> list[WindowKey]:
    return [WindowKey(segment_id=f"seg_{i:03d}", window_idx=0) for i in range(n)]


def test_embed_windows_produces_one_vector_per_window():
    c = Clusterer(StubEmbedClient(dim=64), ClustererConfig(cameras=("FRONT",)))
    embs = c.embed_windows(_windows(6), _FakeStorage())
    assert len(embs) == 6
    assert all(e.dim == 64 for e in embs)
    # deterministic: same window -> same vector
    again = c.embed_windows([WindowKey("seg_000", 0)], _FakeStorage())
    assert again[0].vector == embs[0].vector


def test_full_run_returns_valid_report():
    c = Clusterer(StubEmbedClient(dim=64), ClustererConfig(cameras=("FRONT",),
                                                           hdbscan_min_cluster_size=3))
    report = c.run(_windows(40), _FakeStorage())
    assert isinstance(report, ClusterReport)
    assert report.n_windows == 40
    assert len(report.assignments) == 40
    assert report.n_clusters + report.n_noise >= 0
    # every assignment has 3D viz coords and a valid window id
    for a in report.assignments:
        assert len(a.coords_3d) == 3
        assert a.window_id.segment_id.startswith("seg_")
    # round-trips
    assert ClusterReport.from_json(report.to_json()) == report


def test_too_few_windows_degrades_gracefully():
    c = Clusterer(StubEmbedClient(dim=32), ClustererConfig())
    report = c.run(_windows(2), _FakeStorage())   # below clustering minimum
    assert report.n_windows == 2
    assert report.n_clusters == 0
    assert all(a.is_noise for a in report.assignments)
