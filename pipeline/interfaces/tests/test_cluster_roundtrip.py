"""Round-trip contract tests for the clustering interface types."""

from pipeline.interfaces.cluster import (
    ClusterAssignment,
    ClusterReport,
    WindowEmbedding,
)
from pipeline.interfaces.window import WindowKey


def test_window_embedding_roundtrip():
    e = WindowEmbedding(window_id=WindowKey("seg_a", 0), vector=[0.1, -0.2, 0.3], dim=3)
    e2 = WindowEmbedding.from_json(e.to_json())
    assert e2 == e
    assert e2.window_id == WindowKey("seg_a", 0)


def test_cluster_assignment_roundtrip_and_noise():
    a = ClusterAssignment(
        window_id=WindowKey("seg_b", 2), cluster_id=-1,
        glosh_score=0.91, probability=0.0, coords_3d=[1.0, 2.0, 3.0],
    )
    a2 = ClusterAssignment.from_json(a.to_json())
    assert a2 == a
    assert a2.is_noise is True
    assert ClusterAssignment.from_json(
        ClusterAssignment(WindowKey("s", 0), 4, 0.1, 0.8, [0, 0, 0]).to_json()
    ).is_noise is False


def test_cluster_report_roundtrip():
    r = ClusterReport(
        assignments=[
            ClusterAssignment(WindowKey("s", 0), 0, 0.1, 0.9, [0.0, 0.0, 0.0]),
            ClusterAssignment(WindowKey("s", 1), -1, 0.8, 0.0, [1.0, 1.0, 1.0]),
        ],
        n_clusters=1, n_noise=1, embedding_dim=256,
        config={"umap_seed": 42}, created_at="2026-06-04T00:00:00+00:00",
    )
    r2 = ClusterReport.from_json(r.to_json())
    assert r2 == r
    assert r2.n_windows == 2
