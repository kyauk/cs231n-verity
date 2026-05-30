"""Protocol-satisfaction tests for WindowStorageBase.

The Protocol formalizes the duck-typed contract the Encoder + Judge UI rely
on. Every concrete implementation in pipeline.modules.storage must satisfy it
at runtime. New implementations are caught here, not at first integration.
"""

from __future__ import annotations

from pipeline.interfaces.window import WindowStorageBase


def test_windowstorage_satisfies_protocol() -> None:
    """The canonical WindowStorage must satisfy the Protocol at runtime."""
    from pipeline.modules.storage import WindowStorage
    storage = WindowStorage(bucket_uri="gs://test/prefix")
    assert isinstance(storage, WindowStorageBase)


def test_protocol_has_all_three_methods() -> None:
    """Sanity check: the Protocol declares exactly the three methods the
    Encoder calls. If this changes, every implementation needs to update."""
    methods = {
        name for name in dir(WindowStorageBase)
        if not name.startswith("_")
    }
    assert methods == {
        "list_windows", "get_window_video_url", "get_window_manifest",
    }


def test_object_missing_method_does_not_satisfy_protocol() -> None:
    """An object missing any method must NOT satisfy the Protocol."""
    class Incomplete:
        def list_windows(self, segment_id=None):
            return []
        # missing get_window_video_url and get_window_manifest

    assert not isinstance(Incomplete(), WindowStorageBase)
