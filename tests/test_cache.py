from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from artifact_mgmt._artifact_model import ArtifactModel
from artifact_mgmt._cache import ModelCache
from artifact_mgmt._types import DepSnapshot, FrameworkInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dep_snapshot(framework_name: str = "pickle") -> DepSnapshot:
    return DepSnapshot(
        python_version="3.11.0",
        framework=FrameworkInfo(name=framework_name, version="1.0.0"),
        packages={},
        os="linux-x86_64",
        captured_at="2024-01-01T00:00:00Z",
    )


def _make_artifact(
    model_name: str = "fraud-detector",
    version: str = "1.0",
    framework_name: str = "pickle",
) -> ArtifactModel:
    from artifact_mgmt.serializers._pickle import PickleSerializer
    snap = _make_dep_snapshot(framework_name)
    # Use a picklable native model
    native = {"weights": [1.0, 2.0, 3.0]}
    serializer = PickleSerializer()
    return ArtifactModel(
        native,
        model_name=model_name,
        version=version,
        dep_snapshot=snap,
        serializer=serializer,
    )


# ---------------------------------------------------------------------------
# Story 6.3 — ModelCache
# ---------------------------------------------------------------------------


class TestModelCacheInit:
    def test_creates_cache_dir_if_not_exists(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "new" / "cache"
        assert not cache_dir.exists()
        ModelCache(cache_dir)
        assert cache_dir.exists()

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        ModelCache(str(tmp_path / "cache"))


class TestModelCacheGetMiss:
    def test_get_returns_none_when_no_entry(self, tmp_path: Path) -> None:
        cache = ModelCache(tmp_path)
        result = cache.get("fraud-detector", "1.0")
        assert result is None

    def test_get_returns_none_when_weights_missing(self, tmp_path: Path) -> None:
        cache = ModelCache(tmp_path)
        # Write only meta, no weights
        meta_path = tmp_path / "fraud-detector" / "1.0" / "meta.json"
        meta_path.parent.mkdir(parents=True)
        meta_path.write_text("{}")
        assert cache.get("fraud-detector", "1.0") is None

    def test_get_returns_none_when_meta_missing(self, tmp_path: Path) -> None:
        cache = ModelCache(tmp_path)
        weights_path = tmp_path / "fraud-detector" / "1.0" / "weights"
        weights_path.parent.mkdir(parents=True)
        weights_path.write_bytes(b"data")
        assert cache.get("fraud-detector", "1.0") is None


class TestModelCachePutAndGet:
    def test_put_then_get_returns_artifact(self, tmp_path: Path) -> None:
        cache = ModelCache(tmp_path)
        artifact = _make_artifact()
        cache.put("fraud-detector", "1.0", artifact)
        result = cache.get("fraud-detector", "1.0")
        assert result is not None
        assert isinstance(result, ArtifactModel)

    def test_cached_artifact_has_correct_model_name(self, tmp_path: Path) -> None:
        cache = ModelCache(tmp_path)
        artifact = _make_artifact(model_name="sentiment-model")
        cache.put("sentiment-model", "2.0", artifact)
        result = cache.get("sentiment-model", "2.0")
        assert result is not None
        assert result.model_name == "sentiment-model"

    def test_cached_artifact_has_correct_version(self, tmp_path: Path) -> None:
        cache = ModelCache(tmp_path)
        artifact = _make_artifact(version="3.1")
        cache.put("fraud-detector", "3.1", artifact)
        result = cache.get("fraud-detector", "3.1")
        assert result is not None
        assert result.version == "3.1"

    def test_cached_artifact_model_data_round_trips(self, tmp_path: Path) -> None:
        cache = ModelCache(tmp_path)
        artifact = _make_artifact()
        cache.put("fraud-detector", "1.0", artifact)
        result = cache.get("fraud-detector", "1.0")
        assert result is not None
        assert result.model == {"weights": [1.0, 2.0, 3.0]}

    def test_cached_dep_snapshot_preserved(self, tmp_path: Path) -> None:
        cache = ModelCache(tmp_path)
        artifact = _make_artifact()
        cache.put("fraud-detector", "1.0", artifact)
        result = cache.get("fraud-detector", "1.0")
        assert result is not None
        assert result.dep_snapshot.python_version == "3.11.0"
        assert result.dep_snapshot.framework.name == "pickle"

    def test_cuda_version_persisted_in_meta(self, tmp_path: Path) -> None:
        from artifact_mgmt.serializers._pickle import PickleSerializer
        snap = DepSnapshot(
            python_version="3.11.0",
            framework=FrameworkInfo(name="pickle", version="1.0.0"),
            packages={},
            os="linux-x86_64",
            captured_at="2024-01-01T00:00:00Z",
            cuda_version="11.8",
        )
        artifact = ArtifactModel(
            {"w": 1},
            model_name="gpu-model",
            version="1.0",
            dep_snapshot=snap,
            serializer=PickleSerializer(),
        )
        cache = ModelCache(tmp_path)
        cache.put("gpu-model", "1.0", artifact)
        result = cache.get("gpu-model", "1.0")
        assert result is not None
        assert result.dep_snapshot.cuda_version == "11.8"


class TestModelCachePersistence:
    def test_cached_bytes_survive_new_cache_instance(self, tmp_path: Path) -> None:
        artifact = _make_artifact()

        cache1 = ModelCache(tmp_path)
        cache1.put("fraud-detector", "1.0", artifact)

        # New instance pointing at same dir — simulates process restart
        cache2 = ModelCache(tmp_path)
        result = cache2.get("fraud-detector", "1.0")
        assert result is not None
        assert result.model == {"weights": [1.0, 2.0, 3.0]}


class TestModelCacheInvalidate:
    def test_invalidate_removes_entry(self, tmp_path: Path) -> None:
        cache = ModelCache(tmp_path)
        artifact = _make_artifact()
        cache.put("fraud-detector", "1.0", artifact)
        assert cache.get("fraud-detector", "1.0") is not None

        cache.invalidate("fraud-detector", "1.0")
        assert cache.get("fraud-detector", "1.0") is None

    def test_invalidate_does_not_affect_other_versions(self, tmp_path: Path) -> None:
        cache = ModelCache(tmp_path)
        a1 = _make_artifact(version="1.0")
        a2 = _make_artifact(version="2.0")
        cache.put("fraud-detector", "1.0", a1)
        cache.put("fraud-detector", "2.0", a2)

        cache.invalidate("fraud-detector", "1.0")

        assert cache.get("fraud-detector", "1.0") is None
        assert cache.get("fraud-detector", "2.0") is not None

    def test_invalidate_nonexistent_entry_does_not_raise(self, tmp_path: Path) -> None:
        cache = ModelCache(tmp_path)
        cache.invalidate("nonexistent", "9.9")  # should not raise


class TestModelCacheClear:
    def test_clear_removes_all_entries(self, tmp_path: Path) -> None:
        cache = ModelCache(tmp_path)
        cache.put("model-a", "1.0", _make_artifact(model_name="model-a"))
        cache.put("model-b", "1.0", _make_artifact(model_name="model-b"))

        cache.clear()

        assert cache.get("model-a", "1.0") is None
        assert cache.get("model-b", "1.0") is None

    def test_clear_on_empty_cache_does_not_raise(self, tmp_path: Path) -> None:
        cache = ModelCache(tmp_path)
        cache.clear()  # should not raise


class TestClientCacheDirWiring:
    def test_client_wires_cache_when_cache_dir_provided(self, tmp_path: Path) -> None:
        from artifact_mgmt.client import ArtifactMgmtClient

        mock_creds = MagicMock()
        mock_creds.access_key = "AKIATEST"
        mock_creds.secret_key = "testsecret"
        mock_creds.token = None
        mock_session = MagicMock()
        mock_session.get_credentials.return_value.get_frozen_credentials.return_value = mock_creds

        with patch("artifact_mgmt._http.boto3.Session", return_value=mock_session):
            client = ArtifactMgmtClient(stage="gamma", cache_dir=str(tmp_path))

        assert client._cache is not None
        assert isinstance(client._cache, ModelCache)

    def test_client_cache_is_none_when_no_cache_dir(self) -> None:
        from artifact_mgmt.client import ArtifactMgmtClient

        mock_creds = MagicMock()
        mock_creds.access_key = "AKIATEST"
        mock_creds.secret_key = "testsecret"
        mock_creds.token = None
        mock_session = MagicMock()
        mock_session.get_credentials.return_value.get_frozen_credentials.return_value = mock_creds

        with patch("artifact_mgmt._http.boto3.Session", return_value=mock_session):
            client = ArtifactMgmtClient(stage="gamma")

        assert client._cache is None
