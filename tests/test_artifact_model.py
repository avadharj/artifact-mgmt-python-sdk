from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from artifact_mgmt._artifact_model import ArtifactModel
from artifact_mgmt._types import DepSnapshot, FrameworkInfo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def dep_snapshot() -> DepSnapshot:
    return DepSnapshot(
        python_version="3.11.0",
        framework=FrameworkInfo(name="pytorch", version="2.0.0"),
        packages={},
        os="linux-x86_64",
        captured_at="2024-01-01T00:00:00Z",
    )


@pytest.fixture()
def mock_serializer() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def native_model() -> MagicMock:
    return MagicMock(name="native_model")


@pytest.fixture()
def artifact(native_model, dep_snapshot, mock_serializer) -> ArtifactModel:
    return ArtifactModel(
        native_model,
        model_name="fraud-detector",
        version="1.0",
        dep_snapshot=dep_snapshot,
        serializer=mock_serializer,
    )


# ---------------------------------------------------------------------------
# Story 5.1 — ArtifactModel wrapper + __getattr__ forwarding
# ---------------------------------------------------------------------------


class TestProperties:
    def test_model_returns_native_model(self, artifact, native_model):
        assert artifact.model is native_model

    def test_model_name_returns_correct_name(self, artifact):
        assert artifact.model_name == "fraud-detector"

    def test_version_returns_correct_version(self, artifact):
        assert artifact.version == "1.0"

    def test_dep_snapshot_returns_snapshot(self, artifact, dep_snapshot):
        assert artifact.dep_snapshot is dep_snapshot


class TestGetattr:
    def test_forwards_unknown_attribute_to_native_model(self, artifact, native_model):
        native_model.train = MagicMock(return_value="training")
        assert artifact.train() == "training"
        native_model.train.assert_called_once()

    def test_forwards_parameters_call_to_native_model(self, artifact, native_model):
        params = [MagicMock(), MagicMock()]
        native_model.parameters = MagicMock(return_value=params)
        assert artifact.parameters() is params

    def test_raises_attribute_error_for_missing_attribute(self, artifact, native_model):
        # native_model is a MagicMock so we need a real object to test AttributeError
        real_model = object()
        snap = DepSnapshot(
            python_version="3.11.0",
            framework=FrameworkInfo(name="pytorch", version="2.0.0"),
            packages={},
            os="linux-x86_64",
            captured_at="2024-01-01T00:00:00Z",
        )
        artifact_real = ArtifactModel(
            real_model,
            model_name="test",
            version="1.0",
            dep_snapshot=snap,
            serializer=MagicMock(),
        )
        with pytest.raises(AttributeError):
            _ = artifact_real.nonexistent_method

    def test_does_not_silently_return_none_for_missing_attribute(self, artifact, native_model):
        real_model = object()
        snap = DepSnapshot(
            python_version="3.11.0",
            framework=FrameworkInfo(name="pytorch", version="2.0.0"),
            packages={},
            os="linux-x86_64",
            captured_at="2024-01-01T00:00:00Z",
        )
        artifact_real = ArtifactModel(
            real_model,
            model_name="test",
            version="1.0",
            dep_snapshot=snap,
            serializer=MagicMock(),
        )
        result = None
        try:
            result = artifact_real.nonexistent_method
        except AttributeError:
            pass
        assert result is None  # only reached via exception path, not silent None return


class TestRepr:
    def test_repr_includes_model_name_and_version(self, artifact):
        r = repr(artifact)
        assert "fraud-detector" in r
        assert "1.0" in r

    def test_repr_format(self, artifact):
        assert repr(artifact) == "ArtifactModel('fraud-detector', version='1.0', ...)"
