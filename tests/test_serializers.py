from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from artifact_mgmt._exceptions import UnknownSerializerError, FrameworkNotInstalledError
from artifact_mgmt._types import DepSnapshot, FrameworkInfo
from artifact_mgmt.serializers import SerializerRegistry
from artifact_mgmt.serializers._base import Serializer
from artifact_mgmt.serializers._pickle import PickleSerializer


# ---------------------------------------------------------------------------
# Story 3.1 — SerializerRegistry + Serializer ABC
# ---------------------------------------------------------------------------


def _make_dep_snapshot(framework_name: str) -> DepSnapshot:
    return DepSnapshot(
        python_version="3.11.0",
        framework=FrameworkInfo(name=framework_name, version="1.0.0"),
        packages={},
        os="linux-x86_64",
        captured_at="2024-01-01T00:00:00Z",
    )


class TestSerializerABC:
    def test_serializer_cannot_be_instantiated_directly(self) -> None:
        with pytest.raises(TypeError):
            Serializer()  # type: ignore[abstract]

    def test_concrete_serializer_must_implement_all_abstract_methods(self) -> None:
        # PickleSerializer is a concrete implementation — verify it satisfies ABC
        s = PickleSerializer()
        assert isinstance(s, Serializer)


class TestSerializerRegistryDetect:
    def test_detect_returns_pytorch_serializer_for_nn_module(self) -> None:
        mock_model = MagicMock()
        mock_module_class = MagicMock()
        mock_model.__class__ = mock_module_class

        with patch("artifact_mgmt.serializers._torch.TorchSerializer.can_handle", return_value=True):
            result = SerializerRegistry.detect(mock_model)
            assert result.framework_name == "pytorch"

    def test_detect_returns_sklearn_serializer_for_base_estimator(self) -> None:
        mock_model = MagicMock()
        with (
            patch("artifact_mgmt.serializers._torch.TorchSerializer.can_handle", return_value=False),
            patch("artifact_mgmt.serializers._huggingface.HuggingFaceSerializer.can_handle", return_value=False),
            patch("artifact_mgmt.serializers._sklearn.SklearnSerializer.can_handle", return_value=True),
        ):
            result = SerializerRegistry.detect(mock_model)
            assert result.framework_name == "sklearn"

    def test_detect_returns_huggingface_serializer_for_pretrained_model(self) -> None:
        mock_model = MagicMock()
        with (
            patch("artifact_mgmt.serializers._torch.TorchSerializer.can_handle", return_value=False),
            patch("artifact_mgmt.serializers._huggingface.HuggingFaceSerializer.can_handle", return_value=True),
        ):
            result = SerializerRegistry.detect(mock_model)
            assert result.framework_name == "huggingface"

    def test_detect_returns_tensorflow_serializer_for_keras_model(self) -> None:
        mock_model = MagicMock()
        with (
            patch("artifact_mgmt.serializers._torch.TorchSerializer.can_handle", return_value=False),
            patch("artifact_mgmt.serializers._huggingface.HuggingFaceSerializer.can_handle", return_value=False),
            patch("artifact_mgmt.serializers._sklearn.SklearnSerializer.can_handle", return_value=False),
            patch("artifact_mgmt.serializers._tensorflow.TensorFlowSerializer.can_handle", return_value=True),
        ):
            result = SerializerRegistry.detect(mock_model)
            assert result.framework_name == "tensorflow"

    def test_detect_never_returns_pickle_serializer(self) -> None:
        mock_model = MagicMock()
        with (
            patch("artifact_mgmt.serializers._torch.TorchSerializer.can_handle", return_value=False),
            patch("artifact_mgmt.serializers._huggingface.HuggingFaceSerializer.can_handle", return_value=False),
            patch("artifact_mgmt.serializers._sklearn.SklearnSerializer.can_handle", return_value=False),
            patch("artifact_mgmt.serializers._tensorflow.TensorFlowSerializer.can_handle", return_value=False),
        ):
            with pytest.raises(UnknownSerializerError):
                SerializerRegistry.detect(mock_model)

    def test_detect_raises_unknown_serializer_error_for_unrecognised_model_type(self) -> None:
        class WeirdModel:
            pass

        with (
            patch("artifact_mgmt.serializers._torch.TorchSerializer.can_handle", return_value=False),
            patch("artifact_mgmt.serializers._huggingface.HuggingFaceSerializer.can_handle", return_value=False),
            patch("artifact_mgmt.serializers._sklearn.SklearnSerializer.can_handle", return_value=False),
            patch("artifact_mgmt.serializers._tensorflow.TensorFlowSerializer.can_handle", return_value=False),
        ):
            with pytest.raises(UnknownSerializerError):
                SerializerRegistry.detect(WeirdModel())

    def test_detect_skips_serializer_when_framework_not_installed(self) -> None:
        mock_model = MagicMock()
        with (
            patch(
                "artifact_mgmt.serializers._torch.TorchSerializer.can_handle",
                side_effect=FrameworkNotInstalledError("torch not installed"),
            ),
            patch("artifact_mgmt.serializers._huggingface.HuggingFaceSerializer.can_handle", return_value=True),
        ):
            result = SerializerRegistry.detect(mock_model)
            assert result.framework_name == "huggingface"


class TestSerializerRegistryDetectFromSnapshot:
    def test_detect_from_snapshot_returns_pytorch_serializer(self) -> None:
        snap = _make_dep_snapshot("pytorch")
        result = SerializerRegistry.detect_from_snapshot(snap)
        assert result.framework_name == "pytorch"

    def test_detect_from_snapshot_returns_huggingface_serializer(self) -> None:
        snap = _make_dep_snapshot("huggingface")
        result = SerializerRegistry.detect_from_snapshot(snap)
        assert result.framework_name == "huggingface"

    def test_detect_from_snapshot_returns_sklearn_serializer(self) -> None:
        snap = _make_dep_snapshot("sklearn")
        result = SerializerRegistry.detect_from_snapshot(snap)
        assert result.framework_name == "sklearn"

    def test_detect_from_snapshot_returns_tensorflow_serializer(self) -> None:
        snap = _make_dep_snapshot("tensorflow")
        result = SerializerRegistry.detect_from_snapshot(snap)
        assert result.framework_name == "tensorflow"

    def test_detect_from_snapshot_returns_pickle_serializer(self) -> None:
        snap = _make_dep_snapshot("pickle")
        result = SerializerRegistry.detect_from_snapshot(snap)
        assert result.framework_name == "pickle"

    def test_detect_from_snapshot_raises_unknown_serializer_error_for_unknown_framework(
        self,
    ) -> None:
        snap = _make_dep_snapshot("xgboost-custom")
        with pytest.raises(UnknownSerializerError):
            SerializerRegistry.detect_from_snapshot(snap)


class TestSerializerRegistryGetByName:
    def test_get_by_name_returns_pickle_serializer(self) -> None:
        result = SerializerRegistry.get_by_name("pickle")
        assert result.framework_name == "pickle"

    def test_get_by_name_raises_unknown_serializer_error_for_unknown_name(self) -> None:
        with pytest.raises(UnknownSerializerError):
            SerializerRegistry.get_by_name("nonexistent")


class TestLazyImport:
    def test_importing_artifact_mgmt_does_not_raise_without_framework_packages(
        self,
    ) -> None:
        # Simulate torch/transformers/sklearn/tensorflow not being importable.
        import sys
        blocked = ["torch", "transformers", "sklearn", "tensorflow", "joblib"]
        original = {k: sys.modules.pop(k) for k in blocked if k in sys.modules}
        try:
            import importlib
            import artifact_mgmt  # noqa: F401
            importlib.reload(artifact_mgmt)
        finally:
            sys.modules.update(original)
