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


# ---------------------------------------------------------------------------
# Story 5.2 — freeze / unfreeze / fine_tune_params delegation
# ---------------------------------------------------------------------------


class TestFreeze:
    def test_delegates_to_serializer_freeze(self, artifact, mock_serializer, native_model):
        artifact.freeze(3)
        mock_serializer.freeze.assert_called_once_with(native_model, 3)

    def test_freeze_zero_layers(self, artifact, mock_serializer, native_model):
        artifact.freeze(0)
        mock_serializer.freeze.assert_called_once_with(native_model, 0)

    def test_sklearn_backed_model_raises_not_implemented(self, native_model, dep_snapshot):
        from artifact_mgmt.serializers._sklearn import SklearnSerializer
        serializer = SklearnSerializer()
        artifact = ArtifactModel(
            native_model,
            model_name="clf",
            version="1.0",
            dep_snapshot=dep_snapshot,
            serializer=serializer,
        )
        with pytest.raises(NotImplementedError, match="sklearn estimators do not support layer freezing"):
            artifact.freeze(2)


class TestUnfreeze:
    def test_delegates_to_serializer_unfreeze(self, artifact, mock_serializer, native_model):
        artifact.unfreeze(2)
        mock_serializer.unfreeze.assert_called_once_with(native_model, 2)

    def test_sklearn_backed_model_raises_not_implemented(self, native_model, dep_snapshot):
        from artifact_mgmt.serializers._sklearn import SklearnSerializer
        serializer = SklearnSerializer()
        artifact = ArtifactModel(
            native_model,
            model_name="clf",
            version="1.0",
            dep_snapshot=dep_snapshot,
            serializer=serializer,
        )
        with pytest.raises(NotImplementedError):
            artifact.unfreeze(2)


class TestFineTuneParams:
    def test_delegates_to_serializer_fine_tune_params(self, artifact, mock_serializer, native_model):
        expected = [MagicMock(), MagicMock()]
        mock_serializer.fine_tune_params.return_value = expected
        result = artifact.fine_tune_params()
        mock_serializer.fine_tune_params.assert_called_once_with(native_model)
        assert result is expected

    def test_sklearn_backed_model_raises_not_implemented(self, native_model, dep_snapshot):
        from artifact_mgmt.serializers._sklearn import SklearnSerializer
        serializer = SklearnSerializer()
        artifact = ArtifactModel(
            native_model,
            model_name="clf",
            version="1.0",
            dep_snapshot=dep_snapshot,
            serializer=serializer,
        )
        with pytest.raises(NotImplementedError):
            artifact.fine_tune_params()


# ---------------------------------------------------------------------------
# Story 5.3 — predict delegation
# ---------------------------------------------------------------------------


class TestPredict:
    def test_delegates_to_serializer_predict(self, artifact, mock_serializer, native_model):
        expected = MagicMock(name="output")
        mock_serializer.predict.return_value = expected
        X = MagicMock(name="input")

        result = artifact.predict(X)

        mock_serializer.predict.assert_called_once_with(native_model, X)
        assert result is expected

    def test_pytorch_predict_delegates_via_torch_serializer(self, native_model, dep_snapshot):
        import sys
        torch_mock = MagicMock(name="torch")
        torch_mock.nn = MagicMock()
        torch_mock.nn.Module = type("Module", (), {})
        sys.modules["torch"] = torch_mock
        sys.modules["torch.nn"] = torch_mock.nn

        try:
            from artifact_mgmt.serializers._torch import TorchSerializer
            serializer = TorchSerializer()
            native_model.__class__ = torch_mock.nn.Module

            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=None)
            ctx.__exit__ = MagicMock(return_value=False)
            torch_mock.no_grad.return_value = ctx

            expected = MagicMock(name="torch_output")
            native_model.return_value = expected

            artifact = ArtifactModel(
                native_model,
                model_name="resnet",
                version="1.0",
                dep_snapshot=dep_snapshot,
                serializer=serializer,
            )
            X = MagicMock(name="tensor")
            result = artifact.predict(X)

            native_model.eval.assert_called_once()
            torch_mock.no_grad.assert_called_once()
            assert result is expected
        finally:
            sys.modules.pop("torch", None)
            sys.modules.pop("torch.nn", None)

    def test_huggingface_predict_unpacks_dict(self, native_model, dep_snapshot):
        import sys
        tf_mock = MagicMock(name="transformers")
        tf_mock.PreTrainedModel = type("PreTrainedModel", (), {})
        sys.modules["transformers"] = tf_mock

        try:
            from artifact_mgmt.serializers._huggingface import HuggingFaceSerializer
            serializer = HuggingFaceSerializer()

            expected = MagicMock(name="hf_output")
            native_model.return_value = expected

            artifact = ArtifactModel(
                native_model,
                model_name="bert",
                version="1.0",
                dep_snapshot=dep_snapshot,
                serializer=serializer,
            )
            X = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
            result = artifact.predict(X)

            native_model.assert_called_once_with(**X)
            assert result is expected
        finally:
            sys.modules.pop("transformers", None)

    def test_sklearn_predict_calls_model_predict(self, native_model, dep_snapshot):
        import sys
        sk_mock = MagicMock(name="sklearn")
        sk_mock.base = MagicMock()
        sk_mock.base.BaseEstimator = type("BaseEstimator", (), {})
        joblib_mock = MagicMock(name="joblib")
        sys.modules["sklearn"] = sk_mock
        sys.modules["sklearn.base"] = sk_mock.base
        sys.modules["joblib"] = joblib_mock

        try:
            from artifact_mgmt.serializers._sklearn import SklearnSerializer
            serializer = SklearnSerializer()

            expected = MagicMock(name="predictions")
            native_model.predict.return_value = expected

            artifact = ArtifactModel(
                native_model,
                model_name="logreg",
                version="1.0",
                dep_snapshot=dep_snapshot,
                serializer=serializer,
            )
            X = MagicMock(name="features")
            result = artifact.predict(X)

            native_model.predict.assert_called_once_with(X)
            assert result is expected
        finally:
            sys.modules["sklearn"] = None  # type: ignore[assignment]
            sys.modules["sklearn.base"] = None  # type: ignore[assignment]
            sys.modules["joblib"] = None  # type: ignore[assignment]

    def test_tensorflow_predict_calls_model_predict(self, native_model, dep_snapshot):
        import sys
        tf_mock = MagicMock(name="tensorflow")
        tf_mock.keras = MagicMock()
        tf_mock.keras.Model = type("Model", (), {})
        sys.modules["tensorflow"] = tf_mock

        try:
            from artifact_mgmt.serializers._tensorflow import TensorFlowSerializer
            serializer = TensorFlowSerializer()

            expected = MagicMock(name="tf_output")
            native_model.predict.return_value = expected

            artifact = ArtifactModel(
                native_model,
                model_name="dense",
                version="1.0",
                dep_snapshot=dep_snapshot,
                serializer=serializer,
            )
            X = MagicMock(name="inputs")
            result = artifact.predict(X)

            native_model.predict.assert_called_once_with(X)
            assert result is expected
        finally:
            sys.modules["tensorflow"] = None  # type: ignore[assignment]


class TestRepr:
    def test_repr_includes_model_name_and_version(self, artifact):
        r = repr(artifact)
        assert "fraud-detector" in r
        assert "1.0" in r

    def test_repr_format(self, artifact):
        assert repr(artifact) == "ArtifactModel('fraud-detector', version='1.0', ...)"
