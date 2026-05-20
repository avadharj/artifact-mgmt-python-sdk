from __future__ import annotations

import io
import os
import sys
import tarfile
from unittest.mock import MagicMock
import pytest

from artifact_mgmt._exceptions import FrameworkNotInstalledError
from artifact_mgmt.serializers._tensorflow import TensorFlowSerializer


# ---------------------------------------------------------------------------
# Helpers — minimal fake tensorflow module tree
# ---------------------------------------------------------------------------

def _make_tf_mock() -> MagicMock:
    tf = MagicMock(name="tensorflow")
    keras = MagicMock(name="tensorflow.keras")
    keras.Model = type("Model", (), {})
    keras.models = MagicMock(name="tensorflow.keras.models")
    tf.keras = keras
    return tf


def _install(tf_mock: MagicMock) -> None:
    sys.modules["tensorflow"] = tf_mock


def _remove() -> None:
    sys.modules["tensorflow"] = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tf_mock():
    mock = _make_tf_mock()
    _install(mock)
    yield mock
    _remove()


@pytest.fixture()
def serializer():
    return TensorFlowSerializer()


@pytest.fixture()
def keras_model(tf_mock):
    """Fake model that is an instance of the mocked keras.Model."""
    model = MagicMock()
    model.__class__ = tf_mock.keras.Model
    return model


def _make_tarball(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Story 3.5 — TensorFlowSerializer
# ---------------------------------------------------------------------------


class TestCanHandle:
    def test_returns_true_for_keras_model_instance(self, serializer, tf_mock, keras_model):
        assert serializer.can_handle(keras_model) is True

    def test_returns_false_for_non_keras_model(self, serializer, tf_mock):
        assert serializer.can_handle(object()) is False

    def test_raises_framework_not_installed_when_tensorflow_missing(self, serializer):
        _remove()
        with pytest.raises(
            FrameworkNotInstalledError,
            match="pip install artifact-mgmt-client\\[tensorflow\\]",
        ):
            serializer.can_handle(MagicMock())


class TestSerialize:
    def test_calls_model_save_and_returns_tarball_bytes(
        self, serializer, tf_mock, keras_model
    ):
        def fake_save(tmpdir: str, save_format: str) -> None:
            with open(os.path.join(tmpdir, "saved_model.pb"), "wb") as f:
                f.write(b"pb-data")

        keras_model.save.side_effect = fake_save
        result = serializer.serialize(keras_model)

        assert isinstance(result, bytes)
        assert len(result) > 0
        keras_model.save.assert_called_once()
        _, kwargs = keras_model.save.call_args
        assert kwargs.get("save_format") == "tf"

    def test_tarball_contains_saved_model_files(self, serializer, tf_mock, keras_model):
        def fake_save(tmpdir: str, save_format: str) -> None:
            with open(os.path.join(tmpdir, "saved_model.pb"), "wb") as f:
                f.write(b"pb-data")
            variables_dir = os.path.join(tmpdir, "variables")
            os.makedirs(variables_dir)
            with open(os.path.join(variables_dir, "variables.index"), "wb") as f:
                f.write(b"index")

        keras_model.save.side_effect = fake_save
        data = serializer.serialize(keras_model)

        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            names = {m.name for m in tar.getmembers()}
        assert any("saved_model.pb" in n for n in names)

    def test_raises_framework_not_installed_when_tensorflow_missing(self, serializer):
        _remove()
        with pytest.raises(FrameworkNotInstalledError):
            serializer.serialize(MagicMock())


class TestDeserialize:
    def test_extracts_tarball_and_calls_load_model(self, serializer, tf_mock):
        expected = MagicMock(name="loaded_model")
        tf_mock.keras.models.load_model.return_value = expected

        data = _make_tarball({"saved_model.pb": b"pb-data"})
        result = serializer.deserialize(data)

        assert result is expected
        tf_mock.keras.models.load_model.assert_called_once()

    def test_raises_framework_not_installed_when_tensorflow_missing(self, serializer):
        _remove()
        with pytest.raises(FrameworkNotInstalledError):
            serializer.deserialize(b"bytes")


class TestRoundTrip:
    def test_serialize_then_deserialize_returns_equivalent_model(
        self, serializer, tf_mock, keras_model
    ):
        expected = MagicMock(name="restored_model")
        tf_mock.keras.models.load_model.return_value = expected

        def fake_save(tmpdir: str, save_format: str) -> None:
            with open(os.path.join(tmpdir, "saved_model.pb"), "wb") as f:
                f.write(b"pb-data")

        keras_model.save.side_effect = fake_save

        data = serializer.serialize(keras_model)
        result = serializer.deserialize(data)

        assert result is expected


class TestFreeze:
    def test_sets_trainable_false_on_first_n_layers(
        self, serializer, tf_mock, keras_model
    ):
        layer0 = MagicMock()
        layer0.trainable = True
        layer1 = MagicMock()
        layer1.trainable = True
        layer2 = MagicMock()
        layer2.trainable = True
        keras_model.layers = [layer0, layer1, layer2]

        serializer.freeze(keras_model, 2)

        assert layer0.trainable is False
        assert layer1.trainable is False
        assert layer2.trainable is True  # untouched

    def test_freeze_zero_layers_freezes_nothing(self, serializer, tf_mock, keras_model):
        layer = MagicMock()
        layer.trainable = True
        keras_model.layers = [layer]

        serializer.freeze(keras_model, 0)
        assert layer.trainable is True

    def test_freeze_large_n_freezes_all(self, serializer, tf_mock, keras_model):
        layers = [MagicMock() for _ in range(3)]
        for layer in layers:
            layer.trainable = True
        keras_model.layers = layers

        serializer.freeze(keras_model, 999)
        for layer in layers:
            assert layer.trainable is False

    def test_raises_framework_not_installed_when_tensorflow_missing(self, serializer):
        _remove()
        with pytest.raises(FrameworkNotInstalledError):
            serializer.freeze(MagicMock(), 1)


class TestUnfreeze:
    def test_sets_trainable_true_on_first_n_layers(
        self, serializer, tf_mock, keras_model
    ):
        layer0 = MagicMock()
        layer0.trainable = False
        layer1 = MagicMock()
        layer1.trainable = False
        keras_model.layers = [layer0, layer1]

        serializer.unfreeze(keras_model, 1)

        assert layer0.trainable is True
        assert layer1.trainable is False  # untouched

    def test_raises_framework_not_installed_when_tensorflow_missing(self, serializer):
        _remove()
        with pytest.raises(FrameworkNotInstalledError):
            serializer.unfreeze(MagicMock(), 1)


class TestFineTuneParams:
    def test_returns_trainable_weights_from_all_layers(
        self, serializer, tf_mock, keras_model
    ):
        w0 = MagicMock(name="w0")
        w1 = MagicMock(name="w1")
        layer0 = MagicMock()
        layer0.trainable_weights = [w0]
        layer1 = MagicMock()
        layer1.trainable_weights = [w1]
        keras_model.layers = [layer0, layer1]

        result = serializer.fine_tune_params(keras_model)
        assert result == [w0, w1]

    def test_returns_empty_list_when_no_trainable_weights(
        self, serializer, tf_mock, keras_model
    ):
        layer = MagicMock()
        layer.trainable_weights = []
        keras_model.layers = [layer]

        assert serializer.fine_tune_params(keras_model) == []


class TestPredict:
    def test_calls_model_predict_and_returns_result(
        self, serializer, tf_mock, keras_model
    ):
        expected = MagicMock(name="predictions")
        keras_model.predict.return_value = expected

        X = MagicMock(name="inputs")
        result = serializer.predict(keras_model, X)

        keras_model.predict.assert_called_once_with(X)
        assert result is expected
