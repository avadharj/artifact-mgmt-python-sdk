from __future__ import annotations

import pickle
from unittest.mock import MagicMock
import pytest

from artifact_mgmt.serializers._pickle import PickleSerializer
from artifact_mgmt.serializers import SerializerRegistry


# ---------------------------------------------------------------------------
# Story 3.6 — PickleSerializer
# ---------------------------------------------------------------------------


@pytest.fixture()
def serializer():
    return PickleSerializer()


class TestCanHandle:
    def test_always_returns_false(self, serializer):
        assert serializer.can_handle(MagicMock()) is False

    def test_returns_false_for_any_type(self, serializer):
        for obj in [42, "string", [], {}, object()]:
            assert serializer.can_handle(obj) is False


class TestSerialize:
    def test_returns_pickle_bytes_for_plain_object(self, serializer):
        obj = {"key": "value", "number": 42}
        result = serializer.serialize(obj)
        assert isinstance(result, bytes)
        assert pickle.loads(result) == obj  # noqa: S301

    def test_serialize_produces_deserializable_bytes(self, serializer):
        obj = [1, 2, 3]
        data = serializer.serialize(obj)
        assert pickle.loads(data) == [1, 2, 3]  # noqa: S301


class TestDeserialize:
    def test_returns_original_object_from_pickle_bytes(self, serializer):
        obj = {"model": "params", "score": 0.95}
        data = pickle.dumps(obj)
        result = serializer.deserialize(data)
        assert result == obj


class TestRoundTrip:
    def test_serialize_then_deserialize_plain_python_object(self, serializer):
        obj = {"weights": [0.1, 0.2, 0.3], "bias": 0.5}
        result = serializer.deserialize(serializer.serialize(obj))
        assert result == obj

    def test_round_trip_with_nested_structure(self, serializer):
        obj = {"layers": [{"w": [1, 2]}, {"w": [3, 4]}]}
        assert serializer.deserialize(serializer.serialize(obj)) == obj


class TestTrainingMethodsRaiseNotImplemented:
    def test_freeze_raises_not_implemented(self, serializer):
        with pytest.raises(NotImplementedError):
            serializer.freeze(MagicMock(), 1)

    def test_unfreeze_raises_not_implemented(self, serializer):
        with pytest.raises(NotImplementedError):
            serializer.unfreeze(MagicMock(), 1)

    def test_fine_tune_params_raises_not_implemented(self, serializer):
        with pytest.raises(NotImplementedError):
            serializer.fine_tune_params(MagicMock())

    def test_predict_raises_not_implemented(self, serializer):
        with pytest.raises(NotImplementedError):
            serializer.predict(MagicMock(), MagicMock())


class TestNeverAutoDetected:
    def test_registry_detect_never_returns_pickle_serializer(self):
        from unittest.mock import patch
        mock_model = MagicMock()
        with (
            patch("artifact_mgmt.serializers._torch.TorchSerializer.can_handle", return_value=False),
            patch("artifact_mgmt.serializers._huggingface.HuggingFaceSerializer.can_handle", return_value=False),
            patch("artifact_mgmt.serializers._sklearn.SklearnSerializer.can_handle", return_value=False),
            patch("artifact_mgmt.serializers._tensorflow.TensorFlowSerializer.can_handle", return_value=False),
        ):
            from artifact_mgmt._exceptions import UnknownSerializerError
            with pytest.raises(UnknownSerializerError):
                SerializerRegistry.detect(mock_model)

    def test_get_by_name_pickle_returns_pickle_serializer(self):
        result = SerializerRegistry.get_by_name("pickle")
        assert result.framework_name == "pickle"
        assert isinstance(result, PickleSerializer)
