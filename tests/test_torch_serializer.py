from __future__ import annotations

import sys
from unittest.mock import MagicMock
import pytest

from artifact_mgmt._exceptions import FrameworkNotInstalledError
from artifact_mgmt.serializers._torch import TorchSerializer


# ---------------------------------------------------------------------------
# Helpers — build a minimal fake torch module tree
# ---------------------------------------------------------------------------

def _make_torch_mock() -> MagicMock:
    torch = MagicMock(name="torch")
    nn = MagicMock(name="torch.nn")
    nn.Module = type("Module", (), {})
    torch.nn = nn
    return torch


def _install_torch(torch_mock: MagicMock) -> None:
    sys.modules["torch"] = torch_mock
    sys.modules["torch.nn"] = torch_mock.nn


def _remove_torch() -> None:
    sys.modules.pop("torch", None)
    sys.modules.pop("torch.nn", None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def torch_mock():
    mock = _make_torch_mock()
    _install_torch(mock)
    yield mock
    _remove_torch()


@pytest.fixture()
def serializer():
    return TorchSerializer()


@pytest.fixture()
def nn_model(torch_mock):
    """A fake model that is an instance of the mocked nn.Module."""
    model = MagicMock()
    model.__class__ = torch_mock.nn.Module
    return model


# ---------------------------------------------------------------------------
# Story 3.2 — TorchSerializer
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_serialize_then_deserialize_returns_equivalent_model(self, serializer, torch_mock, nn_model):
        def fake_save(model, buf):
            buf.write(b"serialized-model")

        torch_mock.save.side_effect = fake_save
        torch_mock.load.return_value = nn_model

        data = serializer.serialize(nn_model)
        result = serializer.deserialize(data)

        assert result is nn_model
        torch_mock.save.assert_called_once()
        torch_mock.load.assert_called_once()


class TestCanHandle:
    def test_returns_true_for_nn_module_instance(self, serializer, torch_mock, nn_model):
        assert serializer.can_handle(nn_model) is True

    def test_returns_false_for_non_module(self, serializer, torch_mock):
        assert serializer.can_handle(object()) is False

    def test_raises_framework_not_installed_when_torch_missing(self, serializer):
        _remove_torch()
        with pytest.raises(FrameworkNotInstalledError, match="pip install artifact-mgmt-client\\[torch\\]"):
            serializer.can_handle(MagicMock())


class TestSerialize:
    def test_calls_torch_save_and_returns_bytes(self, serializer, torch_mock, nn_model):
        import io
        buf_holder: list[io.BytesIO] = []

        def fake_save(model, buf):
            buf.write(b"fake-model-bytes")
            buf_holder.append(buf)

        torch_mock.save.side_effect = fake_save
        result = serializer.serialize(nn_model)
        assert result == b"fake-model-bytes"
        torch_mock.save.assert_called_once()

    def test_raises_framework_not_installed_when_torch_missing(self, serializer):
        _remove_torch()
        with pytest.raises(FrameworkNotInstalledError):
            serializer.serialize(MagicMock())


class TestDeserialize:
    def test_returns_loaded_object(self, serializer, torch_mock):
        expected = MagicMock(name="loaded_model")
        torch_mock.load.return_value = expected
        result = serializer.deserialize(b"some-bytes")
        assert result is expected
        torch_mock.load.assert_called_once()

    def test_lightning_ckpt_extracts_state_dict(self, serializer, torch_mock):
        state_dict = {"layer.weight": MagicMock()}
        torch_mock.load.return_value = {"state_dict": state_dict, "epoch": 5}
        result = serializer.deserialize(b"ckpt-bytes")
        assert result is state_dict

    def test_non_ckpt_dict_returned_as_is(self, serializer, torch_mock):
        payload = {"config": "value"}
        torch_mock.load.return_value = payload
        result = serializer.deserialize(b"bytes")
        assert result is payload

    def test_raises_framework_not_installed_when_torch_missing(self, serializer):
        _remove_torch()
        with pytest.raises(FrameworkNotInstalledError):
            serializer.deserialize(b"bytes")


class TestFreeze:
    def test_freeze_sets_requires_grad_false_on_first_n_params(self, serializer, torch_mock, nn_model):
        p0, p1, p2 = MagicMock(), MagicMock(), MagicMock()
        p0.requires_grad = True
        p1.requires_grad = True
        p2.requires_grad = True
        nn_model.named_parameters.return_value = [("p0", p0), ("p1", p1), ("p2", p2)]
        serializer.freeze(nn_model, 2)
        assert p0.requires_grad is False
        assert p1.requires_grad is False
        assert p2.requires_grad is True  # untouched

    def test_freeze_zero_layers_freezes_nothing(self, serializer, torch_mock, nn_model):
        p = MagicMock()
        p.requires_grad = True
        nn_model.named_parameters.return_value = [("p", p)]
        serializer.freeze(nn_model, 0)
        assert p.requires_grad is True

    def test_freeze_large_n_freezes_all(self, serializer, torch_mock, nn_model):
        params = [MagicMock() for _ in range(3)]
        for p in params:
            p.requires_grad = True
        nn_model.named_parameters.return_value = [(f"p{i}", p) for i, p in enumerate(params)]
        serializer.freeze(nn_model, 999)
        for p in params:
            assert p.requires_grad is False

    def test_raises_framework_not_installed_when_torch_missing(self, serializer):
        _remove_torch()
        with pytest.raises(FrameworkNotInstalledError):
            serializer.freeze(MagicMock(), 1)


class TestUnfreeze:
    def test_unfreeze_sets_requires_grad_true_on_first_n_params(self, serializer, torch_mock, nn_model):
        p0, p1, p2 = MagicMock(), MagicMock(), MagicMock()
        p0.requires_grad = False
        p1.requires_grad = False
        p2.requires_grad = False
        nn_model.named_parameters.return_value = [("p0", p0), ("p1", p1), ("p2", p2)]
        serializer.unfreeze(nn_model, 2)
        assert p0.requires_grad is True
        assert p1.requires_grad is True
        assert p2.requires_grad is False  # untouched

    def test_raises_framework_not_installed_when_torch_missing(self, serializer):
        _remove_torch()
        with pytest.raises(FrameworkNotInstalledError):
            serializer.unfreeze(MagicMock(), 1)


class TestFineTuneParams:
    def test_returns_only_trainable_params(self, serializer, torch_mock, nn_model):
        p_frozen = MagicMock()
        p_frozen.requires_grad = False
        p_trainable = MagicMock()
        p_trainable.requires_grad = True
        nn_model.parameters.return_value = [p_frozen, p_trainable]
        result = serializer.fine_tune_params(nn_model)
        assert result == [p_trainable]

    def test_returns_empty_list_when_all_frozen(self, serializer, torch_mock, nn_model):
        p = MagicMock()
        p.requires_grad = False
        nn_model.parameters.return_value = [p]
        assert serializer.fine_tune_params(nn_model) == []


class TestPredict:
    def test_calls_eval_and_no_grad_and_returns_output(self, serializer, torch_mock, nn_model):
        expected_output = MagicMock(name="output")
        nn_model.return_value = expected_output

        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=None)
        ctx.__exit__ = MagicMock(return_value=False)
        torch_mock.no_grad.return_value = ctx

        X = MagicMock(name="input")
        result = serializer.predict(nn_model, X)

        nn_model.eval.assert_called_once()
        torch_mock.no_grad.assert_called_once()
        nn_model.assert_called_once_with(X)
        assert result is expected_output

    def test_raises_framework_not_installed_when_torch_missing(self, serializer):
        _remove_torch()
        with pytest.raises(FrameworkNotInstalledError):
            serializer.predict(MagicMock(), MagicMock())
