from __future__ import annotations

import io
import os
import sys
import tarfile
from unittest.mock import MagicMock
import pytest

from artifact_mgmt._exceptions import FrameworkNotInstalledError
from artifact_mgmt.serializers._huggingface import HuggingFaceSerializer


# ---------------------------------------------------------------------------
# Helpers — build a minimal fake transformers module
# ---------------------------------------------------------------------------

def _make_transformers_mock() -> MagicMock:
    transformers = MagicMock(name="transformers")
    transformers.PreTrainedModel = type("PreTrainedModel", (), {})
    transformers.AutoModel = MagicMock(name="AutoModel")
    return transformers


def _install_transformers(mock: MagicMock) -> None:
    sys.modules["transformers"] = mock


def _remove_transformers() -> None:
    sys.modules.pop("transformers", None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def transformers_mock():
    mock = _make_transformers_mock()
    _install_transformers(mock)
    yield mock
    _remove_transformers()


@pytest.fixture()
def serializer():
    return HuggingFaceSerializer()


@pytest.fixture()
def hf_model(transformers_mock):
    """Fake model that is an instance of the mocked PreTrainedModel."""
    model = MagicMock()
    model.__class__ = transformers_mock.PreTrainedModel
    return model


def _make_tarball(files: dict[str, bytes]) -> bytes:
    """Build an in-memory .tar.gz with the given filename→content mapping."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Story 3.3 — HuggingFaceSerializer
# ---------------------------------------------------------------------------


class TestCanHandle:
    def test_returns_true_for_pretrained_model_instance(self, serializer, transformers_mock, hf_model):
        assert serializer.can_handle(hf_model) is True

    def test_returns_false_for_non_pretrained_model(self, serializer, transformers_mock):
        assert serializer.can_handle(object()) is False

    def test_raises_framework_not_installed_when_transformers_missing(self, serializer):
        _remove_transformers()
        with pytest.raises(
            FrameworkNotInstalledError,
            match="pip install artifact-mgmt-client\\[transformers\\]",
        ):
            serializer.can_handle(MagicMock())


class TestSerialize:
    def test_calls_save_pretrained_and_returns_tarball_bytes(
        self, serializer, transformers_mock, hf_model
    ):
        def fake_save_pretrained(tmpdir: str) -> None:
            # Write a fake config file so the tarball has content
            with open(os.path.join(tmpdir, "config.json"), "wb") as f:
                f.write(b'{"model_type": "bert"}')

        hf_model.save_pretrained.side_effect = fake_save_pretrained
        result = serializer.serialize(hf_model)

        assert isinstance(result, bytes)
        assert len(result) > 0
        hf_model.save_pretrained.assert_called_once()

    def test_tarball_contains_config_json(self, serializer, transformers_mock, hf_model):
        def fake_save_pretrained(tmpdir: str) -> None:
            with open(os.path.join(tmpdir, "config.json"), "wb") as f:
                f.write(b'{"model_type": "bert"}')
            with open(os.path.join(tmpdir, "pytorch_model.bin"), "wb") as f:
                f.write(b"weights")

        hf_model.save_pretrained.side_effect = fake_save_pretrained
        data = serializer.serialize(hf_model)

        names = set()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            names = {m.name for m in tar.getmembers()}
        assert "config.json" in names
        assert "pytorch_model.bin" in names

    def test_tarball_includes_tokenizer_files_when_present(
        self, serializer, transformers_mock, hf_model
    ):
        def fake_save_pretrained(tmpdir: str) -> None:
            for fname in ("config.json", "tokenizer_config.json", "vocab.txt"):
                with open(os.path.join(tmpdir, fname), "wb") as f:
                    f.write(b"data")

        hf_model.save_pretrained.side_effect = fake_save_pretrained
        data = serializer.serialize(hf_model)

        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            names = {m.name for m in tar.getmembers()}
        assert "tokenizer_config.json" in names
        assert "vocab.txt" in names

    def test_raises_framework_not_installed_when_transformers_missing(self, serializer):
        _remove_transformers()
        with pytest.raises(FrameworkNotInstalledError):
            serializer.serialize(MagicMock())


class TestDeserialize:
    def test_extracts_tarball_and_calls_from_pretrained(
        self, serializer, transformers_mock
    ):
        expected_model = MagicMock(name="loaded_model")
        transformers_mock.AutoModel.from_pretrained.return_value = expected_model

        data = _make_tarball({"config.json": b'{"model_type": "bert"}'})
        result = serializer.deserialize(data)

        assert result is expected_model
        transformers_mock.AutoModel.from_pretrained.assert_called_once()

    def test_raises_framework_not_installed_when_transformers_missing(self, serializer):
        _remove_transformers()
        with pytest.raises(FrameworkNotInstalledError):
            serializer.deserialize(b"bytes")


class TestRoundTrip:
    def test_serialize_then_deserialize_returns_equivalent_model(
        self, serializer, transformers_mock, hf_model
    ):
        expected_model = MagicMock(name="restored_model")
        transformers_mock.AutoModel.from_pretrained.return_value = expected_model

        def fake_save_pretrained(tmpdir: str) -> None:
            with open(os.path.join(tmpdir, "config.json"), "wb") as f:
                f.write(b'{"model_type": "bert"}')

        hf_model.save_pretrained.side_effect = fake_save_pretrained

        data = serializer.serialize(hf_model)
        result = serializer.deserialize(data)

        assert result is expected_model


class TestFreeze:
    def test_bert_style_freeze_uses_encoder_layers(
        self, serializer, transformers_mock, hf_model
    ):
        p0, p1, p2 = MagicMock(), MagicMock(), MagicMock()
        for p in (p0, p1, p2):
            p.requires_grad = True

        layer0 = MagicMock()
        layer0.parameters.return_value = [p0]
        layer1 = MagicMock()
        layer1.parameters.return_value = [p1]
        layer2 = MagicMock()
        layer2.parameters.return_value = [p2]

        hf_model.base_model.encoder.layer = [layer0, layer1, layer2]
        serializer.freeze(hf_model, 2)

        assert p0.requires_grad is False
        assert p1.requires_grad is False
        assert p2.requires_grad is True  # untouched

    def test_generic_fallback_freeze_uses_named_children(
        self, serializer, transformers_mock, hf_model
    ):
        p0, p1 = MagicMock(), MagicMock()
        p0.requires_grad = True
        p1.requires_grad = True

        child0 = MagicMock()
        child0.parameters.return_value = [p0]
        child1 = MagicMock()
        child1.parameters.return_value = [p1]

        # No BERT-style encoder — raise AttributeError on access
        del hf_model.base_model
        hf_model.children.return_value = [child0, child1]

        serializer.freeze(hf_model, 1)

        assert p0.requires_grad is False
        assert p1.requires_grad is True  # untouched

    def test_raises_framework_not_installed_when_transformers_missing(self, serializer):
        _remove_transformers()
        with pytest.raises(FrameworkNotInstalledError):
            serializer.freeze(MagicMock(), 1)


class TestUnfreeze:
    def test_bert_style_unfreeze_sets_requires_grad_true(
        self, serializer, transformers_mock, hf_model
    ):
        p0 = MagicMock()
        p0.requires_grad = False
        layer0 = MagicMock()
        layer0.parameters.return_value = [p0]
        hf_model.base_model.encoder.layer = [layer0]

        serializer.unfreeze(hf_model, 1)
        assert p0.requires_grad is True

    def test_generic_fallback_unfreeze(self, serializer, transformers_mock, hf_model):
        p0 = MagicMock()
        p0.requires_grad = False
        child0 = MagicMock()
        child0.parameters.return_value = [p0]
        del hf_model.base_model
        hf_model.children.return_value = [child0]

        serializer.unfreeze(hf_model, 1)
        assert p0.requires_grad is True

    def test_raises_framework_not_installed_when_transformers_missing(self, serializer):
        _remove_transformers()
        with pytest.raises(FrameworkNotInstalledError):
            serializer.unfreeze(MagicMock(), 1)


class TestFineTuneParams:
    def test_returns_only_trainable_params(self, serializer, transformers_mock, hf_model):
        p_frozen = MagicMock()
        p_frozen.requires_grad = False
        p_trainable = MagicMock()
        p_trainable.requires_grad = True
        hf_model.parameters.return_value = [p_frozen, p_trainable]

        result = serializer.fine_tune_params(hf_model)
        assert result == [p_trainable]


class TestPredict:
    def test_calls_model_with_unpacked_dict(self, serializer, transformers_mock, hf_model):
        expected = MagicMock(name="output")
        hf_model.return_value = expected

        X = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        result = serializer.predict(hf_model, X)

        hf_model.assert_called_once_with(**X)
        assert result is expected
