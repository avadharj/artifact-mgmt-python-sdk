from __future__ import annotations

import sys
from unittest.mock import MagicMock
import pytest

from artifact_mgmt._exceptions import FrameworkNotInstalledError
from artifact_mgmt.serializers._sklearn import SklearnSerializer


# ---------------------------------------------------------------------------
# Helpers — minimal fake sklearn + joblib module trees
# ---------------------------------------------------------------------------

def _make_sklearn_mock() -> MagicMock:
    sklearn = MagicMock(name="sklearn")
    base = MagicMock(name="sklearn.base")
    base.BaseEstimator = type("BaseEstimator", (), {})
    sklearn.base = base
    return sklearn


def _make_joblib_mock() -> MagicMock:
    return MagicMock(name="joblib")


def _install(sklearn_mock: MagicMock, joblib_mock: MagicMock) -> None:
    sys.modules["sklearn"] = sklearn_mock
    sys.modules["sklearn.base"] = sklearn_mock.base
    sys.modules["joblib"] = joblib_mock


def _remove() -> None:
    sys.modules.pop("sklearn", None)
    sys.modules.pop("sklearn.base", None)
    sys.modules.pop("joblib", None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sklearn_mock():
    return _make_sklearn_mock()


@pytest.fixture()
def joblib_mock():
    return _make_joblib_mock()


@pytest.fixture(autouse=True)
def install_mocks(sklearn_mock, joblib_mock):
    _install(sklearn_mock, joblib_mock)
    yield
    _remove()


@pytest.fixture()
def serializer():
    return SklearnSerializer()


@pytest.fixture()
def estimator(sklearn_mock):
    """Fake model that is an instance of the mocked BaseEstimator."""
    model = MagicMock()
    model.__class__ = sklearn_mock.base.BaseEstimator
    return model


# ---------------------------------------------------------------------------
# Story 3.4 — SklearnSerializer
# ---------------------------------------------------------------------------


class TestCanHandle:
    def test_returns_true_for_base_estimator_instance(self, serializer, estimator):
        assert serializer.can_handle(estimator) is True

    def test_returns_false_for_non_estimator(self, serializer):
        assert serializer.can_handle(object()) is False

    def test_raises_framework_not_installed_when_sklearn_missing(self, serializer):
        sys.modules["sklearn"] = None  # type: ignore[assignment]
        sys.modules["sklearn.base"] = None  # type: ignore[assignment]
        with pytest.raises(
            FrameworkNotInstalledError,
            match="pip install artifact-mgmt-client\\[sklearn\\]",
        ):
            serializer.can_handle(MagicMock())


class TestSerialize:
    def test_calls_joblib_dump_and_returns_bytes(self, serializer, joblib_mock, estimator):
        def fake_dump(model, buf):
            buf.write(b"serialized-estimator")

        joblib_mock.dump.side_effect = fake_dump
        result = serializer.serialize(estimator)

        assert result == b"serialized-estimator"
        joblib_mock.dump.assert_called_once()

    def test_raises_framework_not_installed_when_joblib_missing(self, serializer):
        sys.modules["joblib"] = None  # type: ignore[assignment]
        with pytest.raises(FrameworkNotInstalledError):
            serializer.serialize(MagicMock())


class TestDeserialize:
    def test_calls_joblib_load_and_returns_model(self, serializer, joblib_mock):
        expected = MagicMock(name="loaded_estimator")
        joblib_mock.load.return_value = expected

        result = serializer.deserialize(b"some-bytes")

        assert result is expected
        joblib_mock.load.assert_called_once()

    def test_raises_framework_not_installed_when_joblib_missing(self, serializer):
        sys.modules["joblib"] = None  # type: ignore[assignment]
        with pytest.raises(FrameworkNotInstalledError):
            serializer.deserialize(b"bytes")


class TestRoundTrip:
    def test_serialize_then_deserialize_returns_equivalent_model(
        self, serializer, joblib_mock, estimator
    ):
        def fake_dump(model, buf):
            buf.write(b"estimator-bytes")

        joblib_mock.dump.side_effect = fake_dump
        joblib_mock.load.return_value = estimator

        data = serializer.serialize(estimator)
        result = serializer.deserialize(data)

        assert result is estimator
        joblib_mock.dump.assert_called_once()
        joblib_mock.load.assert_called_once()


class TestFreeze:
    def test_raises_not_implemented_error(self, serializer, estimator):
        with pytest.raises(NotImplementedError, match="sklearn estimators do not support layer freezing"):
            serializer.freeze(estimator, 1)


class TestUnfreeze:
    def test_raises_not_implemented_error(self, serializer, estimator):
        with pytest.raises(NotImplementedError, match="sklearn estimators do not support layer freezing"):
            serializer.unfreeze(estimator, 1)


class TestFineTuneParams:
    def test_raises_not_implemented_error(self, serializer, estimator):
        with pytest.raises(NotImplementedError, match="sklearn estimators do not support layer freezing"):
            serializer.fine_tune_params(estimator)


class TestPredict:
    def test_calls_model_predict_and_returns_result(self, serializer, estimator):
        expected = MagicMock(name="predictions")
        estimator.predict.return_value = expected

        X = MagicMock(name="features")
        result = serializer.predict(estimator, X)

        estimator.predict.assert_called_once_with(X)
        assert result is expected
