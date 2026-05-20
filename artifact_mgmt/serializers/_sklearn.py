from __future__ import annotations

from artifact_mgmt._exceptions import FrameworkNotInstalledError
from artifact_mgmt.serializers._base import Serializer


class SklearnSerializer(Serializer):
    framework_name = "sklearn"

    def can_handle(self, model: object) -> bool:
        try:
            from sklearn.base import BaseEstimator  # noqa: F401
        except ImportError:
            raise FrameworkNotInstalledError(
                "scikit-learn is not installed. "
                "Install it with: pip install artifact-mgmt-client[sklearn]"
            )
        from sklearn.base import BaseEstimator
        return isinstance(model, BaseEstimator)

    def serialize(self, model: object) -> bytes:
        try:
            import joblib
        except ImportError:
            raise FrameworkNotInstalledError("joblib is not installed.")
        import io
        buf = io.BytesIO()
        joblib.dump(model, buf)
        return buf.getvalue()

    def deserialize(self, data: bytes) -> object:
        try:
            import joblib
        except ImportError:
            raise FrameworkNotInstalledError("joblib is not installed.")
        import io
        return joblib.load(io.BytesIO(data))

    def freeze(self, model: object, n_layers: int) -> None:
        raise NotImplementedError(
            "sklearn estimators do not support layer freezing."
        )

    def unfreeze(self, model: object, n_layers: int) -> None:
        raise NotImplementedError(
            "sklearn estimators do not support layer freezing."
        )

    def fine_tune_params(self, model: object) -> list:  # type: ignore[type-arg]
        raise NotImplementedError(
            "sklearn estimators do not support layer freezing."
        )

    def predict(self, model: object, X: object) -> object:
        return model.predict(X)  # type: ignore[union-attr]
