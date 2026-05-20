from __future__ import annotations

from artifact_mgmt._exceptions import FrameworkNotInstalledError
from artifact_mgmt.serializers._base import Serializer


class TensorFlowSerializer(Serializer):
    framework_name = "tensorflow"

    def can_handle(self, model: object) -> bool:
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError:
            raise FrameworkNotInstalledError(
                "tensorflow is not installed. "
                "Install it with: pip install artifact-mgmt-client[tensorflow]"
            )
        import tensorflow as tf
        return isinstance(model, tf.keras.Model)

    def serialize(self, model: object) -> bytes:
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError:
            raise FrameworkNotInstalledError("tensorflow is not installed.")
        import io
        import tarfile
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir, save_format="tf")  # type: ignore[union-attr]
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                tar.add(tmpdir, arcname=".")
            return buf.getvalue()

    def deserialize(self, data: bytes) -> object:
        try:
            import tensorflow as tf
        except ImportError:
            raise FrameworkNotInstalledError("tensorflow is not installed.")
        import io
        import tarfile
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                tar.extractall(tmpdir, filter="data")
            return tf.keras.models.load_model(tmpdir)

    def freeze(self, model: object, n_layers: int) -> None:
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError:
            raise FrameworkNotInstalledError("tensorflow is not installed.")
        for layer in model.layers[:n_layers]:  # type: ignore[union-attr]
            layer.trainable = False

    def unfreeze(self, model: object, n_layers: int) -> None:
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError:
            raise FrameworkNotInstalledError("tensorflow is not installed.")
        for layer in model.layers[:n_layers]:  # type: ignore[union-attr]
            layer.trainable = True

    def fine_tune_params(self, model: object) -> list:  # type: ignore[type-arg]
        return [w for layer in model.layers for w in layer.trainable_weights]  # type: ignore[union-attr]

    def predict(self, model: object, X: object) -> object:
        return model.predict(X)  # type: ignore[union-attr]
