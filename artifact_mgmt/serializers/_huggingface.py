from __future__ import annotations

from artifact_mgmt._exceptions import FrameworkNotInstalledError
from artifact_mgmt.serializers._base import Serializer


class HuggingFaceSerializer(Serializer):
    framework_name = "huggingface"

    def can_handle(self, model: object) -> bool:
        try:
            from transformers import PreTrainedModel  # noqa: F401
        except ImportError:
            raise FrameworkNotInstalledError(
                "transformers is not installed. "
                "Install it with: pip install artifact-mgmt-client[transformers]"
            )
        from transformers import PreTrainedModel
        return isinstance(model, PreTrainedModel)

    def serialize(self, model: object) -> bytes:
        try:
            import transformers  # noqa: F401
        except ImportError:
            raise FrameworkNotInstalledError("transformers is not installed.")
        import io
        import tarfile
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)  # type: ignore[union-attr]
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                for fname in os.listdir(tmpdir):
                    tar.add(os.path.join(tmpdir, fname), arcname=fname)
            return buf.getvalue()

    def deserialize(self, data: bytes) -> object:
        try:
            from transformers import AutoModel  # noqa: F401
        except ImportError:
            raise FrameworkNotInstalledError("transformers is not installed.")
        from transformers import AutoModel
        import io
        import tarfile
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                tar.extractall(tmpdir)
            return AutoModel.from_pretrained(tmpdir)

    def freeze(self, model: object, n_layers: int) -> None:
        try:
            import transformers  # noqa: F401
        except ImportError:
            raise FrameworkNotInstalledError("transformers is not installed.")
        # Try BERT-style encoder layers first; fall back to named children.
        try:
            layers = model.base_model.encoder.layer  # type: ignore[union-attr]
            for layer in layers[:n_layers]:
                for p in layer.parameters():
                    p.requires_grad = False
        except AttributeError:
            children = list(model.children())  # type: ignore[union-attr]
            for child in children[:n_layers]:
                for p in child.parameters():
                    p.requires_grad = False

    def unfreeze(self, model: object, n_layers: int) -> None:
        try:
            import transformers  # noqa: F401
        except ImportError:
            raise FrameworkNotInstalledError("transformers is not installed.")
        try:
            layers = model.base_model.encoder.layer  # type: ignore[union-attr]
            for layer in layers[:n_layers]:
                for p in layer.parameters():
                    p.requires_grad = True
        except AttributeError:
            children = list(model.children())  # type: ignore[union-attr]
            for child in children[:n_layers]:
                for p in child.parameters():
                    p.requires_grad = True

    def fine_tune_params(self, model: object) -> list:  # type: ignore[type-arg]
        return [p for p in model.parameters() if p.requires_grad]  # type: ignore[union-attr]

    def predict(self, model: object, X: object) -> object:
        return model(**X)  # type: ignore[operator]
