from __future__ import annotations

from artifact_mgmt._exceptions import FrameworkNotInstalledError
from artifact_mgmt.serializers._base import Serializer


class TorchSerializer(Serializer):
    framework_name = "pytorch"

    def can_handle(self, model: object) -> bool:
        try:
            import torch.nn as nn  # noqa: F401
        except ImportError:
            raise FrameworkNotInstalledError(
                "torch is not installed. Install it with: pip install artifact-mgmt-client[torch]"
            )
        import torch.nn as nn
        return isinstance(model, nn.Module)

    def serialize(self, model: object) -> bytes:
        try:
            import torch
        except ImportError:
            raise FrameworkNotInstalledError("torch is not installed.")
        import io
        buf = io.BytesIO()
        torch.save(model, buf)
        return buf.getvalue()

    def deserialize(self, data: bytes) -> object:
        try:
            import torch
        except ImportError:
            raise FrameworkNotInstalledError("torch is not installed.")
        import io
        return torch.load(io.BytesIO(data), weights_only=False)

    def freeze(self, model: object, n_layers: int) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            raise FrameworkNotInstalledError("torch is not installed.")
        params = list(model.named_parameters())  # type: ignore[union-attr]
        for _, p in params[:n_layers]:
            p.requires_grad = False

    def unfreeze(self, model: object, n_layers: int) -> None:
        try:
            import torch.nn as nn  # noqa: F401
        except ImportError:
            raise FrameworkNotInstalledError("torch is not installed.")
        params = list(model.named_parameters())  # type: ignore[union-attr]
        for _, p in params[:n_layers]:
            p.requires_grad = True

    def fine_tune_params(self, model: object) -> list:  # type: ignore[type-arg]
        return [p for p in model.parameters() if p.requires_grad]  # type: ignore[union-attr]

    def predict(self, model: object, X: object) -> object:
        try:
            import torch
        except ImportError:
            raise FrameworkNotInstalledError("torch is not installed.")
        model.eval()  # type: ignore[union-attr]
        with torch.no_grad():
            return model(X)  # type: ignore[operator]
