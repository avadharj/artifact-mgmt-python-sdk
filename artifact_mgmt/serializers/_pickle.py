from __future__ import annotations

import pickle

from artifact_mgmt.serializers._base import Serializer


class PickleSerializer(Serializer):
    framework_name = "pickle"

    def can_handle(self, model: object) -> bool:
        # Never auto-detected — only used via explicit serializer="pickle".
        return False

    def serialize(self, model: object) -> bytes:
        return pickle.dumps(model)

    def deserialize(self, data: bytes) -> object:
        return pickle.loads(data)  # noqa: S301

    def freeze(self, model: object, n_layers: int) -> None:
        raise NotImplementedError(
            "pickle serializer does not support layer freezing."
        )

    def unfreeze(self, model: object, n_layers: int) -> None:
        raise NotImplementedError(
            "pickle serializer does not support layer freezing."
        )

    def fine_tune_params(self, model: object) -> list:  # type: ignore[type-arg]
        raise NotImplementedError(
            "pickle serializer does not support fine_tune_params."
        )

    def predict(self, model: object, X: object) -> object:
        raise NotImplementedError(
            "pickle serializer does not support predict."
        )
