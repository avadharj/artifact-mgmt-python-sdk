from __future__ import annotations

from artifact_mgmt._exceptions import FrameworkNotInstalledError, UnknownSerializerError
from artifact_mgmt._types import DepSnapshot
from artifact_mgmt.serializers._base import Serializer


def _all_serializers() -> list[Serializer]:
    # Imported here so that loading this module never fails due to missing framework packages.
    from artifact_mgmt.serializers._torch import TorchSerializer
    from artifact_mgmt.serializers._huggingface import HuggingFaceSerializer
    from artifact_mgmt.serializers._sklearn import SklearnSerializer
    from artifact_mgmt.serializers._tensorflow import TensorFlowSerializer
    from artifact_mgmt.serializers._pickle import PickleSerializer

    return [
        TorchSerializer(),
        HuggingFaceSerializer(),
        SklearnSerializer(),
        TensorFlowSerializer(),
        PickleSerializer(),
    ]


_FRAMEWORK_NAME_MAP: dict[str, str] = {
    "pytorch": "pytorch",
    "huggingface": "huggingface",
    "sklearn": "sklearn",
    "tensorflow": "tensorflow",
    "pickle": "pickle",
}


class SerializerRegistry:
    @staticmethod
    def detect(model: object) -> Serializer:
        """Detect serializer from model type (for save_model)."""
        for serializer in _all_serializers():
            # PickleSerializer must never be auto-detected.
            if serializer.framework_name == "pickle":
                continue
            try:
                if serializer.can_handle(model):
                    return serializer
            except FrameworkNotInstalledError:
                continue
        raise UnknownSerializerError(
            f"No serializer found for model type {type(model).__name__!r}. "
            "Install the appropriate framework extra or pass serializer='pickle' explicitly."
        )

    @staticmethod
    def detect_from_snapshot(dep_snapshot: DepSnapshot) -> Serializer:
        """Detect serializer from dep_snapshot.framework.name (for load_model)."""
        framework = dep_snapshot.framework.name
        for serializer in _all_serializers():
            if serializer.framework_name == framework:
                return serializer
        raise UnknownSerializerError(
            f"No serializer registered for framework {framework!r}."
        )

    @staticmethod
    def get_by_name(name: str) -> Serializer:
        """Look up a serializer by explicit name (e.g. serializer='pickle')."""
        for serializer in _all_serializers():
            if serializer.framework_name == name:
                return serializer
        raise UnknownSerializerError(
            f"No serializer registered with name {name!r}."
        )
