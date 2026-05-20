from __future__ import annotations

from typing import Any

from artifact_mgmt._types import DepSnapshot
from artifact_mgmt.serializers._base import Serializer


class ArtifactModel:
    def __init__(
        self,
        model: object,
        *,
        model_name: str,
        version: str,
        dep_snapshot: DepSnapshot,
        serializer: Serializer,
    ) -> None:
        self._model = model
        self._model_name = model_name
        self._version = version
        self._dep_snapshot = dep_snapshot
        self._serializer = serializer

    @property
    def model(self) -> object:
        return self._model

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def version(self) -> str:
        return self._version

    @property
    def dep_snapshot(self) -> DepSnapshot:
        return self._dep_snapshot

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def __repr__(self) -> str:
        return f"ArtifactModel({self.model_name!r}, version={self.version!r}, ...)"
