from __future__ import annotations

from abc import ABC, abstractmethod


class Serializer(ABC):
    framework_name: str

    @abstractmethod
    def can_handle(self, model: object) -> bool: ...

    @abstractmethod
    def serialize(self, model: object) -> bytes: ...

    @abstractmethod
    def deserialize(self, data: bytes) -> object: ...

    @abstractmethod
    def freeze(self, model: object, n_layers: int) -> None: ...

    @abstractmethod
    def unfreeze(self, model: object, n_layers: int) -> None: ...

    @abstractmethod
    def fine_tune_params(self, model: object) -> list: ...  # type: ignore[type-arg]

    @abstractmethod
    def predict(self, model: object, X: object) -> object: ...
