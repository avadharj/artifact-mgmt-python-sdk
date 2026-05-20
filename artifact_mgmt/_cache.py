from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from artifact_mgmt._artifact_model import ArtifactModel


class ModelCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self._root = Path(cache_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    def _weights_path(self, model_name: str, version: str) -> Path:
        return self._root / model_name / version / "weights"

    def _meta_path(self, model_name: str, version: str) -> Path:
        return self._root / model_name / version / "meta.json"

    def get(self, model_name: str, version: str) -> ArtifactModel | None:
        weights_path = self._weights_path(model_name, version)
        meta_path = self._meta_path(model_name, version)
        if not weights_path.exists() or not meta_path.exists():
            return None

        from artifact_mgmt._artifact_model import ArtifactModel
        from artifact_mgmt._types import DepSnapshot, FrameworkInfo
        from artifact_mgmt.serializers import SerializerRegistry

        data = weights_path.read_bytes()
        meta = json.loads(meta_path.read_text())

        fw = meta["framework"]
        dep_snapshot = DepSnapshot(
            python_version=meta["pythonVersion"],
            framework=FrameworkInfo(name=fw["name"], version=fw["version"]),
            packages=meta.get("packages", {}),
            os=meta.get("os", ""),
            captured_at=meta.get("capturedAt", ""),
            cuda_version=meta.get("cudaVersion"),
        )

        serializer = SerializerRegistry.detect_from_snapshot(dep_snapshot)
        model = serializer.deserialize(data)

        return ArtifactModel(
            model,
            model_name=model_name,
            version=version,
            dep_snapshot=dep_snapshot,
            serializer=serializer,
        )

    def put(self, model_name: str, version: str, artifact: ArtifactModel) -> None:
        weights_path = self._weights_path(model_name, version)
        meta_path = self._meta_path(model_name, version)
        weights_path.parent.mkdir(parents=True, exist_ok=True)

        snap = artifact.dep_snapshot
        data = artifact._serializer.serialize(artifact.model)
        weights_path.write_bytes(data)

        meta: dict[str, object] = {
            "pythonVersion": snap.python_version,
            "framework": {"name": snap.framework.name, "version": snap.framework.version},
            "packages": snap.packages,
            "os": snap.os,
            "capturedAt": snap.captured_at,
        }
        if snap.cuda_version is not None:
            meta["cudaVersion"] = snap.cuda_version
        meta_path.write_text(json.dumps(meta))

    def invalidate(self, model_name: str, version: str) -> None:
        import shutil
        entry_dir = self._root / model_name / version
        if entry_dir.exists():
            shutil.rmtree(entry_dir)

    def clear(self) -> None:
        import shutil
        for child in self._root.iterdir():
            shutil.rmtree(child)
