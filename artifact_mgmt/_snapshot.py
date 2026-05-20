from __future__ import annotations

import importlib.metadata
import platform
from datetime import datetime, timezone
from typing import Any

from artifact_mgmt._types import DepSnapshot, FrameworkInfo
from artifact_mgmt.serializers import SerializerRegistry


def capture(model: object, override: dict[str, Any] | None = None) -> DepSnapshot:
    """Inspect the current Python environment and return a DepSnapshot."""
    serializer = SerializerRegistry.detect(model)
    framework_name = serializer.framework_name

    _meta = importlib.metadata

    _FRAMEWORK_PACKAGE: dict[str, str] = {
        "pytorch": "torch",
        "huggingface": "transformers",
        "sklearn": "scikit-learn",
        "tensorflow": "tensorflow",
        "pickle": "pickle",
    }
    package_name = _FRAMEWORK_PACKAGE.get(framework_name, framework_name)
    try:
        framework_version = _meta.version(package_name)
    except _meta.PackageNotFoundError:
        framework_version = "unknown"

    packages = {dist.metadata["Name"]: dist.metadata["Version"] for dist in _meta.distributions()}

    cuda_version: str | None = None
    try:
        import torch
        cuda_version = torch.version.cuda
    except ImportError:
        pass

    os_str = platform.system().lower() + "-" + platform.machine()
    captured_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    snapshot = DepSnapshot(
        python_version=platform.python_version(),
        framework=FrameworkInfo(name=framework_name, version=framework_version),
        packages=packages,
        os=os_str,
        captured_at=captured_at,
        cuda_version=cuda_version,
    )

    if override:
        _OVERRIDE_KEY_MAP = {
            "pythonVersion": "python_version",
            "framework": "framework",
            "packages": "packages",
            "cudaVersion": "cuda_version",
            "os": "os",
            "capturedAt": "captured_at",
        }
        for camel_key, value in override.items():
            snake_key = _OVERRIDE_KEY_MAP.get(camel_key, camel_key)
            setattr(snapshot, snake_key, value)

    return snapshot
