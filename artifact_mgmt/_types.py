from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FrameworkInfo:
    name: str
    version: str


@dataclass
class DepSnapshot:
    python_version: str
    framework: FrameworkInfo
    packages: dict[str, str]
    os: str
    captured_at: str
    cuda_version: str | None = None


@dataclass
class Model:
    model_name: str
    owner: str
    status: str
    created_at: str
    updated_at: str
    framework_hint: str | None = None
    description: str | None = None
    latest_major: int = 0
    latest_minor: int = 0


@dataclass
class Version:
    model_name: str
    version: str
    status: str
    dep_snapshot: DepSnapshot
    created_at: str
    upload_url: str | None = None
    upload_url_expires_at: str | None = None
    download_url: str | None = None
    checksum_sha256: str | None = None
