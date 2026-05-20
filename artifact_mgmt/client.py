from __future__ import annotations

import os

from artifact_mgmt._http import HttpClient
from artifact_mgmt._pagination import PageIterator
from artifact_mgmt._types import Model, DepSnapshot, FrameworkInfo, Version

_STAGE_ENDPOINTS: dict[str, str] = {
    "alpha": "https://pi5ywcu3ub.execute-api.us-east-1.amazonaws.com/alpha",
    "gamma": "https://idco76hrk9.execute-api.us-east-1.amazonaws.com/gamma",
}


def _parse_dep_snapshot(raw: dict) -> DepSnapshot:  # type: ignore[type-arg]
    fw = raw["framework"]
    return DepSnapshot(
        python_version=raw["pythonVersion"],
        framework=FrameworkInfo(name=fw["name"], version=fw["version"]),
        packages=raw.get("packages", {}),
        os=raw.get("os", ""),
        captured_at=raw.get("capturedAt", ""),
        cuda_version=raw.get("cudaVersion"),
    )


def _parse_model(raw: dict) -> Model:  # type: ignore[type-arg]
    return Model(
        model_name=raw["modelName"],
        owner=raw["owner"],
        status=raw["status"],
        created_at=raw["createdAt"],
        updated_at=raw["updatedAt"],
        framework_hint=raw.get("frameworkHint"),
        description=raw.get("description"),
        latest_major=raw.get("latestMajor", 0),
        latest_minor=raw.get("latestMinor", 0),
    )


def _parse_version(raw: dict) -> Version:  # type: ignore[type-arg]
    return Version(
        model_name=raw["modelName"],
        version=raw["version"],
        status=raw["status"],
        dep_snapshot=_parse_dep_snapshot(raw["depSnapshot"]),
        created_at=raw["createdAt"],
        upload_url=raw.get("uploadUrl"),
        upload_url_expires_at=raw.get("uploadUrlExpiresAt"),
        download_url=raw.get("downloadUrl"),
        checksum_sha256=raw.get("checksumSha256"),
    )


class ArtifactMgmtClient:
    """Python client for the Artifact Management Service."""

    def __init__(
        self,
        *,
        stage: str | None = None,
        endpoint_url: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        if endpoint_url is None:
            resolved_stage = stage or os.environ.get("ARTIFACT_MGMT_STAGE") or "gamma"
            endpoint_url = _STAGE_ENDPOINTS[resolved_stage]
        self._http = HttpClient(endpoint_url)
        self._cache_dir = cache_dir

    # ------------------------------------------------------------------
    # Model CRUD
    # ------------------------------------------------------------------

    def create_model(
        self,
        model_name: str,
        *,
        framework_hint: str | None = None,
        description: str | None = None,
    ) -> Model:
        body: dict[str, str] = {"modelName": model_name}
        if framework_hint is not None:
            body["frameworkHint"] = framework_hint
        if description is not None:
            body["description"] = description
        raw = self._http.request("POST", "/models", body=body)
        return _parse_model(raw)

    def get_model(self, model_name: str) -> Model:
        raw = self._http.request("GET", f"/models/{model_name}")
        return _parse_model(raw)

    def list_models(self) -> PageIterator[Model]:
        def fetch_page(token: str | None) -> tuple[list[Model], str | None]:
            params = {"pageToken": token} if token else None
            raw = self._http.request("GET", "/models", params=params)
            items = [_parse_model(m) for m in raw.get("items", [])]
            return items, raw.get("nextPageToken")

        return PageIterator(fetch_page)

    def delete_model(self, model_name: str) -> None:
        self._http.request("DELETE", f"/models/{model_name}")
