from __future__ import annotations

import base64
import hashlib
import os
import uuid
from typing import TYPE_CHECKING, Any

import requests as _requests

from artifact_mgmt._artifact_model import ArtifactModel
from artifact_mgmt._http import HttpClient
from artifact_mgmt._pagination import PageIterator
from artifact_mgmt._snapshot import capture as _capture_snapshot
from artifact_mgmt._types import Model, DepSnapshot, FrameworkInfo, Version
from artifact_mgmt.serializers import SerializerRegistry

if TYPE_CHECKING:
    from artifact_mgmt._cache import ModelCache

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
        self._cache: ModelCache | None = None

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

    # ------------------------------------------------------------------
    # Version read CRUD
    # ------------------------------------------------------------------

    def get_version(self, model_name: str, version: str) -> Version:
        raw = self._http.request("GET", f"/models/{model_name}/versions/{version}")
        return _parse_version(raw)

    def get_latest_version(self, model_name: str) -> Version:
        raw = self._http.request("GET", f"/models/{model_name}/versions/latest")
        return _parse_version(raw)

    def list_versions(
        self, model_name: str, *, include_pending: bool = False
    ) -> PageIterator[Version]:
        def fetch_page(token: str | None) -> tuple[list[Version], str | None]:
            params: dict[str, str] = {}
            if token:
                params["pageToken"] = token
            if include_pending:
                params["includePending"] = "true"
            raw = self._http.request(
                "GET", f"/models/{model_name}/versions", params=params or None
            )
            items = [_parse_version(v) for v in raw.get("versions", [])]
            return items, raw.get("nextPageToken")

        return PageIterator(fetch_page)

    def delete_version(self, model_name: str, version: str) -> None:
        self._http.request("DELETE", f"/models/{model_name}/versions/{version}")

    # ------------------------------------------------------------------
    # Version create + upload (internal — called by save_model)
    # ------------------------------------------------------------------

    def _create_version(
        self,
        model_name: str,
        *,
        idempotency_key: str,
        dep_snapshot: DepSnapshot,
        major: int | None = None,
        checksum_sha256: str | None = None,
    ) -> Version:
        body: dict[str, object] = {
            "idempotencyKey": idempotency_key,
            "depSnapshot": {
                "pythonVersion": dep_snapshot.python_version,
                "framework": {
                    "name": dep_snapshot.framework.name,
                    "version": dep_snapshot.framework.version,
                },
                "packages": dep_snapshot.packages,
                "os": dep_snapshot.os,
                "capturedAt": dep_snapshot.captured_at,
                **({"cudaVersion": dep_snapshot.cuda_version} if dep_snapshot.cuda_version else {}),
            },
        }
        if major is not None:
            body["major"] = major
        if checksum_sha256 is not None:
            body["checksumSha256"] = checksum_sha256

        raw = self._http.request(
            "POST", f"/models/{model_name}/versions", body=body
        )
        return _parse_version(raw)

    def _upload_artifact(
        self,
        upload_url: str,
        data: bytes,
        checksum_sha256: str | None = None,
    ) -> None:
        _ = checksum_sha256  # checksum already sent to service at create_version time
        self._http.upload(upload_url, data)

    # ------------------------------------------------------------------
    # ConfirmVersion (internal — called by save_model)
    # ------------------------------------------------------------------

    def _confirm_version(self, model_name: str, version: str) -> Version:
        raw = self._http.request(
            "PUT", f"/models/{model_name}/versions/{version}/confirm", body={}
        )
        return _parse_version(raw)

    # ------------------------------------------------------------------
    # High-level save_model / load_model
    # ------------------------------------------------------------------

    def save_model(
        self,
        model: object,
        model_name: str,
        *,
        major: int | None = None,
        dep_snapshot: dict[str, Any] | None = None,
        serializer: str | None = None,
    ) -> str:
        """Serialize, upload, and confirm a new model version. Returns version string e.g. '2.1'."""
        if serializer is not None:
            ser = SerializerRegistry.get_by_name(serializer)
        else:
            ser = SerializerRegistry.detect(model)

        data = ser.serialize(model)
        checksum_sha256 = base64.b64encode(hashlib.sha256(data).digest()).decode()
        snapshot = _capture_snapshot(model, override=dep_snapshot)
        idempotency_key = str(uuid.uuid4())

        version_obj = self._create_version(
            model_name,
            idempotency_key=idempotency_key,
            dep_snapshot=snapshot,
            major=major,
            checksum_sha256=checksum_sha256,
        )
        self._upload_artifact(version_obj.upload_url or "", data, checksum_sha256)
        confirmed = self._confirm_version(model_name, version_obj.version)
        return confirmed.version

    def load_model(
        self,
        model_name: str,
        *,
        version: str | None = None,
    ) -> ArtifactModel:
        """Download and deserialize a model version. Returns ArtifactModel wrapper."""
        if version is not None:
            version_obj = self.get_version(model_name, version)
        else:
            version_obj = self.get_latest_version(model_name)

        if self._cache is not None:
            cached = self._cache.get(model_name, version_obj.version)
            if cached is not None:
                return cached  # type: ignore[no-any-return]

        response = _requests.get(version_obj.download_url or "")
        response.raise_for_status()
        data = response.content

        serializer = SerializerRegistry.detect_from_snapshot(version_obj.dep_snapshot)
        model: object = serializer.deserialize(data)

        artifact = ArtifactModel(
            model,
            model_name=model_name,
            version=version_obj.version,
            dep_snapshot=version_obj.dep_snapshot,
            serializer=serializer,
        )

        if self._cache is not None:
            self._cache.put(model_name, version_obj.version, artifact)

        return artifact
