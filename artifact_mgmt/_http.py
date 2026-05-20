from __future__ import annotations

from typing import Any

import requests
import boto3
from requests_aws4auth import AWS4Auth  # type: ignore[import-untyped,unused-ignore]

from artifact_mgmt._exceptions import (
    AuthError,
    ModelNotFoundError,
    VersionNotFoundError,
    UploadNotFoundError,
    ModelAlreadyExistsError,
    VersionConflictError,
    IdempotencyMismatchError,
    ChecksumMismatchError,
    ServiceError,
)

_404_CODE_MAP: dict[str, type[Exception]] = {
    "ModelNotFound": ModelNotFoundError,
    "VersionNotFound": VersionNotFoundError,
    "UploadNotFound": UploadNotFoundError,
}

_409_CODE_MAP: dict[str, type[Exception]] = {
    "ModelAlreadyExists": ModelAlreadyExistsError,
    "VersionConflict": VersionConflictError,
    "IdempotencyMismatch": IdempotencyMismatchError,
    "ChecksumMismatch": ChecksumMismatchError,
}


def _raise_for_response(response: requests.Response) -> None:
    status = response.status_code
    if status < 400:
        return

    if status == 403:
        raise AuthError(response.text)

    try:
        body: dict[str, Any] = response.json()
    except Exception:
        body = {}

    code: str = body.get("code", "")

    if status == 404:
        exc_class = _404_CODE_MAP.get(code)
        if exc_class:
            raise exc_class(body.get("message", response.text))
        raise ServiceError(f"404: {response.text}")

    if status == 409:
        exc_class = _409_CODE_MAP.get(code)
        if exc_class:
            raise exc_class(body.get("message", response.text))
        raise ServiceError(f"409: {response.text}")

    if status >= 500:
        raise ServiceError(f"{status}: {response.text}")

    response.raise_for_status()


class HttpClient:
    def __init__(self, endpoint_url: str, region: str = "us-east-1") -> None:
        self._endpoint_url = endpoint_url.rstrip("/")
        session = boto3.Session()
        raw_creds = session.get_credentials()
        if raw_creds is None:
            raise AuthError("No AWS credentials found")
        # Freeze the credentials to get concrete access_key/secret_key/token values.
        frozen = raw_creds.get_frozen_credentials()
        self._auth = AWS4Auth(
            frozen.access_key,
            frozen.secret_key,
            region,
            "execute-api",
            session_token=frozen.token,
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = self._endpoint_url + path
        response = requests.request(
            method,
            url,
            json=body,
            params=params,
            auth=self._auth,
        )
        _raise_for_response(response)
        if response.status_code == 204 or not response.content:
            return {}
        return response.json()  # type: ignore[no-any-return]

    def upload(self, upload_url: str, data: bytes) -> None:
        response = requests.put(
            upload_url,
            data=data,
            headers={"Content-Type": "application/octet-stream"},
        )
        _raise_for_response(response)
