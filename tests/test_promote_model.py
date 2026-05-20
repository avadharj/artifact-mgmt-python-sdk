from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import responses as responses_lib

from artifact_mgmt._exceptions import ModelNotFoundError, VersionNotFoundError
from artifact_mgmt.client import ArtifactMgmtClient

GAMMA_BASE = "https://idco76hrk9.execute-api.us-east-1.amazonaws.com/gamma"
PROD_BASE = "https://afwtpvnxe7.execute-api.us-east-1.amazonaws.com/prod"
DOWNLOAD_URL = "https://s3.example.com/gamma-weights"
UPLOAD_URL = "https://s3.example.com/prod-upload"

RAW_DEP_SNAPSHOT = {
    "pythonVersion": "3.11.0",
    "framework": {"name": "pytorch", "version": "2.1.0"},
    "packages": {"torch": "2.1.0"},
    "os": "linux-x86_64",
    "capturedAt": "2024-01-01T00:00:00Z",
}

RAW_GAMMA_VERSION = {
    "modelName": "fraud-detector",
    "version": "2.1",
    "status": "READY",
    "depSnapshot": RAW_DEP_SNAPSHOT,
    "createdAt": "2024-01-01T00:00:00Z",
    "downloadUrl": DOWNLOAD_URL,
}

RAW_GAMMA_LATEST = {**RAW_GAMMA_VERSION, "version": "3.0"}

RAW_PROD_VERSION_PENDING = {
    "modelName": "fraud-detector",
    "version": "1.0",
    "status": "PENDING",
    "depSnapshot": RAW_DEP_SNAPSHOT,
    "createdAt": "2024-01-01T00:00:00Z",
    "uploadUrl": UPLOAD_URL,
}

RAW_PROD_VERSION_CONFIRMED = {
    **RAW_PROD_VERSION_PENDING,
    "status": "READY",
    "uploadUrl": None,
    "downloadUrl": "https://s3.example.com/prod-weights",
}


def _make_client(stage: str) -> ArtifactMgmtClient:
    mock_creds = MagicMock()
    mock_creds.access_key = "AKIATEST"
    mock_creds.secret_key = "testsecret"
    mock_creds.token = None
    mock_session = MagicMock()
    mock_session.get_credentials.return_value.get_frozen_credentials.return_value = mock_creds
    with patch("artifact_mgmt._http.boto3.Session", return_value=mock_session):
        return ArtifactMgmtClient(stage=stage)


# ---------------------------------------------------------------------------
# Story 7.3 — promote_model
# ---------------------------------------------------------------------------


class TestPromoteModelHappyPath:
    @responses_lib.activate
    def test_returns_new_version_string_in_dest_stage(self) -> None:
        gamma = _make_client("gamma")
        prod = _make_client("prod")

        responses_lib.add(
            responses_lib.GET,
            GAMMA_BASE + "/models/fraud-detector/versions/2.1",
            json=RAW_GAMMA_VERSION, status=200,
        )
        responses_lib.add(responses_lib.GET, DOWNLOAD_URL, body=b"serialized-weights", status=200)
        responses_lib.add(
            responses_lib.POST,
            PROD_BASE + "/models/fraud-detector/versions",
            json=RAW_PROD_VERSION_PENDING, status=201,
        )
        responses_lib.add(responses_lib.PUT, UPLOAD_URL, body=b"", status=200)
        responses_lib.add(
            responses_lib.PUT,
            PROD_BASE + "/models/fraud-detector/versions/1.0/confirm",
            json=RAW_PROD_VERSION_CONFIRMED, status=200,
        )

        result = gamma.promote_model("fraud-detector", version="2.1", dest=prod)
        assert result == "1.0"

    @responses_lib.activate
    def test_bytes_uploaded_to_dest_unchanged(self) -> None:
        gamma = _make_client("gamma")
        prod = _make_client("prod")

        responses_lib.add(
            responses_lib.GET,
            GAMMA_BASE + "/models/fraud-detector/versions/2.1",
            json=RAW_GAMMA_VERSION, status=200,
        )
        responses_lib.add(responses_lib.GET, DOWNLOAD_URL, body=b"exact-bytes", status=200)
        responses_lib.add(
            responses_lib.POST,
            PROD_BASE + "/models/fraud-detector/versions",
            json=RAW_PROD_VERSION_PENDING, status=201,
        )
        responses_lib.add(responses_lib.PUT, UPLOAD_URL, body=b"", status=200)
        responses_lib.add(
            responses_lib.PUT,
            PROD_BASE + "/models/fraud-detector/versions/1.0/confirm",
            json=RAW_PROD_VERSION_CONFIRMED, status=200,
        )

        gamma.promote_model("fraud-detector", version="2.1", dest=prod)

        # S3 upload should contain the exact bytes downloaded from source
        upload_body = responses_lib.calls[3].request.body
        assert upload_body == b"exact-bytes"

    @responses_lib.activate
    def test_dep_snapshot_carried_over_from_source(self) -> None:
        gamma = _make_client("gamma")
        prod = _make_client("prod")

        responses_lib.add(
            responses_lib.GET,
            GAMMA_BASE + "/models/fraud-detector/versions/2.1",
            json=RAW_GAMMA_VERSION, status=200,
        )
        responses_lib.add(responses_lib.GET, DOWNLOAD_URL, body=b"weights", status=200)
        responses_lib.add(
            responses_lib.POST,
            PROD_BASE + "/models/fraud-detector/versions",
            json=RAW_PROD_VERSION_PENDING, status=201,
        )
        responses_lib.add(responses_lib.PUT, UPLOAD_URL, body=b"", status=200)
        responses_lib.add(
            responses_lib.PUT,
            PROD_BASE + "/models/fraud-detector/versions/1.0/confirm",
            json=RAW_PROD_VERSION_CONFIRMED, status=200,
        )

        gamma.promote_model("fraud-detector", version="2.1", dest=prod)

        create_body = json.loads(responses_lib.calls[2].request.body)
        assert create_body["depSnapshot"]["framework"]["name"] == "pytorch"
        assert create_body["depSnapshot"]["pythonVersion"] == "3.11.0"


class TestPromoteModelLatestVersion:
    @responses_lib.activate
    def test_uses_get_latest_version_when_version_is_none(self) -> None:
        gamma = _make_client("gamma")
        prod = _make_client("prod")

        responses_lib.add(
            responses_lib.GET,
            GAMMA_BASE + "/models/fraud-detector/versions/latest",
            json=RAW_GAMMA_LATEST, status=200,
        )
        responses_lib.add(responses_lib.GET, DOWNLOAD_URL, body=b"weights", status=200)
        responses_lib.add(
            responses_lib.POST,
            PROD_BASE + "/models/fraud-detector/versions",
            json=RAW_PROD_VERSION_PENDING, status=201,
        )
        responses_lib.add(responses_lib.PUT, UPLOAD_URL, body=b"", status=200)
        responses_lib.add(
            responses_lib.PUT,
            PROD_BASE + "/models/fraud-detector/versions/1.0/confirm",
            json=RAW_PROD_VERSION_CONFIRMED, status=200,
        )

        result = gamma.promote_model("fraud-detector", dest=prod)

        assert responses_lib.calls[0].request.url.endswith("/versions/latest")
        assert result == "1.0"


class TestPromoteModelMajorBump:
    @responses_lib.activate
    def test_major_passed_to_dest_create_version(self) -> None:
        gamma = _make_client("gamma")
        prod = _make_client("prod")

        raw_v2_pending = {**RAW_PROD_VERSION_PENDING, "version": "2.0"}
        raw_v2_confirmed = {**RAW_PROD_VERSION_CONFIRMED, "version": "2.0"}

        responses_lib.add(
            responses_lib.GET,
            GAMMA_BASE + "/models/fraud-detector/versions/2.1",
            json=RAW_GAMMA_VERSION, status=200,
        )
        responses_lib.add(responses_lib.GET, DOWNLOAD_URL, body=b"weights", status=200)
        responses_lib.add(
            responses_lib.POST,
            PROD_BASE + "/models/fraud-detector/versions",
            json=raw_v2_pending, status=201,
        )
        responses_lib.add(responses_lib.PUT, UPLOAD_URL, body=b"", status=200)
        responses_lib.add(
            responses_lib.PUT,
            PROD_BASE + "/models/fraud-detector/versions/2.0/confirm",
            json=raw_v2_confirmed, status=200,
        )

        result = gamma.promote_model("fraud-detector", version="2.1", dest=prod, major=2)

        create_body = json.loads(responses_lib.calls[2].request.body)
        assert create_body["major"] == 2
        assert result == "2.0"


class TestPromoteModelErrorPropagation:
    @responses_lib.activate
    def test_version_not_found_in_source_raises(self) -> None:
        gamma = _make_client("gamma")
        prod = _make_client("prod")

        responses_lib.add(
            responses_lib.GET,
            GAMMA_BASE + "/models/fraud-detector/versions/9.9",
            json={"code": "VersionNotFound", "message": "not found"},
            status=404,
        )

        with pytest.raises(VersionNotFoundError):
            gamma.promote_model("fraud-detector", version="9.9", dest=prod)

    @responses_lib.activate
    def test_model_not_found_in_dest_raises(self) -> None:
        gamma = _make_client("gamma")
        prod = _make_client("prod")

        responses_lib.add(
            responses_lib.GET,
            GAMMA_BASE + "/models/fraud-detector/versions/2.1",
            json=RAW_GAMMA_VERSION, status=200,
        )
        responses_lib.add(responses_lib.GET, DOWNLOAD_URL, body=b"weights", status=200)
        responses_lib.add(
            responses_lib.POST,
            PROD_BASE + "/models/fraud-detector/versions",
            json={"code": "ModelNotFound", "message": "model does not exist in dest"},
            status=404,
        )

        with pytest.raises(ModelNotFoundError):
            gamma.promote_model("fraud-detector", version="2.1", dest=prod)

    @responses_lib.activate
    def test_missing_download_url_raises_version_not_found(self) -> None:
        gamma = _make_client("gamma")
        prod = _make_client("prod")

        raw_no_url = {**RAW_GAMMA_VERSION, "downloadUrl": None}
        responses_lib.add(
            responses_lib.GET,
            GAMMA_BASE + "/models/fraud-detector/versions/2.1",
            json=raw_no_url, status=200,
        )

        with pytest.raises(VersionNotFoundError):
            gamma.promote_model("fraud-detector", version="2.1", dest=prod)
