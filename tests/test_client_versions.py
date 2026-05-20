from __future__ import annotations

import base64
import hashlib
import json
from unittest.mock import MagicMock, patch

import pytest
import responses as responses_lib

from artifact_mgmt.client import ArtifactMgmtClient
from artifact_mgmt._exceptions import VersionNotFoundError, VersionConflictError
from artifact_mgmt._types import Version, DepSnapshot, FrameworkInfo

BASE_URL = "https://idco76hrk9.execute-api.us-east-1.amazonaws.com/gamma"

RAW_DEP_SNAPSHOT = {
    "pythonVersion": "3.11.0",
    "framework": {"name": "pytorch", "version": "2.1.0"},
    "packages": {"torch": "2.1.0"},
    "os": "linux-x86_64",
    "capturedAt": "2024-01-01T00:00:00Z",
    "cudaVersion": None,
}

RAW_VERSION = {
    "modelName": "fraud-detector",
    "version": "1.0",
    "status": "READY",
    "depSnapshot": RAW_DEP_SNAPSHOT,
    "createdAt": "2024-01-01T00:00:00Z",
    "downloadUrl": "https://s3.example.com/weights",
}

RAW_VERSION_2 = {**RAW_VERSION, "version": "1.1"}
RAW_VERSION_3 = {**RAW_VERSION, "version": "2.0"}


@pytest.fixture
def client() -> ArtifactMgmtClient:
    mock_creds = MagicMock()
    mock_creds.access_key = "AKIATEST"
    mock_creds.secret_key = "testsecret"
    mock_creds.token = None

    mock_session = MagicMock()
    mock_session.get_credentials.return_value.get_frozen_credentials.return_value = mock_creds

    with patch("artifact_mgmt._http.boto3.Session", return_value=mock_session):
        return ArtifactMgmtClient(stage="gamma")


class TestGetVersion:
    @responses_lib.activate
    def test_get_version_returns_version_with_correct_fields(
        self, client: ArtifactMgmtClient
    ) -> None:
        responses_lib.add(
            responses_lib.GET, BASE_URL + "/models/fraud-detector/versions/1.0",
            json=RAW_VERSION,
            status=200,
        )
        version = client.get_version("fraud-detector", "1.0")
        assert isinstance(version, Version)
        assert version.model_name == "fraud-detector"
        assert version.version == "1.0"
        assert version.status == "READY"

    @responses_lib.activate
    def test_get_version_uses_dotted_string_version_in_path(
        self, client: ArtifactMgmtClient
    ) -> None:
        responses_lib.add(
            responses_lib.GET, BASE_URL + "/models/fraud-detector/versions/2.1",
            json={**RAW_VERSION, "version": "2.1"},
            status=200,
        )
        version = client.get_version("fraud-detector", "2.1")
        assert version.version == "2.1"
        assert "/versions/2.1" in responses_lib.calls[0].request.url

    @responses_lib.activate
    def test_get_version_raises_version_not_found_error_when_version_does_not_exist(
        self, client: ArtifactMgmtClient
    ) -> None:
        responses_lib.add(
            responses_lib.GET, BASE_URL + "/models/fraud-detector/versions/9.9",
            json={"code": "VersionNotFound", "message": "not found"},
            status=404,
        )
        with pytest.raises(VersionNotFoundError):
            client.get_version("fraud-detector", "9.9")


class TestGetLatestVersion:
    @responses_lib.activate
    def test_get_latest_version_hits_latest_endpoint(
        self, client: ArtifactMgmtClient
    ) -> None:
        responses_lib.add(
            responses_lib.GET, BASE_URL + "/models/fraud-detector/versions/latest",
            json=RAW_VERSION,
            status=200,
        )
        version = client.get_latest_version("fraud-detector")
        assert version.version == "1.0"
        assert responses_lib.calls[0].request.url.endswith("/versions/latest")

    @responses_lib.activate
    def test_get_latest_version_raises_version_not_found_when_no_ready_version_exists(
        self, client: ArtifactMgmtClient
    ) -> None:
        responses_lib.add(
            responses_lib.GET, BASE_URL + "/models/fraud-detector/versions/latest",
            json={"code": "VersionNotFound", "message": "no ready version"},
            status=404,
        )
        with pytest.raises(VersionNotFoundError):
            client.get_latest_version("fraud-detector")


class TestListVersions:
    @responses_lib.activate
    def test_list_versions_returns_lazy_iterator(self, client: ArtifactMgmtClient) -> None:
        responses_lib.add(
            responses_lib.GET, BASE_URL + "/models/fraud-detector/versions",
            json={"versions": [RAW_VERSION], "nextPageToken": None},
            status=200,
        )
        iterator = client.list_versions("fraud-detector")
        assert len(responses_lib.calls) == 0
        result = list(iterator)
        assert len(result) == 1
        assert result[0].version == "1.0"

    @responses_lib.activate
    def test_list_versions_uses_versions_response_key(
        self, client: ArtifactMgmtClient
    ) -> None:
        responses_lib.add(
            responses_lib.GET, BASE_URL + "/models/fraud-detector/versions",
            json={"versions": [RAW_VERSION, RAW_VERSION_2], "nextPageToken": None},
            status=200,
        )
        result = list(client.list_versions("fraud-detector"))
        assert [v.version for v in result] == ["1.0", "1.1"]

    @responses_lib.activate
    def test_list_versions_paginates_across_three_pages(
        self, client: ArtifactMgmtClient
    ) -> None:
        responses_lib.add(
            responses_lib.GET, BASE_URL + "/models/fraud-detector/versions",
            json={"versions": [RAW_VERSION], "nextPageToken": "tok1"},
            status=200,
        )
        responses_lib.add(
            responses_lib.GET, BASE_URL + "/models/fraud-detector/versions",
            json={"versions": [RAW_VERSION_2], "nextPageToken": "tok2"},
            status=200,
        )
        responses_lib.add(
            responses_lib.GET, BASE_URL + "/models/fraud-detector/versions",
            json={"versions": [RAW_VERSION_3], "nextPageToken": None},
            status=200,
        )
        result = list(client.list_versions("fraud-detector"))
        assert [v.version for v in result] == ["1.0", "1.1", "2.0"]
        assert len(responses_lib.calls) == 3

    @responses_lib.activate
    def test_list_versions_passes_include_pending_param_when_true(
        self, client: ArtifactMgmtClient
    ) -> None:
        responses_lib.add(
            responses_lib.GET, BASE_URL + "/models/fraud-detector/versions",
            json={"versions": [], "nextPageToken": None},
            status=200,
        )
        list(client.list_versions("fraud-detector", include_pending=True))
        assert "includePending=true" in responses_lib.calls[0].request.url

    @responses_lib.activate
    def test_list_versions_omits_include_pending_param_when_false(
        self, client: ArtifactMgmtClient
    ) -> None:
        responses_lib.add(
            responses_lib.GET, BASE_URL + "/models/fraud-detector/versions",
            json={"versions": [], "nextPageToken": None},
            status=200,
        )
        list(client.list_versions("fraud-detector", include_pending=False))
        assert "includePending" not in responses_lib.calls[0].request.url


class TestDeleteVersion:
    @responses_lib.activate
    def test_delete_version_returns_none_on_204(self, client: ArtifactMgmtClient) -> None:
        responses_lib.add(
            responses_lib.DELETE, BASE_URL + "/models/fraud-detector/versions/1.0",
            status=204,
        )
        result = client.delete_version("fraud-detector", "1.0")
        assert result is None

    @responses_lib.activate
    def test_delete_version_raises_version_not_found_error_when_version_does_not_exist(
        self, client: ArtifactMgmtClient
    ) -> None:
        responses_lib.add(
            responses_lib.DELETE, BASE_URL + "/models/fraud-detector/versions/9.9",
            json={"code": "VersionNotFound", "message": "not found"},
            status=404,
        )
        with pytest.raises(VersionNotFoundError):
            client.delete_version("fraud-detector", "9.9")


# ---------------------------------------------------------------------------
# Story 2.3 — _create_version + _upload_artifact
# ---------------------------------------------------------------------------

DEP_SNAPSHOT = DepSnapshot(
    python_version="3.11.0",
    framework=FrameworkInfo(name="pytorch", version="2.1.0"),
    packages={"torch": "2.1.0"},
    os="linux-x86_64",
    captured_at="2024-01-01T00:00:00Z",
)

RAW_PENDING_VERSION = {
    "modelName": "fraud-detector",
    "version": "1.0",
    "status": "PENDING",
    "depSnapshot": RAW_DEP_SNAPSHOT,
    "createdAt": "2024-01-01T00:00:00Z",
    "uploadUrl": "https://s3.example.com/upload",
    "uploadUrlExpiresAt": "2024-01-01T01:00:00Z",
}


class TestCreateVersion:
    @responses_lib.activate
    def test_create_version_returns_pending_version_with_upload_url(
        self, client: ArtifactMgmtClient
    ) -> None:
        responses_lib.add(
            responses_lib.POST, BASE_URL + "/models/fraud-detector/versions",
            json=RAW_PENDING_VERSION,
            status=201,
        )
        version = client._create_version(
            "fraud-detector",
            idempotency_key="test-uuid",
            dep_snapshot=DEP_SNAPSHOT,
        )
        assert isinstance(version, Version)
        assert version.status == "PENDING"
        assert version.upload_url == "https://s3.example.com/upload"

    @responses_lib.activate
    def test_create_version_sends_idempotency_key_and_dep_snapshot(
        self, client: ArtifactMgmtClient
    ) -> None:
        responses_lib.add(
            responses_lib.POST, BASE_URL + "/models/fraud-detector/versions",
            json=RAW_PENDING_VERSION,
            status=201,
        )
        client._create_version(
            "fraud-detector",
            idempotency_key="my-key",
            dep_snapshot=DEP_SNAPSHOT,
        )
        body = json.loads(responses_lib.calls[0].request.body)
        assert body["idempotencyKey"] == "my-key"
        assert "depSnapshot" in body

    @responses_lib.activate
    def test_create_version_sends_major_when_provided(
        self, client: ArtifactMgmtClient
    ) -> None:
        responses_lib.add(
            responses_lib.POST, BASE_URL + "/models/fraud-detector/versions",
            json={**RAW_PENDING_VERSION, "version": "2.0"},
            status=201,
        )
        client._create_version(
            "fraud-detector",
            idempotency_key="my-key",
            dep_snapshot=DEP_SNAPSHOT,
            major=2,
        )
        body = json.loads(responses_lib.calls[0].request.body)
        assert body["major"] == 2

    @responses_lib.activate
    def test_create_version_omits_major_when_not_provided(
        self, client: ArtifactMgmtClient
    ) -> None:
        responses_lib.add(
            responses_lib.POST, BASE_URL + "/models/fraud-detector/versions",
            json=RAW_PENDING_VERSION,
            status=201,
        )
        client._create_version(
            "fraud-detector",
            idempotency_key="my-key",
            dep_snapshot=DEP_SNAPSHOT,
        )
        body = json.loads(responses_lib.calls[0].request.body)
        assert "major" not in body

    @responses_lib.activate
    def test_create_version_sends_checksum_when_provided(
        self, client: ArtifactMgmtClient
    ) -> None:
        responses_lib.add(
            responses_lib.POST, BASE_URL + "/models/fraud-detector/versions",
            json=RAW_PENDING_VERSION,
            status=201,
        )
        checksum = base64.b64encode(hashlib.sha256(b"data").digest()).decode()
        client._create_version(
            "fraud-detector",
            idempotency_key="my-key",
            dep_snapshot=DEP_SNAPSHOT,
            checksum_sha256=checksum,
        )
        body = json.loads(responses_lib.calls[0].request.body)
        assert body["checksumSha256"] == checksum

    @responses_lib.activate
    def test_create_version_checksum_is_base64_sha256_not_hex(
        self, client: ArtifactMgmtClient
    ) -> None:
        data = b"model weights"
        expected = base64.b64encode(hashlib.sha256(data).digest()).decode()
        wrong_hex = hashlib.sha256(data).hexdigest()

        responses_lib.add(
            responses_lib.POST, BASE_URL + "/models/fraud-detector/versions",
            json=RAW_PENDING_VERSION,
            status=201,
        )
        client._create_version(
            "fraud-detector",
            idempotency_key="my-key",
            dep_snapshot=DEP_SNAPSHOT,
            checksum_sha256=expected,
        )
        body = json.loads(responses_lib.calls[0].request.body)
        assert body["checksumSha256"] == expected
        assert body["checksumSha256"] != wrong_hex

    @responses_lib.activate
    def test_create_version_idempotency_replay_on_200_handled_identically_to_201(
        self, client: ArtifactMgmtClient
    ) -> None:
        responses_lib.add(
            responses_lib.POST, BASE_URL + "/models/fraud-detector/versions",
            json=RAW_PENDING_VERSION,
            status=200,
        )
        version = client._create_version(
            "fraud-detector",
            idempotency_key="my-key",
            dep_snapshot=DEP_SNAPSHOT,
        )
        assert version.upload_url == "https://s3.example.com/upload"

    @responses_lib.activate
    def test_create_version_includes_cuda_version_when_present(
        self, client: ArtifactMgmtClient
    ) -> None:
        snap_with_cuda = DepSnapshot(
            python_version="3.11.0",
            framework=FrameworkInfo(name="pytorch", version="2.1.0"),
            packages={},
            os="linux-x86_64",
            captured_at="2024-01-01T00:00:00Z",
            cuda_version="12.1",
        )
        responses_lib.add(
            responses_lib.POST, BASE_URL + "/models/fraud-detector/versions",
            json=RAW_PENDING_VERSION,
            status=201,
        )
        client._create_version(
            "fraud-detector",
            idempotency_key="my-key",
            dep_snapshot=snap_with_cuda,
        )
        body = json.loads(responses_lib.calls[0].request.body)
        assert body["depSnapshot"]["cudaVersion"] == "12.1"

    @responses_lib.activate
    def test_create_version_raises_version_conflict_error_on_409(
        self, client: ArtifactMgmtClient
    ) -> None:
        responses_lib.add(
            responses_lib.POST, BASE_URL + "/models/fraud-detector/versions",
            json={"code": "VersionConflict", "message": "conflict"},
            status=409,
        )
        with pytest.raises(VersionConflictError):
            client._create_version(
                "fraud-detector",
                idempotency_key="my-key",
                dep_snapshot=DEP_SNAPSHOT,
            )


class TestUploadArtifact:
    @responses_lib.activate
    def test_upload_artifact_sends_bytes_to_upload_url(
        self, client: ArtifactMgmtClient
    ) -> None:
        responses_lib.add(
            responses_lib.PUT, "https://s3.example.com/upload",
            status=200,
        )
        client._upload_artifact("https://s3.example.com/upload", b"model bytes")
        assert responses_lib.calls[0].request.body == b"model bytes"

    @responses_lib.activate
    def test_upload_artifact_sets_content_type_header(
        self, client: ArtifactMgmtClient
    ) -> None:
        responses_lib.add(
            responses_lib.PUT, "https://s3.example.com/upload",
            status=200,
        )
        client._upload_artifact("https://s3.example.com/upload", b"data")
        assert (
            responses_lib.calls[0].request.headers["Content-Type"]
            == "application/octet-stream"
        )
