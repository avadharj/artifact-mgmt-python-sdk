from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import responses as responses_lib

from artifact_mgmt._artifact_model import ArtifactModel
from artifact_mgmt._exceptions import VersionNotFoundError
from artifact_mgmt.client import ArtifactMgmtClient

BASE_URL = "https://idco76hrk9.execute-api.us-east-1.amazonaws.com/gamma"
DOWNLOAD_URL = "https://s3.example.com/weights-presigned"

RAW_DEP_SNAPSHOT = {
    "pythonVersion": "3.11.0",
    "framework": {"name": "pytorch", "version": "2.1.0"},
    "packages": {"torch": "2.1.0"},
    "os": "linux-x86_64",
    "capturedAt": "2024-01-01T00:00:00Z",
}

RAW_VERSION = {
    "modelName": "fraud-detector",
    "version": "1.0",
    "status": "READY",
    "depSnapshot": RAW_DEP_SNAPSHOT,
    "createdAt": "2024-01-01T00:00:00Z",
    "downloadUrl": DOWNLOAD_URL,
}

RAW_VERSION_LATEST = {**RAW_VERSION, "version": "1.3"}


@pytest.fixture()
def client() -> ArtifactMgmtClient:
    mock_creds = MagicMock()
    mock_creds.access_key = "AKIATEST"
    mock_creds.secret_key = "testsecret"
    mock_creds.token = None
    mock_session = MagicMock()
    mock_session.get_credentials.return_value.get_frozen_credentials.return_value = mock_creds
    with patch("artifact_mgmt._http.boto3.Session", return_value=mock_session):
        return ArtifactMgmtClient(stage="gamma")


@pytest.fixture()
def mock_serializer() -> MagicMock:
    ser = MagicMock()
    ser.framework_name = "pytorch"
    ser.deserialize.return_value = MagicMock(name="native_model")
    return ser


# ---------------------------------------------------------------------------
# Story 6.2 — load_model
# ---------------------------------------------------------------------------


class TestLoadModelHappyPath:
    @responses_lib.activate
    def test_returns_artifact_model_with_correct_model_name(
        self, client: ArtifactMgmtClient, mock_serializer: MagicMock
    ) -> None:
        responses_lib.add(
            responses_lib.GET,
            BASE_URL + "/models/fraud-detector/versions/1.0",
            json=RAW_VERSION, status=200,
        )
        responses_lib.add(responses_lib.GET, DOWNLOAD_URL, body=b"fake-weights", status=200)

        with patch("artifact_mgmt.client.SerializerRegistry.detect_from_snapshot", return_value=mock_serializer):
            artifact = client.load_model("fraud-detector", version="1.0")

        assert isinstance(artifact, ArtifactModel)
        assert artifact.model_name == "fraud-detector"

    @responses_lib.activate
    def test_artifact_version_matches_requested_version(
        self, client: ArtifactMgmtClient, mock_serializer: MagicMock
    ) -> None:
        responses_lib.add(
            responses_lib.GET,
            BASE_URL + "/models/fraud-detector/versions/1.0",
            json=RAW_VERSION, status=200,
        )
        responses_lib.add(responses_lib.GET, DOWNLOAD_URL, body=b"fake-weights", status=200)

        with patch("artifact_mgmt.client.SerializerRegistry.detect_from_snapshot", return_value=mock_serializer):
            artifact = client.load_model("fraud-detector", version="1.0")

        assert artifact.version == "1.0"

    @responses_lib.activate
    def test_artifact_dep_snapshot_populated(
        self, client: ArtifactMgmtClient, mock_serializer: MagicMock
    ) -> None:
        responses_lib.add(
            responses_lib.GET,
            BASE_URL + "/models/fraud-detector/versions/1.0",
            json=RAW_VERSION, status=200,
        )
        responses_lib.add(responses_lib.GET, DOWNLOAD_URL, body=b"fake-weights", status=200)

        with patch("artifact_mgmt.client.SerializerRegistry.detect_from_snapshot", return_value=mock_serializer):
            artifact = client.load_model("fraud-detector", version="1.0")

        assert artifact.dep_snapshot.framework.name == "pytorch"


class TestLoadModelLatestVersion:
    @responses_lib.activate
    def test_uses_get_latest_version_when_version_is_none(
        self, client: ArtifactMgmtClient, mock_serializer: MagicMock
    ) -> None:
        responses_lib.add(
            responses_lib.GET,
            BASE_URL + "/models/fraud-detector/versions/latest",
            json=RAW_VERSION_LATEST, status=200,
        )
        responses_lib.add(responses_lib.GET, DOWNLOAD_URL, body=b"fake-weights", status=200)

        with patch("artifact_mgmt.client.SerializerRegistry.detect_from_snapshot", return_value=mock_serializer):
            artifact = client.load_model("fraud-detector")

        assert artifact.version == "1.3"
        assert responses_lib.calls[0].request.url.endswith("/versions/latest")


class TestLoadModelDownload:
    @responses_lib.activate
    def test_downloads_from_presigned_url_without_sigv4(
        self, client: ArtifactMgmtClient, mock_serializer: MagicMock
    ) -> None:
        responses_lib.add(
            responses_lib.GET,
            BASE_URL + "/models/fraud-detector/versions/1.0",
            json=RAW_VERSION, status=200,
        )
        responses_lib.add(responses_lib.GET, DOWNLOAD_URL, body=b"weights-data", status=200)

        with patch("artifact_mgmt.client.SerializerRegistry.detect_from_snapshot", return_value=mock_serializer):
            client.load_model("fraud-detector", version="1.0")

        # The S3 presigned request should NOT have an Authorization header (no SigV4)
        s3_request = responses_lib.calls[1].request
        assert s3_request.url == DOWNLOAD_URL
        assert "Authorization" not in s3_request.headers

    @responses_lib.activate
    def test_passes_downloaded_bytes_to_deserializer(
        self, client: ArtifactMgmtClient, mock_serializer: MagicMock
    ) -> None:
        responses_lib.add(
            responses_lib.GET,
            BASE_URL + "/models/fraud-detector/versions/1.0",
            json=RAW_VERSION, status=200,
        )
        responses_lib.add(responses_lib.GET, DOWNLOAD_URL, body=b"real-weights", status=200)

        with patch("artifact_mgmt.client.SerializerRegistry.detect_from_snapshot", return_value=mock_serializer):
            client.load_model("fraud-detector", version="1.0")

        mock_serializer.deserialize.assert_called_once_with(b"real-weights")


class TestLoadModelCache:
    @responses_lib.activate
    def test_cache_miss_downloads_and_caches(
        self, client: ArtifactMgmtClient, mock_serializer: MagicMock
    ) -> None:
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        client._cache = mock_cache

        responses_lib.add(
            responses_lib.GET,
            BASE_URL + "/models/fraud-detector/versions/1.0",
            json=RAW_VERSION, status=200,
        )
        responses_lib.add(responses_lib.GET, DOWNLOAD_URL, body=b"fake-weights", status=200)

        with patch("artifact_mgmt.client.SerializerRegistry.detect_from_snapshot", return_value=mock_serializer):
            artifact = client.load_model("fraud-detector", version="1.0")

        mock_cache.get.assert_called_once_with("fraud-detector", "1.0")
        mock_cache.put.assert_called_once_with("fraud-detector", "1.0", artifact)
        assert len(responses_lib.calls) == 2  # version fetch + S3 download

    @responses_lib.activate
    def test_cache_hit_returns_cached_artifact_without_http_call(
        self, client: ArtifactMgmtClient, mock_serializer: MagicMock
    ) -> None:
        cached_artifact = MagicMock(spec=ArtifactModel)
        mock_cache = MagicMock()
        mock_cache.get.return_value = cached_artifact
        client._cache = mock_cache

        responses_lib.add(
            responses_lib.GET,
            BASE_URL + "/models/fraud-detector/versions/1.0",
            json=RAW_VERSION, status=200,
        )
        # No S3 download registered — if it's called the test would fail

        with patch("artifact_mgmt.client.SerializerRegistry.detect_from_snapshot", return_value=mock_serializer):
            artifact = client.load_model("fraud-detector", version="1.0")

        assert artifact is cached_artifact
        # Only 1 HTTP call (get_version), no S3 download
        assert len(responses_lib.calls) == 1


class TestLoadModelErrorPropagation:
    @responses_lib.activate
    def test_version_not_found_error_propagates(
        self, client: ArtifactMgmtClient
    ) -> None:
        responses_lib.add(
            responses_lib.GET,
            BASE_URL + "/models/fraud-detector/versions/9.9",
            json={"code": "VersionNotFound", "message": "not found"},
            status=404,
        )

        with pytest.raises(VersionNotFoundError):
            client.load_model("fraud-detector", version="9.9")
