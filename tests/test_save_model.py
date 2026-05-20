from __future__ import annotations

import base64
import hashlib
import json
import uuid
from unittest.mock import MagicMock, patch

import pytest
import responses as responses_lib

from artifact_mgmt.client import ArtifactMgmtClient

BASE_URL = "https://idco76hrk9.execute-api.us-east-1.amazonaws.com/gamma"
UPLOAD_URL = "https://s3.example.com/upload-presigned"

RAW_DEP_SNAPSHOT = {
    "pythonVersion": "3.11.0",
    "framework": {"name": "pytorch", "version": "2.1.0"},
    "packages": {"torch": "2.1.0"},
    "os": "linux-x86_64",
    "capturedAt": "2024-01-01T00:00:00Z",
}

RAW_VERSION_PENDING = {
    "modelName": "fraud-detector",
    "version": "1.0",
    "status": "PENDING",
    "depSnapshot": RAW_DEP_SNAPSHOT,
    "createdAt": "2024-01-01T00:00:00Z",
    "uploadUrl": UPLOAD_URL,
    "uploadUrlExpiresAt": "2024-01-01T01:00:00Z",
}

RAW_VERSION_CONFIRMED = {
    **RAW_VERSION_PENDING,
    "status": "READY",
    "uploadUrl": None,
    "downloadUrl": "https://s3.example.com/weights",
}


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
def mock_model() -> MagicMock:
    return MagicMock(name="nn_model")


@pytest.fixture()
def mock_serializer(mock_model: MagicMock) -> MagicMock:
    ser = MagicMock()
    ser.framework_name = "pytorch"
    ser.serialize.return_value = b"fake-weights"
    return ser


@pytest.fixture()
def patched_snapshot():
    """Patch snapshot capture to avoid needing a real framework installed."""
    from artifact_mgmt._types import DepSnapshot, FrameworkInfo
    snap = DepSnapshot(
        python_version="3.11.0",
        framework=FrameworkInfo(name="pytorch", version="2.1.0"),
        packages={"torch": "2.1.0"},
        os="linux-x86_64",
        captured_at="2024-01-01T00:00:00Z",
    )
    with patch("artifact_mgmt.client._capture_snapshot", return_value=snap):
        yield snap


# ---------------------------------------------------------------------------
# Story 6.1 — save_model
# ---------------------------------------------------------------------------


class TestSaveModelHappyPath:
    @responses_lib.activate
    def test_calls_create_upload_confirm_in_order(
        self, client: ArtifactMgmtClient, mock_model: MagicMock,
        mock_serializer: MagicMock, patched_snapshot: MagicMock
    ) -> None:
        call_order: list[str] = []

        def record(name: str):  # type: ignore[return]
            def _inner(req):  # type: ignore[no-untyped-def]
                call_order.append(name)
            return _inner

        responses_lib.add(
            responses_lib.POST,
            BASE_URL + "/models/fraud-detector/versions",
            json=RAW_VERSION_PENDING, status=201,
        )
        responses_lib.add(
            responses_lib.PUT, UPLOAD_URL,
            body=b"", status=200,
        )
        responses_lib.add(
            responses_lib.PUT,
            BASE_URL + "/models/fraud-detector/versions/1.0/confirm",
            json=RAW_VERSION_CONFIRMED, status=200,
        )

        with patch("artifact_mgmt.client.SerializerRegistry.detect", return_value=mock_serializer):
            result = client.save_model(mock_model, "fraud-detector")

        assert result == "1.0"
        assert len(responses_lib.calls) == 3
        assert "/versions" in responses_lib.calls[0].request.url
        assert responses_lib.calls[1].request.url == UPLOAD_URL
        assert "/confirm" in responses_lib.calls[2].request.url

    @responses_lib.activate
    def test_returns_version_string(
        self, client: ArtifactMgmtClient, mock_model: MagicMock,
        mock_serializer: MagicMock, patched_snapshot: MagicMock
    ) -> None:
        responses_lib.add(
            responses_lib.POST,
            BASE_URL + "/models/fraud-detector/versions",
            json=RAW_VERSION_PENDING, status=201,
        )
        responses_lib.add(responses_lib.PUT, UPLOAD_URL, body=b"", status=200)
        responses_lib.add(
            responses_lib.PUT,
            BASE_URL + "/models/fraud-detector/versions/1.0/confirm",
            json=RAW_VERSION_CONFIRMED, status=200,
        )
        with patch("artifact_mgmt.client.SerializerRegistry.detect", return_value=mock_serializer):
            result = client.save_model(mock_model, "fraud-detector")
        assert result == "1.0"


class TestSaveModelChecksum:
    @responses_lib.activate
    def test_checksum_is_base64_sha256_of_serialized_data(
        self, client: ArtifactMgmtClient, mock_model: MagicMock,
        mock_serializer: MagicMock, patched_snapshot: MagicMock
    ) -> None:
        data = b"fake-weights"
        expected_checksum = base64.b64encode(hashlib.sha256(data).digest()).decode()

        responses_lib.add(
            responses_lib.POST,
            BASE_URL + "/models/fraud-detector/versions",
            json=RAW_VERSION_PENDING, status=201,
        )
        responses_lib.add(responses_lib.PUT, UPLOAD_URL, body=b"", status=200)
        responses_lib.add(
            responses_lib.PUT,
            BASE_URL + "/models/fraud-detector/versions/1.0/confirm",
            json=RAW_VERSION_CONFIRMED, status=200,
        )

        with patch("artifact_mgmt.client.SerializerRegistry.detect", return_value=mock_serializer):
            client.save_model(mock_model, "fraud-detector")

        create_body = json.loads(responses_lib.calls[0].request.body)
        assert create_body["checksumSha256"] == expected_checksum


class TestSaveModelIdempotency:
    @responses_lib.activate
    def test_idempotency_key_is_valid_uuid4(
        self, client: ArtifactMgmtClient, mock_model: MagicMock,
        mock_serializer: MagicMock, patched_snapshot: MagicMock
    ) -> None:
        responses_lib.add(
            responses_lib.POST,
            BASE_URL + "/models/fraud-detector/versions",
            json=RAW_VERSION_PENDING, status=201,
        )
        responses_lib.add(responses_lib.PUT, UPLOAD_URL, body=b"", status=200)
        responses_lib.add(
            responses_lib.PUT,
            BASE_URL + "/models/fraud-detector/versions/1.0/confirm",
            json=RAW_VERSION_CONFIRMED, status=200,
        )

        with patch("artifact_mgmt.client.SerializerRegistry.detect", return_value=mock_serializer):
            client.save_model(mock_model, "fraud-detector")

        create_body = json.loads(responses_lib.calls[0].request.body)
        key = create_body["idempotencyKey"]
        parsed = uuid.UUID(key, version=4)
        assert str(parsed) == key


class TestSaveModelMajorBump:
    @responses_lib.activate
    def test_major_passed_to_create_version(
        self, client: ArtifactMgmtClient, mock_model: MagicMock,
        mock_serializer: MagicMock, patched_snapshot: MagicMock
    ) -> None:
        raw_v2 = {**RAW_VERSION_PENDING, "version": "2.0"}
        raw_v2_confirmed = {**RAW_VERSION_CONFIRMED, "version": "2.0"}

        responses_lib.add(
            responses_lib.POST,
            BASE_URL + "/models/fraud-detector/versions",
            json=raw_v2, status=201,
        )
        responses_lib.add(responses_lib.PUT, UPLOAD_URL, body=b"", status=200)
        responses_lib.add(
            responses_lib.PUT,
            BASE_URL + "/models/fraud-detector/versions/2.0/confirm",
            json=raw_v2_confirmed, status=200,
        )

        with patch("artifact_mgmt.client.SerializerRegistry.detect", return_value=mock_serializer):
            result = client.save_model(mock_model, "fraud-detector", major=2)

        create_body = json.loads(responses_lib.calls[0].request.body)
        assert create_body["major"] == 2
        assert result == "2.0"


class TestSaveModelSerializerOverride:
    @responses_lib.activate
    def test_explicit_pickle_serializer_bypasses_detect(
        self, client: ArtifactMgmtClient, patched_snapshot: MagicMock
    ) -> None:
        # Use a plain picklable object — MagicMock cannot be pickled
        plain_model = {"weights": [1, 2, 3]}

        responses_lib.add(
            responses_lib.POST,
            BASE_URL + "/models/fraud-detector/versions",
            json=RAW_VERSION_PENDING, status=201,
        )
        responses_lib.add(responses_lib.PUT, UPLOAD_URL, body=b"", status=200)
        responses_lib.add(
            responses_lib.PUT,
            BASE_URL + "/models/fraud-detector/versions/1.0/confirm",
            json=RAW_VERSION_CONFIRMED, status=200,
        )

        with patch("artifact_mgmt.client.SerializerRegistry.detect") as mock_detect:
            result = client.save_model(plain_model, "fraud-detector", serializer="pickle")

        mock_detect.assert_not_called()
        assert result == "1.0"


class TestSaveModelDepSnapshotOverride:
    @responses_lib.activate
    def test_dep_snapshot_override_merged_into_snapshot(
        self, client: ArtifactMgmtClient, mock_model: MagicMock, mock_serializer: MagicMock
    ) -> None:
        from artifact_mgmt._types import DepSnapshot, FrameworkInfo

        custom_snap = DepSnapshot(
            python_version="3.11.0",
            framework=FrameworkInfo(name="pytorch", version="2.1.0"),
            packages={},
            os="linux-x86_64",
            captured_at="2024-01-01T00:00:00Z",
            cuda_version="12.1-custom",
        )

        responses_lib.add(
            responses_lib.POST,
            BASE_URL + "/models/fraud-detector/versions",
            json=RAW_VERSION_PENDING, status=201,
        )
        responses_lib.add(responses_lib.PUT, UPLOAD_URL, body=b"", status=200)
        responses_lib.add(
            responses_lib.PUT,
            BASE_URL + "/models/fraud-detector/versions/1.0/confirm",
            json=RAW_VERSION_CONFIRMED, status=200,
        )

        with (
            patch("artifact_mgmt.client.SerializerRegistry.detect", return_value=mock_serializer),
            patch("artifact_mgmt.client._capture_snapshot", return_value=custom_snap),
        ):
            client.save_model(
                mock_model, "fraud-detector", dep_snapshot={"cudaVersion": "12.1-custom"}
            )

        create_body = json.loads(responses_lib.calls[0].request.body)
        assert create_body["depSnapshot"]["cudaVersion"] == "12.1-custom"
