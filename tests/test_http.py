from __future__ import annotations

import json
import pytest
import responses as responses_lib

from unittest.mock import patch, MagicMock
from artifact_mgmt._http import HttpClient
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


BASE_URL = "https://api.example.com/test"


def _make_client() -> HttpClient:
    mock_creds = MagicMock()
    mock_creds.access_key = "AKIATEST"
    mock_creds.secret_key = "testsecret"
    mock_creds.token = None

    mock_boto_session = MagicMock()
    mock_boto_session.get_credentials.return_value.get_frozen_credentials.return_value = mock_creds

    with patch("artifact_mgmt._http.boto3.Session", return_value=mock_boto_session):
        return HttpClient(BASE_URL)


@pytest.fixture
def client() -> HttpClient:
    return _make_client()


def test_client_raises_auth_error_when_no_aws_credentials() -> None:
    mock_boto_session = MagicMock()
    mock_boto_session.get_credentials.return_value = None
    with patch("artifact_mgmt._http.boto3.Session", return_value=mock_boto_session):
        with pytest.raises(AuthError):
            HttpClient(BASE_URL)


class TestRaiseForResponse:
    @responses_lib.activate
    def test_403_raises_auth_error(self, client: HttpClient) -> None:
        responses_lib.add(responses_lib.GET, BASE_URL + "/models", status=403, body="Forbidden")
        with pytest.raises(AuthError):
            client.request("GET", "/models")

    @responses_lib.activate
    def test_404_model_not_found_raises_model_not_found_error(self, client: HttpClient) -> None:
        responses_lib.add(
            responses_lib.GET, BASE_URL + "/models/x",
            json={"code": "ModelNotFound", "message": "not found"},
            status=404,
        )
        with pytest.raises(ModelNotFoundError):
            client.request("GET", "/models/x")

    @responses_lib.activate
    def test_404_version_not_found_raises_version_not_found_error(self, client: HttpClient) -> None:
        responses_lib.add(
            responses_lib.GET, BASE_URL + "/models/x/versions/1.0",
            json={"code": "VersionNotFound", "message": "not found"},
            status=404,
        )
        with pytest.raises(VersionNotFoundError):
            client.request("GET", "/models/x/versions/1.0")

    @responses_lib.activate
    def test_404_upload_not_found_raises_upload_not_found_error(self, client: HttpClient) -> None:
        responses_lib.add(
            responses_lib.GET, BASE_URL + "/upload",
            json={"code": "UploadNotFound", "message": "not found"},
            status=404,
        )
        with pytest.raises(UploadNotFoundError):
            client.request("GET", "/upload")

    @responses_lib.activate
    def test_404_unknown_code_raises_service_error(self, client: HttpClient) -> None:
        responses_lib.add(
            responses_lib.GET, BASE_URL + "/foo",
            json={"code": "SomethingElse"},
            status=404,
        )
        with pytest.raises(ServiceError):
            client.request("GET", "/foo")

    @responses_lib.activate
    def test_409_model_already_exists_raises_correct_error(self, client: HttpClient) -> None:
        responses_lib.add(
            responses_lib.POST, BASE_URL + "/models",
            json={"code": "ModelAlreadyExists"},
            status=409,
        )
        with pytest.raises(ModelAlreadyExistsError):
            client.request("POST", "/models", body={"modelName": "x"})

    @responses_lib.activate
    def test_409_version_conflict_raises_version_conflict_error(self, client: HttpClient) -> None:
        responses_lib.add(
            responses_lib.POST, BASE_URL + "/versions",
            json={"code": "VersionConflict"},
            status=409,
        )
        with pytest.raises(VersionConflictError):
            client.request("POST", "/versions", body={})

    @responses_lib.activate
    def test_409_idempotency_mismatch_raises_correct_error(self, client: HttpClient) -> None:
        responses_lib.add(
            responses_lib.POST, BASE_URL + "/versions",
            json={"code": "IdempotencyMismatch"},
            status=409,
        )
        with pytest.raises(IdempotencyMismatchError):
            client.request("POST", "/versions", body={})

    @responses_lib.activate
    def test_409_checksum_mismatch_raises_correct_error(self, client: HttpClient) -> None:
        responses_lib.add(
            responses_lib.POST, BASE_URL + "/versions",
            json={"code": "ChecksumMismatch"},
            status=409,
        )
        with pytest.raises(ChecksumMismatchError):
            client.request("POST", "/versions", body={})

    @responses_lib.activate
    def test_409_unknown_code_raises_service_error(self, client: HttpClient) -> None:
        responses_lib.add(
            responses_lib.POST, BASE_URL + "/foo",
            json={"code": "Unknown"},
            status=409,
        )
        with pytest.raises(ServiceError):
            client.request("POST", "/foo", body={})

    @responses_lib.activate
    def test_500_raises_service_error(self, client: HttpClient) -> None:
        responses_lib.add(responses_lib.GET, BASE_URL + "/models", status=500, body="oops")
        with pytest.raises(ServiceError):
            client.request("GET", "/models")

    @responses_lib.activate
    def test_400_raises_requests_http_error(self, client: HttpClient) -> None:
        responses_lib.add(responses_lib.POST, BASE_URL + "/models", status=400, body="bad request")
        with pytest.raises(Exception):
            client.request("POST", "/models", body={})

    @responses_lib.activate
    def test_503_raises_service_error(self, client: HttpClient) -> None:
        responses_lib.add(responses_lib.GET, BASE_URL + "/models", status=503, body="unavailable")
        with pytest.raises(ServiceError):
            client.request("GET", "/models")


class TestRequestMethod:
    @responses_lib.activate
    def test_200_returns_parsed_json(self, client: HttpClient) -> None:
        responses_lib.add(
            responses_lib.GET, BASE_URL + "/models",
            json={"items": []},
            status=200,
        )
        result = client.request("GET", "/models")
        assert result == {"items": []}

    @responses_lib.activate
    def test_204_returns_empty_dict(self, client: HttpClient) -> None:
        responses_lib.add(responses_lib.DELETE, BASE_URL + "/models/x", status=204)
        result = client.request("DELETE", "/models/x")
        assert result == {}

    @responses_lib.activate
    def test_request_passes_query_params(self, client: HttpClient) -> None:
        responses_lib.add(
            responses_lib.GET, BASE_URL + "/models",
            json={"items": []},
            status=200,
        )
        client.request("GET", "/models", params={"pageToken": "tok123"})
        assert "pageToken=tok123" in responses_lib.calls[0].request.url

    @responses_lib.activate
    def test_request_sends_json_body(self, client: HttpClient) -> None:
        responses_lib.add(
            responses_lib.POST, BASE_URL + "/models",
            json={"modelName": "x"},
            status=201,
        )
        client.request("POST", "/models", body={"modelName": "x"})
        sent = json.loads(responses_lib.calls[0].request.body)
        assert sent == {"modelName": "x"}


class TestUpload:
    @responses_lib.activate
    def test_upload_sets_content_type_header(self, client: HttpClient) -> None:
        responses_lib.add(responses_lib.PUT, "https://s3.example.com/upload", status=200)
        client.upload("https://s3.example.com/upload", b"model bytes")
        assert responses_lib.calls[0].request.headers["Content-Type"] == "application/octet-stream"

    @responses_lib.activate
    def test_upload_sends_correct_bytes(self, client: HttpClient) -> None:
        responses_lib.add(responses_lib.PUT, "https://s3.example.com/upload", status=200)
        client.upload("https://s3.example.com/upload", b"hello")
        assert responses_lib.calls[0].request.body == b"hello"

    @responses_lib.activate
    def test_upload_403_raises_auth_error(self, client: HttpClient) -> None:
        responses_lib.add(responses_lib.PUT, "https://s3.example.com/upload", status=403, body="forbidden")
        with pytest.raises(AuthError):
            client.upload("https://s3.example.com/upload", b"data")
