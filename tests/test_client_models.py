from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import responses as responses_lib

from artifact_mgmt.client import ArtifactMgmtClient, _parse_dep_snapshot, _parse_version
from artifact_mgmt._exceptions import ModelNotFoundError, ModelAlreadyExistsError
from artifact_mgmt._types import Model

BASE_URL = "https://idco76hrk9.execute-api.us-east-1.amazonaws.com/gamma"

RAW_MODEL = {
    "modelName": "fraud-detector",
    "owner": "team-ml",
    "status": "ACTIVE",
    "createdAt": "2024-01-01T00:00:00Z",
    "updatedAt": "2024-01-01T00:00:00Z",
    "frameworkHint": "pytorch",
    "description": "fraud model",
    "latestMajor": 1,
    "latestMinor": 2,
}


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


RAW_DEP_SNAPSHOT = {
    "pythonVersion": "3.11.0",
    "framework": {"name": "pytorch", "version": "2.1.0"},
    "packages": {"torch": "2.1.0"},
    "os": "linux-x86_64",
    "capturedAt": "2024-01-01T00:00:00Z",
    "cudaVersion": "12.1",
}

RAW_VERSION = {
    "modelName": "fraud-detector",
    "version": "1.0",
    "status": "READY",
    "depSnapshot": RAW_DEP_SNAPSHOT,
    "createdAt": "2024-01-01T00:00:00Z",
    "uploadUrl": None,
    "downloadUrl": "https://s3.example.com/weights",
}


class TestParseHelpers:
    def test_parse_dep_snapshot_maps_all_fields_correctly(self) -> None:
        snap = _parse_dep_snapshot(RAW_DEP_SNAPSHOT)
        assert snap.python_version == "3.11.0"
        assert snap.framework.name == "pytorch"
        assert snap.framework.version == "2.1.0"
        assert snap.packages == {"torch": "2.1.0"}
        assert snap.os == "linux-x86_64"
        assert snap.captured_at == "2024-01-01T00:00:00Z"
        assert snap.cuda_version == "12.1"

    def test_parse_version_maps_all_fields_correctly(self) -> None:
        version = _parse_version(RAW_VERSION)
        assert version.model_name == "fraud-detector"
        assert version.version == "1.0"
        assert version.status == "READY"
        assert version.download_url == "https://s3.example.com/weights"
        assert version.dep_snapshot.framework.name == "pytorch"


class TestCreateModel:
    @responses_lib.activate
    def test_create_model_returns_model_on_success(self, client: ArtifactMgmtClient) -> None:
        responses_lib.add(
            responses_lib.POST, BASE_URL + "/models",
            json=RAW_MODEL,
            status=201,
        )
        model = client.create_model("fraud-detector", framework_hint="pytorch", description="fraud model")
        assert isinstance(model, Model)
        assert model.model_name == "fraud-detector"
        assert model.framework_hint == "pytorch"
        assert model.description == "fraud model"

    @responses_lib.activate
    def test_create_model_sends_only_allowed_fields(self, client: ArtifactMgmtClient) -> None:
        responses_lib.add(
            responses_lib.POST, BASE_URL + "/models",
            json=RAW_MODEL,
            status=201,
        )
        client.create_model("fraud-detector", framework_hint="pytorch")
        import json
        body = json.loads(responses_lib.calls[0].request.body)
        assert set(body.keys()) <= {"modelName", "frameworkHint", "description"}

    @responses_lib.activate
    def test_create_model_raises_model_already_exists_error_on_409(
        self, client: ArtifactMgmtClient
    ) -> None:
        responses_lib.add(
            responses_lib.POST, BASE_URL + "/models",
            json={"code": "ModelAlreadyExists", "message": "already exists"},
            status=409,
        )
        with pytest.raises(ModelAlreadyExistsError):
            client.create_model("fraud-detector")

    @responses_lib.activate
    def test_create_model_without_optional_fields_omits_them_from_body(
        self, client: ArtifactMgmtClient
    ) -> None:
        responses_lib.add(
            responses_lib.POST, BASE_URL + "/models",
            json=RAW_MODEL,
            status=201,
        )
        client.create_model("fraud-detector")
        import json
        body = json.loads(responses_lib.calls[0].request.body)
        assert "frameworkHint" not in body
        assert "description" not in body


class TestGetModel:
    @responses_lib.activate
    def test_get_model_returns_model_with_correct_fields(self, client: ArtifactMgmtClient) -> None:
        responses_lib.add(
            responses_lib.GET, BASE_URL + "/models/fraud-detector",
            json=RAW_MODEL,
            status=200,
        )
        model = client.get_model("fraud-detector")
        assert model.model_name == "fraud-detector"
        assert model.owner == "team-ml"
        assert model.latest_major == 1
        assert model.latest_minor == 2

    @responses_lib.activate
    def test_get_model_raises_model_not_found_error_when_model_does_not_exist(
        self, client: ArtifactMgmtClient
    ) -> None:
        responses_lib.add(
            responses_lib.GET, BASE_URL + "/models/nonexistent",
            json={"code": "ModelNotFound", "message": "not found"},
            status=404,
        )
        with pytest.raises(ModelNotFoundError):
            client.get_model("nonexistent")


class TestListModels:
    @responses_lib.activate
    def test_list_models_returns_page_iterator(self, client: ArtifactMgmtClient) -> None:
        responses_lib.add(
            responses_lib.GET, BASE_URL + "/models",
            json={"items": [RAW_MODEL], "nextPageToken": None},
            status=200,
        )
        result = list(client.list_models())
        assert len(result) == 1
        assert result[0].model_name == "fraud-detector"

    @responses_lib.activate
    def test_list_models_is_lazy_iterator_not_a_list(self, client: ArtifactMgmtClient) -> None:
        responses_lib.add(
            responses_lib.GET, BASE_URL + "/models",
            json={"items": [], "nextPageToken": None},
            status=200,
        )
        iterator = client.list_models()
        # Should not have made any HTTP call yet
        assert len(responses_lib.calls) == 0
        list(iterator)
        assert len(responses_lib.calls) == 1

    @responses_lib.activate
    def test_list_models_paginates_across_multiple_pages(self, client: ArtifactMgmtClient) -> None:
        model_b = {**RAW_MODEL, "modelName": "churn-predictor"}
        model_c = {**RAW_MODEL, "modelName": "recommender"}
        responses_lib.add(
            responses_lib.GET, BASE_URL + "/models",
            json={"items": [RAW_MODEL], "nextPageToken": "tok1"},
            status=200,
        )
        responses_lib.add(
            responses_lib.GET, BASE_URL + "/models",
            json={"items": [model_b], "nextPageToken": "tok2"},
            status=200,
        )
        responses_lib.add(
            responses_lib.GET, BASE_URL + "/models",
            json={"items": [model_c], "nextPageToken": None},
            status=200,
        )
        result = list(client.list_models())
        assert [m.model_name for m in result] == ["fraud-detector", "churn-predictor", "recommender"]
        assert len(responses_lib.calls) == 3

    @responses_lib.activate
    def test_list_models_uses_items_response_key(self, client: ArtifactMgmtClient) -> None:
        responses_lib.add(
            responses_lib.GET, BASE_URL + "/models",
            json={"items": [RAW_MODEL], "nextPageToken": None},
            status=200,
        )
        result = list(client.list_models())
        assert result[0].model_name == "fraud-detector"


class TestDeleteModel:
    @responses_lib.activate
    def test_delete_model_returns_none_on_204(self, client: ArtifactMgmtClient) -> None:
        responses_lib.add(
            responses_lib.DELETE, BASE_URL + "/models/fraud-detector",
            status=204,
        )
        result = client.delete_model("fraud-detector")
        assert result is None

    @responses_lib.activate
    def test_delete_model_raises_model_not_found_error_when_model_does_not_exist(
        self, client: ArtifactMgmtClient
    ) -> None:
        responses_lib.add(
            responses_lib.DELETE, BASE_URL + "/models/nonexistent",
            json={"code": "ModelNotFound", "message": "not found"},
            status=404,
        )
        with pytest.raises(ModelNotFoundError):
            client.delete_model("nonexistent")
