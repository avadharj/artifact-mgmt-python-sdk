import pytest
from artifact_mgmt import (
    ArtifactMgmtError,
    ModelNotFoundError,
    ModelAlreadyExistsError,
    VersionNotFoundError,
    VersionConflictError,
    IdempotencyMismatchError,
    ChecksumMismatchError,
    UploadNotFoundError,
    AuthError,
    ServiceError,
    FrameworkNotInstalledError,
    UnknownSerializerError,
)


ALL_EXCEPTIONS = [
    ModelNotFoundError,
    ModelAlreadyExistsError,
    VersionNotFoundError,
    VersionConflictError,
    IdempotencyMismatchError,
    ChecksumMismatchError,
    UploadNotFoundError,
    AuthError,
    ServiceError,
    FrameworkNotInstalledError,
    UnknownSerializerError,
]


@pytest.mark.parametrize("exc_class", ALL_EXCEPTIONS)
def test_all_exceptions_are_artifact_mgmt_error(exc_class):
    instance = exc_class("test message")
    assert isinstance(instance, ArtifactMgmtError)


@pytest.mark.parametrize("exc_class", ALL_EXCEPTIONS)
def test_all_exceptions_are_catchable_as_base(exc_class):
    with pytest.raises(ArtifactMgmtError):
        raise exc_class("test message")
