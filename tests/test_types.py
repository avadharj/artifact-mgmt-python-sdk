from dataclasses import asdict
from artifact_mgmt._types import FrameworkInfo, DepSnapshot, Model, Version


def _make_framework_info() -> FrameworkInfo:
    return FrameworkInfo(name="pytorch", version="2.1.0")


def _make_dep_snapshot(**kwargs: object) -> DepSnapshot:
    return DepSnapshot(
        python_version="3.11.0",
        framework=_make_framework_info(),
        packages={"torch": "2.1.0"},
        os="linux-x86_64",
        captured_at="2024-01-01T00:00:00Z",
        **kwargs,  # type: ignore[arg-type]
    )


def _make_model(**kwargs: object) -> Model:
    return Model(
        model_name="fraud-detector",
        owner="team-ml",
        status="ACTIVE",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        **kwargs,  # type: ignore[arg-type]
    )


def _make_version(**kwargs: object) -> Version:
    return Version(
        model_name="fraud-detector",
        version="1.0",
        status="READY",
        dep_snapshot=_make_dep_snapshot(),
        created_at="2024-01-01T00:00:00Z",
        **kwargs,  # type: ignore[arg-type]
    )


class TestFrameworkInfo:
    def test_round_trip_preserves_all_fields(self) -> None:
        original = _make_framework_info()
        d = asdict(original)
        restored = FrameworkInfo(**d)
        assert restored == original

    def test_fields_accessible(self) -> None:
        fi = _make_framework_info()
        assert fi.name == "pytorch"
        assert fi.version == "2.1.0"


class TestDepSnapshot:
    def test_round_trip_preserves_all_fields(self) -> None:
        original = _make_dep_snapshot()
        d = asdict(original)
        restored = DepSnapshot(
            python_version=d["python_version"],
            framework=FrameworkInfo(**d["framework"]),
            packages=d["packages"],
            os=d["os"],
            captured_at=d["captured_at"],
            cuda_version=d["cuda_version"],
        )
        assert restored == original

    def test_cuda_version_defaults_to_none(self) -> None:
        snap = _make_dep_snapshot()
        assert snap.cuda_version is None

    def test_cuda_version_can_be_set(self) -> None:
        snap = _make_dep_snapshot(cuda_version="12.1")
        assert snap.cuda_version == "12.1"


class TestModel:
    def test_round_trip_preserves_all_fields(self) -> None:
        original = _make_model(
            framework_hint="pytorch",
            description="fraud model",
            latest_major=2,
            latest_minor=3,
        )
        d = asdict(original)
        restored = Model(**d)
        assert restored == original

    def test_optional_fields_default_to_none(self) -> None:
        m = _make_model()
        assert m.framework_hint is None
        assert m.description is None

    def test_version_counters_default_to_zero(self) -> None:
        m = _make_model()
        assert m.latest_major == 0
        assert m.latest_minor == 0


class TestVersion:
    def test_round_trip_preserves_all_fields(self) -> None:
        original = _make_version(
            upload_url="https://s3.example.com/upload",
            upload_url_expires_at="2024-01-01T01:00:00Z",
            download_url="https://s3.example.com/download",
            checksum_sha256="abc123==",
        )
        d = asdict(original)
        restored = Version(
            model_name=d["model_name"],
            version=d["version"],
            status=d["status"],
            dep_snapshot=DepSnapshot(
                python_version=d["dep_snapshot"]["python_version"],
                framework=FrameworkInfo(**d["dep_snapshot"]["framework"]),
                packages=d["dep_snapshot"]["packages"],
                os=d["dep_snapshot"]["os"],
                captured_at=d["dep_snapshot"]["captured_at"],
                cuda_version=d["dep_snapshot"]["cuda_version"],
            ),
            created_at=d["created_at"],
            upload_url=d["upload_url"],
            upload_url_expires_at=d["upload_url_expires_at"],
            download_url=d["download_url"],
            checksum_sha256=d["checksum_sha256"],
        )
        assert restored == original

    def test_optional_fields_default_to_none(self) -> None:
        v = _make_version()
        assert v.upload_url is None
        assert v.upload_url_expires_at is None
        assert v.download_url is None
        assert v.checksum_sha256 is None
