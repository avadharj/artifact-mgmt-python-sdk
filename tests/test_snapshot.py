from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch
from artifact_mgmt._types import DepSnapshot
from artifact_mgmt._snapshot import capture


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_dist(name: str, version: str) -> MagicMock:
    dist = MagicMock()
    dist.metadata = {"Name": name, "Version": version}
    return dist


def _make_torch_mock(cuda_version: str | None = "11.8") -> MagicMock:
    torch = MagicMock(name="torch")
    torch.version = MagicMock()
    torch.version.cuda = cuda_version
    return torch


# ---------------------------------------------------------------------------
# Story 4.1 — capture()
# ---------------------------------------------------------------------------


class TestCaptureBasicFields:
    def test_returns_dep_snapshot_instance(self):
        with (
            patch("artifact_mgmt._snapshot.SerializerRegistry.detect") as mock_detect,
            patch("artifact_mgmt._snapshot.platform.python_version", return_value="3.11.0"),
            patch("artifact_mgmt._snapshot.platform.system", return_value="Linux"),
            patch("artifact_mgmt._snapshot.platform.machine", return_value="x86_64"),
            patch("importlib.metadata.version", return_value="2.0.0"),
            patch("importlib.metadata.distributions", return_value=[]),
        ):
            mock_detect.return_value = MagicMock(framework_name="pytorch")
            sys.modules.pop("torch", None)
            result = capture(MagicMock())
        assert isinstance(result, DepSnapshot)

    def test_python_version_from_platform(self):
        with (
            patch("artifact_mgmt._snapshot.SerializerRegistry.detect") as mock_detect,
            patch("artifact_mgmt._snapshot.platform.python_version", return_value="3.11.5"),
            patch("artifact_mgmt._snapshot.platform.system", return_value="Darwin"),
            patch("artifact_mgmt._snapshot.platform.machine", return_value="arm64"),
            patch("importlib.metadata.version", return_value="2.0.0"),
            patch("importlib.metadata.distributions", return_value=[]),
        ):
            mock_detect.return_value = MagicMock(framework_name="pytorch")
            sys.modules.pop("torch", None)
            result = capture(MagicMock())
        assert result.python_version == "3.11.5"

    def test_os_string_is_system_and_machine(self):
        with (
            patch("artifact_mgmt._snapshot.SerializerRegistry.detect") as mock_detect,
            patch("artifact_mgmt._snapshot.platform.python_version", return_value="3.11.0"),
            patch("artifact_mgmt._snapshot.platform.system", return_value="Linux"),
            patch("artifact_mgmt._snapshot.platform.machine", return_value="x86_64"),
            patch("importlib.metadata.version", return_value="1.0"),
            patch("importlib.metadata.distributions", return_value=[]),
        ):
            mock_detect.return_value = MagicMock(framework_name="sklearn")
            sys.modules.pop("torch", None)
            result = capture(MagicMock())
        assert result.os == "linux-x86_64"

    def test_captured_at_is_iso8601_utc_string(self):
        with (
            patch("artifact_mgmt._snapshot.SerializerRegistry.detect") as mock_detect,
            patch("artifact_mgmt._snapshot.platform.python_version", return_value="3.11.0"),
            patch("artifact_mgmt._snapshot.platform.system", return_value="Linux"),
            patch("artifact_mgmt._snapshot.platform.machine", return_value="x86_64"),
            patch("importlib.metadata.version", return_value="1.0"),
            patch("importlib.metadata.distributions", return_value=[]),
        ):
            mock_detect.return_value = MagicMock(framework_name="sklearn")
            sys.modules.pop("torch", None)
            result = capture(MagicMock())
        assert result.captured_at.endswith("Z")
        assert "T" in result.captured_at


class TestCaptureFramework:
    def test_framework_name_comes_from_registry_detect(self):
        with (
            patch("artifact_mgmt._snapshot.SerializerRegistry.detect") as mock_detect,
            patch("artifact_mgmt._snapshot.platform.python_version", return_value="3.11.0"),
            patch("artifact_mgmt._snapshot.platform.system", return_value="Linux"),
            patch("artifact_mgmt._snapshot.platform.machine", return_value="x86_64"),
            patch("importlib.metadata.version", return_value="1.12.0"),
            patch("importlib.metadata.distributions", return_value=[]),
        ):
            mock_detect.return_value = MagicMock(framework_name="pytorch")
            sys.modules.pop("torch", None)
            result = capture(MagicMock())
        assert result.framework.name == "pytorch"
        assert result.framework.version == "1.12.0"

    def test_framework_version_unknown_when_package_not_found(self):
        import importlib.metadata as _meta
        with (
            patch("artifact_mgmt._snapshot.SerializerRegistry.detect") as mock_detect,
            patch("artifact_mgmt._snapshot.platform.python_version", return_value="3.11.0"),
            patch("artifact_mgmt._snapshot.platform.system", return_value="Linux"),
            patch("artifact_mgmt._snapshot.platform.machine", return_value="x86_64"),
            patch(
                "importlib.metadata.version",
                side_effect=_meta.PackageNotFoundError("torch"),
            ),
            patch("importlib.metadata.distributions", return_value=[]),
        ):
            mock_detect.return_value = MagicMock(framework_name="pytorch")
            sys.modules.pop("torch", None)
            result = capture(MagicMock())
        assert result.framework.version == "unknown"


class TestCapturePackages:
    def test_packages_dict_populated_from_distributions(self):
        dists = [_make_fake_dist("numpy", "1.24.0"), _make_fake_dist("pandas", "2.0.1")]
        with (
            patch("artifact_mgmt._snapshot.SerializerRegistry.detect") as mock_detect,
            patch("artifact_mgmt._snapshot.platform.python_version", return_value="3.11.0"),
            patch("artifact_mgmt._snapshot.platform.system", return_value="Linux"),
            patch("artifact_mgmt._snapshot.platform.machine", return_value="x86_64"),
            patch("importlib.metadata.version", return_value="1.0"),
            patch("importlib.metadata.distributions", return_value=dists),
        ):
            mock_detect.return_value = MagicMock(framework_name="sklearn")
            sys.modules.pop("torch", None)
            result = capture(MagicMock())
        assert result.packages == {"numpy": "1.24.0", "pandas": "2.0.1"}


class TestCaptureCudaVersion:
    def test_cuda_version_is_none_when_torch_not_installed(self):
        with (
            patch("artifact_mgmt._snapshot.SerializerRegistry.detect") as mock_detect,
            patch("artifact_mgmt._snapshot.platform.python_version", return_value="3.11.0"),
            patch("artifact_mgmt._snapshot.platform.system", return_value="Linux"),
            patch("artifact_mgmt._snapshot.platform.machine", return_value="x86_64"),
            patch("importlib.metadata.version", return_value="1.0"),
            patch("importlib.metadata.distributions", return_value=[]),
        ):
            mock_detect.return_value = MagicMock(framework_name="sklearn")
            sys.modules["torch"] = None  # type: ignore[assignment]
            result = capture(MagicMock())
            sys.modules.pop("torch", None)
        assert result.cuda_version is None

    def test_cuda_version_populated_when_torch_available(self):
        torch_mock = _make_torch_mock(cuda_version="11.8")
        with (
            patch("artifact_mgmt._snapshot.SerializerRegistry.detect") as mock_detect,
            patch("artifact_mgmt._snapshot.platform.python_version", return_value="3.11.0"),
            patch("artifact_mgmt._snapshot.platform.system", return_value="Linux"),
            patch("artifact_mgmt._snapshot.platform.machine", return_value="x86_64"),
            patch("importlib.metadata.version", return_value="2.0.0"),
            patch("importlib.metadata.distributions", return_value=[]),
            patch.dict(sys.modules, {"torch": torch_mock}),
        ):
            mock_detect.return_value = MagicMock(framework_name="pytorch")
            result = capture(MagicMock())
        assert result.cuda_version == "11.8"

    def test_cuda_version_none_when_torch_has_no_cuda(self):
        torch_mock = _make_torch_mock(cuda_version=None)
        with (
            patch("artifact_mgmt._snapshot.SerializerRegistry.detect") as mock_detect,
            patch("artifact_mgmt._snapshot.platform.python_version", return_value="3.11.0"),
            patch("artifact_mgmt._snapshot.platform.system", return_value="Linux"),
            patch("artifact_mgmt._snapshot.platform.machine", return_value="x86_64"),
            patch("importlib.metadata.version", return_value="2.0.0"),
            patch("importlib.metadata.distributions", return_value=[]),
            patch.dict(sys.modules, {"torch": torch_mock}),
        ):
            mock_detect.return_value = MagicMock(framework_name="pytorch")
            result = capture(MagicMock())
        assert result.cuda_version is None


class TestCaptureOverride:
    def _base_patches(self, mock_detect: MagicMock) -> list:
        return []

    def test_override_cuda_version(self):
        with (
            patch("artifact_mgmt._snapshot.SerializerRegistry.detect") as mock_detect,
            patch("artifact_mgmt._snapshot.platform.python_version", return_value="3.11.0"),
            patch("artifact_mgmt._snapshot.platform.system", return_value="Linux"),
            patch("artifact_mgmt._snapshot.platform.machine", return_value="x86_64"),
            patch("importlib.metadata.version", return_value="1.0"),
            patch("importlib.metadata.distributions", return_value=[]),
        ):
            mock_detect.return_value = MagicMock(framework_name="sklearn")
            sys.modules["torch"] = None  # type: ignore[assignment]
            result = capture(MagicMock(), override={"cudaVersion": "12.1-custom"})
            sys.modules.pop("torch", None)
        assert result.cuda_version == "12.1-custom"

    def test_override_only_affects_specified_field(self):
        with (
            patch("artifact_mgmt._snapshot.SerializerRegistry.detect") as mock_detect,
            patch("artifact_mgmt._snapshot.platform.python_version", return_value="3.11.0"),
            patch("artifact_mgmt._snapshot.platform.system", return_value="Linux"),
            patch("artifact_mgmt._snapshot.platform.machine", return_value="x86_64"),
            patch("importlib.metadata.version", return_value="1.0"),
            patch("importlib.metadata.distributions", return_value=[]),
        ):
            mock_detect.return_value = MagicMock(framework_name="sklearn")
            sys.modules["torch"] = None  # type: ignore[assignment]
            result = capture(MagicMock(), override={"cudaVersion": "12.1"})
            sys.modules.pop("torch", None)
        assert result.python_version == "3.11.0"
        assert result.os == "linux-x86_64"
        assert result.cuda_version == "12.1"

    def test_override_python_version(self):
        with (
            patch("artifact_mgmt._snapshot.SerializerRegistry.detect") as mock_detect,
            patch("artifact_mgmt._snapshot.platform.python_version", return_value="3.11.0"),
            patch("artifact_mgmt._snapshot.platform.system", return_value="Linux"),
            patch("artifact_mgmt._snapshot.platform.machine", return_value="x86_64"),
            patch("importlib.metadata.version", return_value="1.0"),
            patch("importlib.metadata.distributions", return_value=[]),
        ):
            mock_detect.return_value = MagicMock(framework_name="sklearn")
            sys.modules["torch"] = None  # type: ignore[assignment]
            result = capture(MagicMock(), override={"pythonVersion": "3.12.0"})
            sys.modules.pop("torch", None)
        assert result.python_version == "3.12.0"

    def test_no_override_does_not_mutate_snapshot(self):
        with (
            patch("artifact_mgmt._snapshot.SerializerRegistry.detect") as mock_detect,
            patch("artifact_mgmt._snapshot.platform.python_version", return_value="3.11.0"),
            patch("artifact_mgmt._snapshot.platform.system", return_value="Linux"),
            patch("artifact_mgmt._snapshot.platform.machine", return_value="x86_64"),
            patch("importlib.metadata.version", return_value="1.0"),
            patch("importlib.metadata.distributions", return_value=[]),
        ):
            mock_detect.return_value = MagicMock(framework_name="sklearn")
            sys.modules["torch"] = None  # type: ignore[assignment]
            result = capture(MagicMock(), override=None)
            sys.modules.pop("torch", None)
        assert result.python_version == "3.11.0"
