# Python SDK for Artifact Management Service — Implementation Plan

## Context

Scientists who save and load ML models currently have to write boilerplate for every interaction with the artifact management service: constructing SigV4-signed HTTP requests, serializing models in the right format per framework, capturing environment snapshots, uploading to S3 presigned URLs, and confirming uploads. This library eliminates all of that.

The primary public API is two methods — `save_model()` and `load_model()` — plus low-level CRUD for model/version management. `load_model()` returns an `ArtifactModel` wrapper that carries version metadata and exposes cross-framework training utilities (`freeze`, `predict`, `fine_tune_params`) while forwarding any unknown attribute access to the native model object via `__getattr__`, so scientists never need to unwrap it.

**Locked design decisions:**
- `load_model()` returns `ArtifactModel` wrapper with `__getattr__` forwarding to native model
- Serialization backends: PyTorch/Lightning, HuggingFace Transformers, sklearn, TF/Keras, pickle fallback
- `dep_snapshot` is auto-captured at save time; scientist can pass overrides for specific fields
- Local disk cache is opt-in via `cache_dir=` constructor arg or `ARTIFACT_MGMT_CACHE_DIR` env var
- Stage configured via `ArtifactMgmtClient(stage="gamma")`; falls back to `ARTIFACT_MGMT_STAGE` env var. Supported named stages: `alpha`, `gamma`, `prod`. (`beta` is an internal pipeline stage, not scientist-facing.)
- AWS credentials via boto3 default chain (env vars → `~/.aws/credentials` → IAM role)
- `promote_model()` is the cross-stage promotion primitive — it loads from a source stage and saves to a destination stage in one call. Each stage is a fully isolated silo (separate DDB tables + S3 bucket); there is no automatic cross-stage sync.

**Live API gotchas:**
- `ListModels` response key is `"items"`, `ListVersions` is `"versions"`
- `ConfirmVersion` is `PUT` (not POST), body must be `{}`
- `checksumSha256` must be base64-encoded SHA-256, not hex
- S3 presigned PUT requires `Content-Type: application/octet-stream` — omitting it → 403
- Version path param is dotted string `"1.0"`, never split into major/minor

---

## Test coverage policy

**Every story's unit tests must achieve ≥ 90% line coverage on the files it introduces.** Coverage is measured with `pytest-cov` and enforced as a CI hard gate — the build fails if coverage drops below 90% across `artifact_mgmt/`. Per-story coverage notes are called out in each story's AC; the global gate is in Story 7.2.

---

## Repository layout (target)

```
artifact_mgmt/
├── __init__.py              # public exports: ArtifactMgmtClient, ArtifactModel, exceptions
├── client.py                # ArtifactMgmtClient — main entry point
├── _artifact_model.py       # ArtifactModel wrapper with __getattr__ forwarding
├── _types.py                # dataclasses: Model, Version, DepSnapshot, FrameworkInfo
├── _exceptions.py           # exception hierarchy
├── _http.py                 # SigV4 HTTP layer (requests + requests-aws4auth)
├── _pagination.py           # PageIterator — auto-fetches next pages
├── _snapshot.py             # dep_snapshot auto-capture (sys, importlib.metadata, platform)
├── _cache.py                # local disk cache keyed by model_name + version
└── serializers/
    ├── __init__.py          # SerializerRegistry: detect from model type or dep_snapshot
    ├── _base.py             # Serializer ABC: serialize(model)->bytes, deserialize(bytes)->model
    ├── _torch.py            # torch.save / torch.load state_dict; Lightning .ckpt support
    ├── _huggingface.py      # save_pretrained / from_pretrained; bundles tokenizer + config
    ├── _sklearn.py          # joblib.dump / joblib.load (covers sklearn, XGBoost, LightGBM)
    ├── _tensorflow.py       # model.save / tf.keras.models.load_model
    └── _pickle.py           # generic pickle fallback

tests/
├── conftest.py              # shared fixtures: mock HTTP responses, tiny model stubs
├── test_client_models.py    # Model CRUD unit tests
├── test_client_versions.py  # Version CRUD + upload + confirm unit tests
├── test_artifact_model.py   # ArtifactModel wrapper + __getattr__ + training utils
├── test_serializers.py      # round-trip tests per serializer (no real framework needed for most)
├── test_snapshot.py         # dep_snapshot capture + override
├── test_pagination.py       # PageIterator multi-page traversal
└── test_cache.py            # cache hit/miss/invalidation
```

---

## Public API surface

```python
from artifact_mgmt import ArtifactMgmtClient

# --- Client setup ---
client = ArtifactMgmtClient(stage="gamma")                          # resolves endpoint internally
client = ArtifactMgmtClient(stage="prod")                           # prod stage
client = ArtifactMgmtClient(stage="gamma", cache_dir="~/.artifact-mgmt/cache")
client = ArtifactMgmtClient(endpoint_url="https://...")            # custom deployment

# --- High-level (primary scientist API) ---
artifact = client.load_model("fraud-detector")                      # latest READY version
artifact = client.load_model("fraud-detector", version="2.1")       # specific version
client.save_model(model, "fraud-detector")                          # minor bump, auto snapshot
client.save_model(model, "fraud-detector", major=2)                 # major bump
client.save_model(model, "fraud-detector", dep_snapshot={"cudaVersion": "12.1-custom"})

# --- Cross-stage promotion ---
gamma_client = ArtifactMgmtClient(stage="gamma")
prod_client = ArtifactMgmtClient(stage="prod")
prod_version = gamma_client.promote_model(
    "fraud-detector", version="2.1", dest=prod_client
)                                                                    # -> version string in prod

# --- ArtifactModel wrapper ---
artifact.model                        # native nn.Module / PreTrainedModel / etc
artifact.version                      # "2.1"
artifact.model_name                   # "fraud-detector"
artifact.dep_snapshot                 # DepSnapshot dataclass
artifact.freeze(n_layers=6)           # framework-agnostic layer freezing
artifact.unfreeze(n_layers=6)         # unfreeze
artifact.fine_tune_params()           # list of trainable params
artifact.predict(X)                   # framework-agnostic inference
artifact.train()                      # forwarded to native model via __getattr__
artifact.parameters()                 # forwarded

# --- Low-level CRUD ---
client.create_model("fraud-detector", framework_hint="pytorch", description="...")
client.get_model("fraud-detector")                                  # -> Model
for m in client.list_models():                                      # PageIterator
    print(m.model_name)
client.delete_model("fraud-detector")

client.get_version("fraud-detector", "2.1")                        # -> Version
client.get_latest_version("fraud-detector")                        # -> Version
for v in client.list_versions("fraud-detector", include_pending=False):
    print(v.version)
client.delete_version("fraud-detector", "2.1")
```

---

## Epics and Stories

### Epic 1 — Foundation: types, exceptions, HTTP, pagination [~2 days]

**Goal:** All building blocks every other epic depends on. No business logic yet.

---

#### Story 1.1 — Exception hierarchy [S]

**File:** `artifact_mgmt/_exceptions.py`

```python
class ArtifactMgmtError(Exception): ...          # base; all SDK errors inherit from this
class ModelNotFoundError(ArtifactMgmtError): ...
class ModelAlreadyExistsError(ArtifactMgmtError): ...
class VersionNotFoundError(ArtifactMgmtError): ...
class VersionConflictError(ArtifactMgmtError): ...   # 409 concurrent increment
class IdempotencyMismatchError(ArtifactMgmtError): ...
class ChecksumMismatchError(ArtifactMgmtError): ...
class UploadNotFoundError(ArtifactMgmtError): ...
class AuthError(ArtifactMgmtError): ...              # 403
class ServiceError(ArtifactMgmtError): ...           # 5xx
class FrameworkNotInstalledError(ArtifactMgmtError): ...  # lazy import failed
class UnknownSerializerError(ArtifactMgmtError): ...
```

HTTP status → exception mapping lives in `_http.py` (Story 1.3), not here.

**AC:**
- All exception classes defined and importable from `artifact_mgmt`
- `ArtifactMgmtError` is the common base — callers can catch everything with one clause
- Unit test: each exception is an instance of `ArtifactMgmtError`
- Coverage ≥ 90% on `_exceptions.py`

---

#### Story 1.2 — Data types [S]

**File:** `artifact_mgmt/_types.py`

```python
@dataclass
class FrameworkInfo:
    name: str          # "pytorch" | "huggingface" | "sklearn" | "tensorflow" | "pickle"
    version: str

@dataclass
class DepSnapshot:
    python_version: str
    framework: FrameworkInfo
    packages: dict[str, str]
    os: str
    captured_at: str
    cuda_version: str | None = None

@dataclass
class Model:
    model_name: str
    owner: str
    status: str
    created_at: str
    updated_at: str
    framework_hint: str | None = None
    description: str | None = None
    latest_major: int = 0
    latest_minor: int = 0

@dataclass
class Version:
    model_name: str
    version: str           # "2.1"
    status: str            # PENDING | READY | DELETED | FAILED
    dep_snapshot: DepSnapshot
    created_at: str
    upload_url: str | None = None        # present on CreateVersion response only
    upload_url_expires_at: str | None = None
    download_url: str | None = None      # present on Get/GetLatest response
    checksum_sha256: str | None = None
```

Field names are camelCase in JSON; `_types.py` exposes snake_case Python dataclasses. Conversion handled in `_http.py`.

**AC:**
- All dataclasses defined, importable
- `DepSnapshot.cuda_version` is optional (None if not a GPU environment)
- Unit test: round-trip dict → dataclass → dict preserves all fields
- Coverage ≥ 90% on `_types.py`

---

#### Story 1.3 — SigV4 HTTP client [M]

**File:** `artifact_mgmt/_http.py`

```python
class HttpClient:
    def __init__(self, endpoint_url: str, region: str = "us-east-1"):
        # builds requests-aws4auth from boto3 default credential chain
        ...

    def request(self, method: str, path: str, *, body: dict | None = None,
                params: dict | None = None) -> dict:
        # signs request, sends, parses JSON response
        # maps HTTP status codes to exceptions
        ...
```

Status → exception mapping:
- 403 → `AuthError`
- 404 + `code: "ModelNotFound"` → `ModelNotFoundError`
- 404 + `code: "VersionNotFound"` → `VersionNotFoundError`
- 404 + `code: "UploadNotFound"` → `UploadNotFoundError`
- 409 + `code: "ModelAlreadyExists"` → `ModelAlreadyExistsError`
- 409 + `code: "VersionConflict"` → `VersionConflictError`
- 409 + `code: "IdempotencyMismatch"` → `IdempotencyMismatchError`
- 409 + `code: "ChecksumMismatch"` → `ChecksumMismatchError`
- 5xx → `ServiceError`

S3 upload is a plain `requests.put()` (no SigV4 — URL is already signed). Must include `Content-Type: application/octet-stream`.

**AC:**
- All status → exception mappings covered
- Unit tests use `responses` library to mock HTTP; no real network calls
- S3 upload helper sets correct Content-Type header
- boto3 credential chain used (env vars → `~/.aws/credentials` → IAM role)
- Coverage ≥ 90% on `_http.py`

---

#### Story 1.4 — Pagination iterator [S]

**File:** `artifact_mgmt/_pagination.py`

```python
class PageIterator(Generic[T]):
    """Lazy iterator that fetches the next page only when exhausted."""
    def __init__(self, fetch_page: Callable[[str | None], tuple[list[T], str | None]]): ...
    def __iter__(self) -> Iterator[T]: ...
    def __next__(self) -> T: ...
```

`fetch_page` receives the current `pageToken` (None on first call) and returns `(items, next_page_token)`. Iterator is exhausted when `next_page_token` is None.

**AC:**
- Fetches pages lazily (second page not fetched until first is exhausted)
- Works for both `list_models` (key: `items`) and `list_versions` (key: `versions`)
- Unit test: 3-page sequence, assert `fetch_page` called exactly 3 times
- Unit test: empty first page returns empty iterator immediately
- Coverage ≥ 90% on `_pagination.py`

---

### Epic 2 — Low-level REST API client [~1.5 days]

**Goal:** Full CRUD for Models and Versions. No serialization or framework logic.

---

#### Story 2.1 — Model CRUD [M]

**File:** `artifact_mgmt/client.py` (partial — Model methods only)

Methods: `create_model`, `get_model`, `list_models`, `delete_model`

Implementation notes:
- `list_models` returns `PageIterator[Model]` — wraps `GET /models` with page token threading
- `create_model` sends only `modelName`, `frameworkHint`, `description` — no other fields (unknown fields → 400)
- `delete_model` returns `None` on 204

**AC:**
- Happy path + error paths unit-tested with `responses` mock for each method
- `list_models` returns a lazy iterator, not a list
- `create_model` with duplicate name raises `ModelAlreadyExistsError`
- `get_model` on missing name raises `ModelNotFoundError`
- Coverage ≥ 90% on model-related paths in `client.py`

---

#### Story 2.2 — Version CRUD [M]

**File:** `artifact_mgmt/client.py` (partial — Version methods)

Methods: `get_version`, `get_latest_version`, `list_versions`, `delete_version`

Implementation notes:
- All version path params use dotted string `"1.0"` format
- `list_versions` returns `PageIterator[Version]` — key in response is `"versions"` (not `"items"`)
- `get_latest_version` hits `GET /models/{modelName}/versions/latest`
- `list_versions(include_pending=False)` passes `?includePending=true` when True

**AC:**
- All methods tested with `responses` mock
- `get_latest_version` on model with no READY versions raises `VersionNotFoundError`
- `list_versions` pagination correct across 3-page mock
- Coverage ≥ 90% on version read paths in `client.py`

---

#### Story 2.3 — CreateVersion + upload [M]

**File:** `artifact_mgmt/client.py`

Internal method `_create_version` + `_upload_artifact`:

```python
def _create_version(self, model_name, *, idempotency_key, dep_snapshot,
                    major=None, checksum_sha256=None) -> Version: ...

def _upload_artifact(self, upload_url: str, data: bytes,
                     checksum_sha256: str | None = None) -> None: ...
```

Notes:
- `idempotencyKey` and `depSnapshot` are required fields in the request body
- `checksumSha256` is base64-encoded `hashlib.sha256(data).digest()` — not hex
- Idempotency replay returns HTTP 200 (not 201) with a fresh `uploadUrl`
- S3 PUT must include `Content-Type: application/octet-stream`

**AC:**
- Happy path: create version → response has `version`, `uploadUrl`, `status=PENDING`
- Idempotency replay (200) handled identically to 201
- `checksumSha256` encoding tested: base64(sha256(data)) matches expected value
- Upload sends correct Content-Type header
- Coverage ≥ 90% on create/upload paths in `client.py`

---

#### Story 2.4 — ConfirmVersion [S]

**File:** `artifact_mgmt/client.py`

```python
def _confirm_version(self, model_name: str, version: str) -> Version: ...
```

Notes:
- Method is `PUT` (not POST)
- Body must be `{}` — null body → 400
- Path is `/models/{modelName}/versions/{version}/confirm` where version = `"1.0"`

**AC:**
- Correct HTTP method (PUT) verified in test
- Empty JSON body `{}` always sent
- Returns updated `Version` with `status=READY`
- Coverage ≥ 90% on confirm path in `client.py`

---

### Epic 3 — Serialization backends [~2 days]

**Goal:** Pluggable save/load per framework with auto-detection.

---

#### Story 3.1 — Serializer base + registry [M]

**Files:** `artifact_mgmt/serializers/_base.py`, `artifact_mgmt/serializers/__init__.py`

```python
class Serializer(ABC):
    framework_name: str   # "pytorch" | "huggingface" | "sklearn" | "tensorflow" | "pickle"

    @abstractmethod
    def can_handle(self, model: object) -> bool: ...

    @abstractmethod
    def serialize(self, model: object) -> bytes: ...

    @abstractmethod
    def deserialize(self, data: bytes) -> object: ...

    @abstractmethod
    def freeze(self, model: object, n_layers: int) -> None: ...

    @abstractmethod
    def unfreeze(self, model: object, n_layers: int) -> None: ...

    @abstractmethod
    def fine_tune_params(self, model: object) -> list: ...

    @abstractmethod
    def predict(self, model: object, X: object) -> object: ...


class SerializerRegistry:
    @staticmethod
    def detect(model: object) -> Serializer:
        """Detect serializer from model type (for save_model)."""

    @staticmethod
    def detect_from_snapshot(dep_snapshot: DepSnapshot) -> Serializer:
        """Detect serializer from dep_snapshot.framework.name (for load_model)."""
```

All framework imports inside serializers are **lazy** (inside method bodies) — the registry must not fail on import if torch/transformers/sklearn/tensorflow are not installed. Raises `FrameworkNotInstalledError` with install instructions if the required package is missing at call time.

**AC:**
- Registry correctly detects serializer from model type for each framework
- Registry correctly maps `dep_snapshot.framework.name` to serializer
- Lazy import: importing `artifact_mgmt` with only `requests` installed does not raise
- `UnknownSerializerError` raised for unrecognised model type when no serializer matches
- Coverage ≥ 90% on `serializers/__init__.py` and `serializers/_base.py`

---

#### Story 3.2 — PyTorch / Lightning serializer [M]

**File:** `artifact_mgmt/serializers/_torch.py`

Serialization:
- `serialize`: saves full model object via `torch.save(model, buf)` — required for round-trip without user providing the class definition
- `deserialize`: `torch.load(buf)`
- Lightning `.ckpt` detection: if `data` parses as a checkpoint dict with `state_dict` key, extract it

Training abstractions:
- `freeze(model, n_layers)`: iterate `model.named_parameters()`, set `requires_grad=False` on first `n_layers` layers by parameter group
- `unfreeze(model, n_layers)`: reverse
- `fine_tune_params(model)`: `[p for p in model.parameters() if p.requires_grad]`
- `predict(model, X)`: `model.eval(); torch.no_grad(); return model(X)`

**AC:**
- Round-trip test: serialize → deserialize → output matches for a tiny `nn.Linear(4, 2)` stub
- `freeze(model, 0)` freezes no params; `freeze(model, 999)` freezes all
- `predict` sets `eval()` mode and uses `no_grad` context
- `FrameworkNotInstalledError` raised with install hint if `torch` not importable
- Coverage ≥ 90% on `serializers/_torch.py`

---

#### Story 3.3 — HuggingFace Transformers serializer [M]

**File:** `artifact_mgmt/serializers/_huggingface.py`

Serialization:
- `serialize`: `model.save_pretrained(tmpdir)` → tar the directory → bytes. Includes `config.json`, weights (`pytorch_model.bin` or `model.safetensors`), tokenizer files if present.
- `deserialize`: untar to tempdir → `AutoModel.from_pretrained(tmpdir)`

Training abstractions:
- `freeze(model, n_layers)`: freeze `model.base_model.encoder.layer[:n_layers]` (BERT-style). Falls back to freezing first `n_layers` named children for non-BERT architectures.
- `predict(model, X)`: `model(**X)` — X is expected to be a dict of input tensors (HF convention)

**AC:**
- Round-trip test uses a tiny `AutoModel` stub (mocked — no real download)
- Tarball correctly bundles and extracts config + weights
- `freeze` handles both BERT-style and generic architectures
- Tokenizer files included in tarball when present
- Coverage ≥ 90% on `serializers/_huggingface.py`

---

#### Story 3.4 — sklearn serializer [S]

**File:** `artifact_mgmt/serializers/_sklearn.py`

Serialization:
- `serialize`: `joblib.dump(model, BytesIO())` → bytes
- `deserialize`: `joblib.load(BytesIO(data))`

Training abstractions:
- `freeze` / `unfreeze` / `fine_tune_params`: raise `NotImplementedError` with message "sklearn estimators do not support layer freezing"
- `predict(model, X)`: `model.predict(X)` — works for any sklearn-compatible estimator including XGBoost/LightGBM sklearn API

**AC:**
- Round-trip test with `sklearn.linear_model.LogisticRegression` stub (no real fit needed)
- `freeze` raises `NotImplementedError` with clear message
- `predict` calls `model.predict(X)`
- Coverage ≥ 90% on `serializers/_sklearn.py`

---

#### Story 3.5 — TF/Keras serializer [S]

**File:** `artifact_mgmt/serializers/_tensorflow.py`

Serialization:
- `serialize`: `model.save(tmpdir, save_format='tf')` → tar directory → bytes
- `deserialize`: untar → `tf.keras.models.load_model(tmpdir)`

Training abstractions:
- `freeze(model, n_layers)`: `model.layers[i].trainable = False` for first `n_layers`
- `predict(model, X)`: `model.predict(X)`

**AC:**
- Round-trip test with mocked Keras model
- `freeze` sets `layer.trainable = False` on correct layers
- Coverage ≥ 90% on `serializers/_tensorflow.py`

---

#### Story 3.6 — Pickle fallback serializer [S]

**File:** `artifact_mgmt/serializers/_pickle.py`

Serialization: `pickle.dumps(model)` / `pickle.loads(data)`

Training abstractions: all raise `NotImplementedError` — pickle is a generic fallback for save/load only.

Opt-in only: `client.save_model(model, "name", serializer="pickle")`. Never auto-detected.

**AC:**
- Round-trip test with a plain Python object
- Never returned by `SerializerRegistry.detect()` without explicit opt-in
- All training abstract methods raise `NotImplementedError`
- Coverage ≥ 90% on `serializers/_pickle.py`

---

### Epic 4 — dep_snapshot auto-capture [~0.5 days]

---

#### Story 4.1 — Environment inspector [M]

**File:** `artifact_mgmt/_snapshot.py`

```python
def capture(model: object, override: dict | None = None) -> DepSnapshot:
    """Inspect the current Python environment and return a DepSnapshot."""
```

Captures:
- `python_version`: `platform.python_version()`
- `framework`: detected from model type via `SerializerRegistry`; version via `importlib.metadata.version(framework_package)`
- `packages`: `{dist.name: dist.version for dist in importlib.metadata.distributions()}`
- `cuda_version`: `torch.version.cuda` if torch available, else None
- `os`: `platform.system().lower() + "-" + platform.machine()` → `"linux-x86_64"`
- `captured_at`: `datetime.utcnow().isoformat() + "Z"`

Override: if `override` dict provided, merge it on top of auto-captured values — scientist can set `{"cudaVersion": "12.1-custom"}` without touching anything else.

**AC:**
- `capture()` returns a valid `DepSnapshot` with all required fields populated
- `cuda_version` is `None` in a CPU-only environment (no exception)
- Override merging: `capture(model, {"cudaVersion": "12.1"})` sets only that field
- Unit test: mock `importlib.metadata`, `platform`, `sys`; verify correct extraction
- Coverage ≥ 90% on `_snapshot.py`

---

### Epic 5 — ArtifactModel wrapper [~1 day]

---

#### Story 5.1 — ArtifactModel wrapper + __getattr__ forwarding [M]

**File:** `artifact_mgmt/_artifact_model.py`

```python
class ArtifactModel:
    def __init__(self, model: object, *, model_name: str, version: str,
                 dep_snapshot: DepSnapshot, serializer: Serializer): ...

    @property
    def model(self) -> object: ...          # native model object
    @property
    def model_name(self) -> str: ...
    @property
    def version(self) -> str: ...
    @property
    def dep_snapshot(self) -> DepSnapshot: ...

    def __getattr__(self, name: str) -> Any:
        # forward to self._model for anything not defined on ArtifactModel
        return getattr(self._model, name)

    def __repr__(self) -> str:
        return f"ArtifactModel({self.model_name!r}, version={self.version!r}, ...)"
```

**AC:**
- `artifact.model` returns the native model object
- `artifact.train()` forwards to native model (tested with a mock object)
- `artifact.parameters()` forwards to native model
- `repr` includes model_name and version
- Accessing a non-existent attribute raises `AttributeError` (not silently returning None)
- Coverage ≥ 90% on `_artifact_model.py`

---

#### Story 5.2 — Training abstractions: freeze / unfreeze / fine_tune_params [M]

**File:** `artifact_mgmt/_artifact_model.py` (methods on ArtifactModel)

```python
def freeze(self, n_layers: int) -> None:
    self._serializer.freeze(self._model, n_layers)

def unfreeze(self, n_layers: int) -> None:
    self._serializer.unfreeze(self._model, n_layers)

def fine_tune_params(self) -> list:
    return self._serializer.fine_tune_params(self._model)
```

These delegate to the serializer that was used to load the model — serializer carries the framework-specific knowledge.

**AC:**
- `freeze` / `unfreeze` / `fine_tune_params` delegate to the correct serializer
- Calling `freeze` on an sklearn-backed model raises `NotImplementedError` with clear message
- Unit test with mock serializer verifying delegation
- Coverage ≥ 90% on training abstraction methods in `_artifact_model.py`

---

#### Story 5.3 — Inference abstraction: predict [S]

**File:** `artifact_mgmt/_artifact_model.py`

```python
def predict(self, X: object) -> object:
    return self._serializer.predict(self._model, X)
```

Framework-specific behaviour lives in each serializer (Story 3.x). `ArtifactModel.predict` is just delegation.

**AC:**
- PyTorch: sets `eval()`, wraps in `torch.no_grad()`
- HuggingFace: calls `model(**X)`
- sklearn: calls `model.predict(X)`
- TF/Keras: calls `model.predict(X)`
- Unit test per framework using mock models
- Coverage ≥ 90% on `predict` path in `_artifact_model.py`

---

### Epic 6 — High-level save_model / load_model + cache [~1.5 days]

---

#### Story 6.1 — save_model [L]

**File:** `artifact_mgmt/client.py`

```python
def save_model(self, model: object, model_name: str, *,
               major: int | None = None,
               dep_snapshot: dict | None = None,
               serializer: str | None = None) -> str:
    """Serialize, upload, and confirm a new model version. Returns version string e.g. '2.1'."""
```

Full orchestration:
1. Detect serializer (or use `serializer=` override)
2. `serializer.serialize(model)` → `data: bytes`
3. Compute `checksum_sha256 = base64.b64encode(hashlib.sha256(data).digest()).decode()`
4. `snapshot.capture(model, override=dep_snapshot)` → `DepSnapshot`
5. `idempotency_key = str(uuid.uuid4())`
6. `_create_version(model_name, ...)` → `Version` with `upload_url`
7. `_upload_artifact(upload_url, data, checksum_sha256)`
8. `_confirm_version(model_name, version)` → confirmed `Version`
9. Return `version` string

**AC:**
- Happy path: `save_model(tiny_model, "test")` calls create_version → upload → confirm in order
- Major bump: `save_model(model, "test", major=2)` passes `major=2` to create_version
- Checksum computed and passed correctly (base64 SHA-256)
- Idempotency key is a valid UUID4
- `dep_snapshot` override fields merged correctly
- All three HTTP calls verified in unit test with `responses` mock
- Coverage ≥ 90% on `save_model` path in `client.py`

---

#### Story 6.2 — load_model [L]

**File:** `artifact_mgmt/client.py`

```python
def load_model(self, model_name: str, *,
               version: str | None = None) -> ArtifactModel:
    """Download and deserialize a model version. Returns ArtifactModel wrapper."""
```

Full orchestration:
1. `get_version(model_name, version)` or `get_latest_version(model_name)` → `Version` with `download_url`
2. Check cache: if `self._cache` and hit → return cached `ArtifactModel`
3. `requests.get(download_url)` → `data: bytes` (no SigV4 — presigned GET URL)
4. `SerializerRegistry.detect_from_snapshot(version.dep_snapshot)` → `Serializer`
5. `serializer.deserialize(data)` → native model
6. Wrap in `ArtifactModel(model, model_name=..., version=..., dep_snapshot=..., serializer=...)`
7. Cache if enabled
8. Return `ArtifactModel`

**AC:**
- Happy path: mock HTTP + tiny serializer → returns `ArtifactModel` with correct metadata
- `artifact.version` matches requested version
- `artifact.model_name` matches
- Cache miss → downloads; cache hit → no HTTP call
- `get_latest_version` used when `version=None`
- `VersionNotFoundError` propagates correctly when no READY version exists
- Coverage ≥ 90% on `load_model` path in `client.py`

---

#### Story 6.3 — Local disk cache [M]

**File:** `artifact_mgmt/_cache.py`

```python
class ModelCache:
    def __init__(self, cache_dir: str | Path): ...

    def get(self, model_name: str, version: str) -> ArtifactModel | None: ...
    def put(self, model_name: str, version: str, artifact: ArtifactModel) -> None: ...
    def invalidate(self, model_name: str, version: str) -> None: ...
    def clear(self) -> None: ...
```

Cache layout: `{cache_dir}/{model_name}/{version}/weights` (raw bytes) + `{version}/meta.json` (DepSnapshot + framework name for deserializing on cache hit).

Cache is only engaged when `ArtifactMgmtClient` is constructed with `cache_dir=`. Default is `None` (no cache).

**AC:**
- Cache miss returns `None`; `put` then `get` returns the artifact
- Cached bytes survive a new `ModelCache` instance pointing at the same dir (persists across process restarts)
- `invalidate` removes the cached entry; subsequent `get` returns `None`
- `clear` removes all cached entries
- Unit tests use `tmp_path` pytest fixture — no real disk state between tests
- Coverage ≥ 90% on `_cache.py`

---

### Epic 7 — Packaging, CI, and developer experience [~1 day]

---

#### Story 7.1 — Optional extras in pyproject.toml [S]

**File:** `pyproject.toml`

```toml
[project.optional-dependencies]
torch        = ["torch>=2.0", "pytorch-lightning>=2.0"]
transformers = ["transformers>=4.30", "safetensors>=0.4"]
sklearn      = ["scikit-learn>=1.3", "joblib>=1.3"]
tensorflow   = ["tensorflow>=2.13"]
all          = [...]   # union of all above
dev          = ["pytest>=8.0", "pytest-cov>=5.0", "pytest-mock>=3.12", "responses>=0.25",
                "ruff>=0.4", "mypy>=1.10", "boto3-stubs[s3]>=1.34"]
```

Scientists install only what they need:
```
pip install artifact-mgmt-client[torch]
pip install artifact-mgmt-client[transformers]
pip install artifact-mgmt-client[all]
```

**AC:**
- `pip install -e .[torch]` installs torch + lightning, not transformers/sklearn/tensorflow
- `pip install -e .` (no extras) installs only requests + requests-aws4auth + boto3
- All extras listed in `[all]`

---

#### Story 7.2 — ruff + mypy + CI workflow [S]

**Files:** `pyproject.toml` (tool configs, coverage config), `.github/workflows/ci.yml`

CI steps:
1. `pip install -e ".[dev]"`
2. `ruff check artifact_mgmt tests`
3. `mypy artifact_mgmt`
4. `pytest tests/ -v --tb=short --cov=artifact_mgmt --cov-report=term-missing --cov-fail-under=90`

Coverage config in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
addopts = "--cov=artifact_mgmt --cov-report=term-missing --cov-fail-under=90"

[tool.coverage.run]
omit = ["artifact_mgmt/serializers/_tensorflow.py"]  # omit only if TF not installed in CI

[tool.coverage.report]
fail_under = 90
show_missing = true
```

`dev` extras must include `pytest-cov>=5.0`.

mypy: `strict = true`, `ignore_missing_imports = true` (framework stubs are optional).

**AC:**
- CI fails if overall coverage of `artifact_mgmt/` drops below 90%
- Coverage report uploaded as CI artifact so failures are inspectable
- CI passes on a clean checkout with no framework extras installed (serializer tests mock imports)
- `ruff` and `mypy` clean with no suppressions on core files

---

#### Story 7.3 — prod endpoint + promote_model [M]

**Files:** `artifact_mgmt/client.py`, `artifact_mgmt/CLAUDE.md` (endpoint table)

**Background:** Each stage (alpha, gamma, prod) is a fully isolated silo — separate DDB tables, separate S3 bucket, no automatic cross-stage data sharing. The SDK currently only exposes `alpha` and `gamma` as named stages. `prod` exists and is deployed by the pipeline but its URL was never captured as a named constant, leaving scientists with no way to target prod without manually looking up the CloudFormation stack output. `promote_model` is the convenience primitive that addresses the cross-stage workflow: validate in gamma, promote to prod in one call.

**Implementation notes:**

Add `prod` to `_STAGE_ENDPOINTS` (look up the actual URL from the `ArtifactMgmt-Api-prod` CloudFormation stack output — key `ApiUrl`):

```python
_STAGE_ENDPOINTS: dict[str, str] = {
    "alpha": "https://pi5ywcu3ub.execute-api.us-east-1.amazonaws.com/alpha",
    "gamma": "https://idco76hrk9.execute-api.us-east-1.amazonaws.com/gamma",
    "prod":  "https://<prod-api-id>.execute-api.us-east-1.amazonaws.com/prod",  # fill in from CF output
}
```

Add `promote_model` to `ArtifactMgmtClient`:

```python
def promote_model(
    self,
    model_name: str,
    *,
    version: str | None = None,
    dest: "ArtifactMgmtClient",
    major: int | None = None,
) -> str:
    """
    Load a model version from this client's stage and save it to dest's stage.
    Returns the new version string in the destination stage.

    Typical usage:
        gamma_client.promote_model("fraud-detector", version="2.1", dest=prod_client)
    """
```

Orchestration:
1. `self.get_version(model_name, version)` or `self.get_latest_version(model_name)` → `Version` with `download_url`
2. `requests.get(version.download_url)` → raw bytes (no SigV4 — presigned GET)
3. `dest.save_model_bytes(bytes, model_name, dep_snapshot=version.dep_snapshot, major=major)` — internal method that skips serialization and snapshot capture, using the already-serialized bytes and existing dep_snapshot directly
4. Return the new version string

Add internal `save_model_bytes` to avoid re-serializing bytes that are already in wire format:

```python
def _save_model_bytes(
    self,
    data: bytes,
    model_name: str,
    *,
    dep_snapshot: DepSnapshot,
    major: int | None = None,
) -> str:
    """Skip serialization — used by promote_model with already-serialized bytes."""
```

**Key design decisions:**
- `promote_model` lives on the *source* client (`gamma_client.promote_model(..., dest=prod_client)`) — reads from self, writes to dest. This is natural: the scientist is promoting *from* gamma.
- The promoted version gets a fresh version number in the destination stage (minor bump by default, or `major=` override). It does NOT attempt to preserve the same version string — version numbers are stage-local counters.
- The dep_snapshot from the source version is carried over unchanged — the promoted artifact was built in the same environment.
- `promote_model` does NOT create the model in the destination if it doesn't exist — the scientist must call `dest.create_model(...)` first. This keeps `promote_model` single-purpose and avoids silent model creation in prod.

**AC:**
- `gamma_client.promote_model("fraud-detector", version="2.1", dest=prod_client)` returns a version string (e.g. `"1.0"`)
- `promote_model` without `version=` promotes the latest READY version from the source stage
- The bytes downloaded from source are uploaded unchanged to the destination (no re-serialization)
- The dep_snapshot from the source version is preserved in the destination version
- `promote_model` raises `ModelNotFoundError` if the model doesn't exist in the destination stage (no silent creation)
- `promote_model` raises `VersionNotFoundError` if the requested version doesn't exist or has no READY status in source
- Unit tests mock both source and dest HTTP with `responses` library
- Coverage ≥ 90% on `promote_model` and `_save_model_bytes` paths

---

#### Story 7.4 — Brazil-style dependency pinning [S]

**Files:** `requirements.in`, `requirements.txt`, `requirements-dev.in`, `requirements-dev.txt`

Lock every dependency to an exact hash-pinned version using `pip-tools` so that installs are bit-for-bit reproducible across machines and CI runs. This mirrors the pinning strategy used in the backend service (Gradle lockfiles for Java, `npm ci` for TypeScript).

**Implementation notes:**
- Add `pip-tools` to the `dev` extras in `pyproject.toml`.
- Create `requirements.in` listing only the three base deps (`requests`, `requests-aws4auth`, `boto3`) without version ranges.
- Create `requirements-dev.in` that includes `-r requirements.in` plus all dev extras.
- Run `pip-compile --generate-hashes requirements.in -o requirements.txt` and `pip-compile --generate-hashes requirements-dev.in -o requirements-dev.txt` to produce hash-pinned lockfiles.
- Commit both `.in` source files and both generated `.txt` lockfiles.
- Add a CI step after install that runs `pip-compile --generate-hashes requirements.in -o /tmp/req-check.txt && diff requirements.txt /tmp/req-check.txt` — fails if lockfile is stale.
- Framework extras (torch, transformers, etc.) are intentionally excluded from the lockfile — they are user-controlled optional installs.

**Why this matters:**
- A silent patch release of `requests` or `boto3` cannot break a scientist's environment unexpectedly.
- Every CI run uses the exact dependency graph that was tested, not "latest compatible."
- Onboarding a new machine or container always produces an identical environment.

**AC:**
- `pip install -r requirements.txt` installs base deps at pinned versions with hash verification
- `pip install -r requirements-dev.txt` installs dev environment at pinned versions
- CI fails if `requirements.txt` is out of date relative to `requirements.in`
- Adding a new base dependency without re-running `pip-compile` causes CI to fail
- Framework optional extras are not included in the lockfiles

---

## Sequencing

| Phase | Stories | Days | Notes |
|---|---|---|---|
| 0 | 7.0 | 0 | Write this doc to `docs/IMPLEMENTATION.md` — first commit |
| 1 | 1.1 → 1.4 | 1–2 | Foundation — unblocks everything |
| 2 | 2.1 → 2.4 | 1.5 | REST client — unblocks Epic 6 |
| 3 | 3.1 → 3.6 | 2 | Serializers — unblocks Epic 5 + 6 |
| 4 | 4.1 | 0.5 | Snapshot — unblocks save_model |
| 5 | 5.1 → 5.3 | 1 | ArtifactModel — unblocks load_model |
| 6 | 6.1 → 6.3 | 1.5 | High-level API — the visible surface |
| 7 | 7.1 → 7.4 | 1.5 | Packaging + CI + prod endpoint + promote_model + dependency pinning |

**Total: ~9 working days**

**Coverage enforcement summary:** every story targets ≥ 90% on its own files; Story 7.2 wires `--cov-fail-under=90` as a global CI hard gate. `pytest-cov>=5.0` is required in the `dev` extras.

---

## Verification

End-to-end smoke test against gamma (requires AWS credentials):

```python
from artifact_mgmt import ArtifactMgmtClient
import torch.nn as nn

client = ArtifactMgmtClient(stage="gamma")

# Save
model = nn.Linear(4, 2)
version = client.save_model(model, "sdk-smoke-test")
assert version.startswith("1.")

# Load
artifact = client.load_model("sdk-smoke-test")
assert artifact.version == version
assert isinstance(artifact.model, nn.Module)

# Freeze
artifact.freeze(n_layers=1)
assert all(not p.requires_grad for name, p in artifact.model.named_parameters()
           if "weight" in name)

# Native passthrough
artifact.eval()   # forwarded via __getattr__ — no AttributeError

# Cleanup
client.delete_model("sdk-smoke-test")
```

Unit tests run offline with `responses` mocks — no AWS credentials required.
