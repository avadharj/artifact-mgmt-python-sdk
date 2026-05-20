# CLAUDE.md

Working notes for Claude Code on the **artifact-mgmt Python SDK**. Read this first every session.

## What this repo is

A Python client library (`artifact-mgmt-client`) that wraps the Artifact Management Service REST API. Scientists use it to save and load trained ML models with one call, without writing SigV4 boilerplate, serialization logic, environment snapshots, or S3 uploads themselves.

**Authoritative spec:** `docs/IMPLEMENTATION.md`. Every epic and story is broken down there with file paths, method signatures, and acceptance criteria. When this file and `IMPLEMENTATION.md` disagree, `IMPLEMENTATION.md` wins on mechanics; `CLAUDE.md` wins on workflow.

**Live API contract:** See API gotchas section below — these were discovered by live-testing gamma and are non-negotiable.

## Tech stack (locked)

- **Language:** Python 3.11+
- **HTTP:** `requests` + `requests-aws4auth` (SigV4)
- **AWS credentials:** boto3 default chain (env vars → `~/.aws/credentials` → IAM role)
- **Serialization backends:** PyTorch/Lightning, HuggingFace Transformers, sklearn/joblib, TF/Keras, pickle fallback
- **Testing:** pytest + `responses` library (no real network calls in unit tests)
- **Linting/typing:** ruff + mypy (strict)
- **Coverage:** pytest-cov, gate ≥ 90% (`--cov-fail-under=90`)
- **Packaging:** pyproject.toml, optional extras per framework

Do not add new dependencies to the base install (`requests`, `requests-aws4auth`, `boto3` only). Framework packages belong in `[project.optional-dependencies]`.

## Repo layout

```
artifact_mgmt/
├── __init__.py              # public exports
├── client.py                # ArtifactMgmtClient
├── _artifact_model.py       # ArtifactModel wrapper + __getattr__ forwarding
├── _types.py                # dataclasses: Model, Version, DepSnapshot, FrameworkInfo
├── _exceptions.py           # exception hierarchy
├── _http.py                 # SigV4 HTTP layer
├── _pagination.py           # PageIterator
├── _snapshot.py             # dep_snapshot auto-capture
├── _cache.py                # local disk cache
└── serializers/
    ├── __init__.py          # SerializerRegistry
    ├── _base.py             # Serializer ABC
    ├── _torch.py            # PyTorch / Lightning
    ├── _huggingface.py      # HuggingFace Transformers
    ├── _sklearn.py          # sklearn / XGBoost / LightGBM via joblib
    ├── _tensorflow.py       # TF/Keras
    └── _pickle.py           # generic pickle fallback

tests/
├── conftest.py
├── test_client_models.py
├── test_client_versions.py
├── test_artifact_model.py
├── test_serializers.py
├── test_snapshot.py
├── test_pagination.py
└── test_cache.py
```

## Commands

| Task | Command | Notes |
|---|---|---|
| Install (dev) | `pip install -e ".[dev]"` | Installs base deps + dev extras |
| Install with framework | `pip install -e ".[dev,torch]"` | Add framework extras as needed |
| Run tests | `pytest tests/ -v` | All tests must pass, no skips |
| Run tests + coverage | `pytest tests/ -v --cov=artifact_mgmt --cov-report=term-missing` | Gate is ≥ 90% |
| Lint | `ruff check artifact_mgmt tests` | Must exit 0 |
| Type check | `mypy artifact_mgmt` | Must exit 0 (strict mode) |
| All checks | `ruff check artifact_mgmt tests && mypy artifact_mgmt && pytest tests/ -v` | Run before every commit |

**Always run before committing:**
```
ruff check artifact_mgmt tests && mypy artifact_mgmt && pytest tests/ --cov=artifact_mgmt --cov-report=term-missing --cov-fail-under=90
```

## Workflow rules

**One story per session.** `docs/IMPLEMENTATION.md` is sized so each story is a focused unit. Don't pull a whole epic — implement one story, get its AC green, commit it.

**Acceptance criteria are not suggestions.** Each story lists AC items including coverage thresholds. If you can't satisfy one, surface it explicitly rather than declaring done.

**Always invoke the sdk-reviewer subagent before declaring a story done.** This is non-negotiable. The reviewer runs in a fresh context and catches things the writer cannot see.

**No real network calls in unit tests.** All HTTP is mocked via the `responses` library. Tests must run offline with no AWS credentials.

**Lazy imports in all serializer files.** Never add top-level `import torch`, `import transformers`, `import sklearn`, or `import tensorflow` in any file under `serializers/`. Imports go inside method bodies. `import artifact_mgmt` must succeed with only `requests`, `requests-aws4auth`, and `boto3` installed.

**Base deps stay minimal.** `pyproject.toml` base `dependencies` = `requests`, `requests-aws4auth`, `boto3` only. Framework packages go in optional extras.

## API gotchas (automatic FAIL in reviewer if violated)

These were discovered by live-testing gamma. They differ from what you'd guess from the Smithy IDL.

- **`checksumSha256`** must be base64-encoded SHA-256 digest: `base64.b64encode(hashlib.sha256(data).digest()).decode()`. Never hex (`hexdigest()`).
- **`ConfirmVersion`** is `PUT /models/{modelName}/versions/{version}/confirm`, not POST. Body must be `{}` — null body → 400.
- **S3 presigned PUT** requires `Content-Type: application/octet-stream` header. Omitting it → 403 SignatureDoesNotMatch.
- **`ListModels` response** key is `"items"`, not `"models"`.
- **`ListVersions` response** key is `"versions"`, not `"items"`.
- **Version path param** is always a dotted string `"1.0"` — never split into major/minor path segments.
- **Idempotency replay** returns HTTP 200 (not 201) with a fresh `uploadUrl`. Must be handled identically to 201.
- **`CreateModel`** — do NOT send `framework`, `frameworkVersion`, `trainingMetadata`, `depSnapshot`. Unknown fields → 400.

## Subtleties to remember

**`ArtifactModel.__getattr__` must raise `AttributeError` for missing attrs.** `getattr(self._model, name)` does this naturally — but don't add a try/except that swallows `AttributeError` and returns `None`. Tests verify this behavior.

**`load_model` uses `SerializerRegistry.detect_from_snapshot`**, not type detection. The native model isn't known until after deserialization — we pick the serializer from `dep_snapshot.framework.name` in the `Version` response.

**PyTorch serializer saves the full model object** (`torch.save(model, buf)`), not just `state_dict`. This is required for round-trip deserialization without the user providing the class definition.

**HuggingFace serializer tarballs the `save_pretrained` output directory** — includes `config.json`, weights, and tokenizer files if present. Deserialize by untarring to a tempdir and calling `AutoModel.from_pretrained(tmpdir)`.

**sklearn `freeze/unfreeze/fine_tune_params` raise `NotImplementedError`** — sklearn estimators don't support layer freezing. This is expected; tests verify the message is clear.

**Cache is opt-in** via `ArtifactMgmtClient(cache_dir=...)` or `ARTIFACT_MGMT_CACHE_DIR` env var. Default is `None` (no cache). Never engage the cache unless the client was constructed with a `cache_dir`.

**Stage configuration** — `ArtifactMgmtClient(stage="gamma")` resolves the endpoint internally. Falls back to `ARTIFACT_MGMT_STAGE` env var. Supported stages: `alpha`, `gamma`, `prod`.

**Endpoint URLs:**
- Alpha: `https://pi5ywcu3ub.execute-api.us-east-1.amazonaws.com/alpha`
- Gamma: `https://idco76hrk9.execute-api.us-east-1.amazonaws.com/gamma`
- Prod: `https://afwtpvnxe7.execute-api.us-east-1.amazonaws.com/prod`

## Things to ask before doing

If a request matches any of these, stop and ask the user:

- "Add a new AWS service or dependency to base deps"
- "Add top-level framework imports to a serializer file"
- "Implement multiple stories in one session"
- "Skip the reviewer step for a story"
- "Change the public API surface" (ArtifactMgmtClient signatures, ArtifactModel properties)
- "Modify the live endpoint URLs"
- "Change the base authentication mechanism away from SigV4/boto3"

## Style

**Python:** Dataclasses for value objects. Type annotations on all public functions and methods. No bare `except:` — only specific exception types. Optional for nullable returns where it improves clarity.

**Testing:** Given-When-Then structure for unit tests, reflected in test names (e.g., `test_get_model_raises_not_found_when_model_does_not_exist`). Use `responses` to mock all HTTP. Use `pytest.fixture` and `tmp_path` for test isolation.

**Imports:** Absolute imports only. No wildcard imports. Framework imports inside method bodies in serializer files.

**Commits:** Conventional commits format (`feat:`, `fix:`, `chore:`, `docs:`). Reference the story (`feat(epic-1): implement story 1.3 SigV4 HTTP client`).

## Two-agent workflow

**Writer agent** (default, the one you're talking to):
- Implements one story at a time from `docs/IMPLEMENTATION.md`
- Writes code and tests
- Runs lint, type check, and tests locally before invoking the reviewer
- **Must invoke the sdk-reviewer subagent before declaring a story complete**

**sdk-reviewer agent** (subagent, fresh context):
- Defined in `.claude/agents/sdk-reviewer.md`
- Has no memory of the conversation that produced the code
- Checks: AC verification with actual coverage numbers, API gotcha violations, lazy import discipline, dependency integrity, scope discipline, ruff+mypy clean, GWT test structure
- Returns: PASS, FAIL (with blocking issues), or NEEDS-CLARIFICATION

**Workflow per story:**
1. Writer implements the story.
2. Writer runs `ruff check artifact_mgmt tests && mypy artifact_mgmt && pytest tests/ -v --cov=artifact_mgmt --cov-report=term-missing`.
3. Writer invokes the sdk-reviewer subagent: "Review the changes for story N.M against `docs/IMPLEMENTATION.md`."
4. Reviewer returns PASS / FAIL / NEEDS-CLARIFICATION.
5. On FAIL: writer addresses each blocking issue, re-invokes reviewer. Loop until PASS.
6. On PASS: writer commits with the conventional commit message.

**The writer must not skip the reviewer step**, even on stories that feel obviously correct.

## What success looks like per session

By the end of a session you should have:
1. One story's code committed with a conventional commit message.
2. All acceptance criteria verified — actually run, not "should be green."
3. `ruff` and `mypy` clean with no suppressions on core files.
4. Coverage ≥ 90% on files introduced by the story.
5. The sdk-reviewer subagent has returned PASS on the changes.

If you can't get there, surface the blocker rather than declaring partial completion.
 