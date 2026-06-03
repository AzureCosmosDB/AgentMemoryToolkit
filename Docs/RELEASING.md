# Release process

Maintainer runbook for cutting a new release of `azure-cosmos-agent-memory`.

## Versioning

This project uses [PEP 440](https://peps.python.org/pep-0440/) versioning:

| Type | Example | Notes |
|------|---------|-------|
| Beta | `0.1.0b1`, `0.1.0b2` | Public preview; breaking changes allowed |
| Release candidate | `0.1.0rc1` | Stabilizing; no breaking changes from this point |
| Final | `0.1.0` | GA; semver rules apply afterwards |
| Patch | `0.1.1` | Bug fixes only |
| Minor | `0.2.0` | Backwards-compatible additions |
| Major | `1.0.0` | Allowed to break; document the migration path |

## Pre-flight checklist

Before cutting a release:

1. **CI green on `main`** — every workflow in `.github/workflows/` passes.
2. **Unit tests pass locally** in a fresh venv:
   ```bash
   python -m venv /tmp/release-venv
   source /tmp/release-venv/bin/activate
   pip install -e ".[dev]"
   pytest tests/unit -q
   ```
3. **Integration tests pass** against a live environment (`azd up`
   environment is fine):
   ```bash
   AGENT_MEMORY_RUN_INTEGRATION=true pytest tests/integration -q
   ```
4. **Samples and notebooks work** — every script under `Samples/` runs to
   completion against the live environment.
5. **No uncommitted local changes** — `git status` clean.

## Cutting a release

1. **Bump the version** in `pyproject.toml`:
2. **Update `CHANGELOG.md`** — add a new section with the version, the
   date, and a summary of changes. Move entries from the unreleased
   section if you keep one.
3. **Bump the Function app's SDK pin** in `function_app/requirements.txt`
   to match — `azure-cosmos-agent-memory==<NEW_VERSION>`. The FA installs
   the SDK from PyPI, so the pin must move in lockstep with the SDK
   release. (If the release workflow fails after merge, the FA will be
   pinned to a non-existent version until you cut a follow-up patch —
   coordinate the merge + release-workflow run together.)
4. **Open a PR** with the version bump + CHANGELOG + updated FA pin.
   Suggested title: `Release v<NEW_VERSION>`. Get it reviewed and merged
   to `main`.
5. **Trigger the release workflow**:
   - Navigate to **Actions → release → Run workflow**
   - Pick the `main` branch
   - Click **Run workflow**
6. **The workflow does the rest** — see `.github/workflows/release.yml`:
   - Builds `dist/*.whl` + `dist/*.tar.gz`
   - Asserts no namespace `__init__.py` shadows are in the wheel
   - Publishes to **TestPyPI** via trusted publishing
   - Installs from TestPyPI into a clean env and runs the unit suite
   - Publishes to **PyPI** via trusted publishing
   - Creates a **GitHub Release** with tag `v<NEW_VERSION>` and the
     wheel + sdist attached
7. **Verify** on <https://pypi.org/project/azure-cosmos-agent-memory/>
   that the new version landed and the README + classifiers render.

## Namespace package note

`azure-cosmos-agent-memory` installs files under `azure/cosmos/agent_memory/`.
It MUST NOT ship `azure/__init__.py` or `azure/cosmos/__init__.py` — those
are owned by the `azure-cosmos` package. The wheel build is configured
(`[tool.setuptools.packages.find]` with `include = ["azure.cosmos.agent_memory*"]`
and `namespaces = true`) so that only the `agent_memory` subtree is packaged.
After every release, sanity-check with:

```bash
python -m zipfile -l dist/azure_cosmos_agent_memory-*.whl | grep '__init__.py' | head
```

You should see `azure/cosmos/agent_memory/__init__.py` and its descendants,
but never the standalone `azure/__init__.py` or `azure/cosmos/__init__.py`.

## Yanking a bad release

If a release ships a regression:

1. **Yank** the version on PyPI (does not delete; marks installs as a warning):
   ```bash
   twine yank azure-cosmos-agent-memory --version <BAD_VERSION>
   ```
2. **Cut a new patch release** with the fix and updated CHANGELOG.
3. Add a deprecation note to the GitHub Release pointing at the
   replacement version.

Never delete a PyPI release — yank it and ship a fixed version instead.
