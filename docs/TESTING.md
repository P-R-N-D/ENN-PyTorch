# ENN-PyTorch Testing Guidance

Only document and run commands that match real repository paths. The repository currently provides pytest debug tests under `enn_torch_dev/debug`. Do not invent formatter, linter, type-checker, or CI requirements.

Report outcomes distinctly:

- tests run and passed;
- tests run and failed;
- tests not run because dependencies are missing;
- tests skipped because CUDA or another optional backend is unavailable.

## Change-scope command table

| Change scope | Primary command | Existing path checked |
|---|---|---|
| `enn_torch/__init__.py` | Targeted import/API smoke check for stable top-level exports, for example `python - <<'PY'` with imports from `enn_torch`; report missing dependencies instead of installing them | No dedicated stable-package pytest suite is currently identified |
| `enn_torch/core/**` | Targeted import/API smoke check for changed stable core modules; run related debug tests only when the change touches shared behavior covered there | No dedicated stable-package pytest suite is currently identified |
| `enn_torch/data/**` | Targeted import/API smoke check for changed stable data modules; run related debug tests only when the change touches shared behavior covered there | No dedicated stable-package pytest suite is currently identified |
| `enn_torch/nn/**` | Targeted import/API smoke check for changed stable nn modules; run related debug tests only when the change touches shared behavior covered there | No dedicated stable-package pytest suite is currently identified |
| `enn_torch/runtime/**` | Targeted import/API smoke check for changed stable runtime modules; run related debug tests only when the change touches shared behavior covered there | No dedicated stable-package pytest suite is currently identified |
| `enn_torch_dev/data/**` | `python -m pytest enn_torch_dev/debug/data -q` | `enn_torch_dev/debug/data` |
| `enn_torch_dev/executor/**` | `python -m pytest enn_torch_dev/debug/executor -q` | `enn_torch_dev/debug/executor` |
| `enn_torch_dev/nn/**` | `python -m pytest enn_torch_dev/debug/nn -q` | `enn_torch_dev/debug/nn` |
| `enn_torch_dev/runtime/**` | `python -m pytest enn_torch_dev/debug/runtime -q` | `enn_torch_dev/debug/runtime` |
| Runtime pass source factory | `python -m pytest enn_torch_dev/debug/runtime/test_runtime_source_factory.py -q` | `enn_torch_dev/debug/runtime/test_runtime_source_factory.py` |
| Development runtime end-to-end integration | `python -m pytest enn_torch_dev/debug/runtime/test_runtime_integration.py -q` | `enn_torch_dev/debug/runtime/test_runtime_integration.py` |
| Public executor exports | `python -m pytest enn_torch_dev/debug/executor/test_public_api_exports.py -q` | `enn_torch_dev/debug/executor/test_public_api_exports.py` |
| Cross-area changes | `python -m pytest enn_torch_dev/debug -q` | `enn_torch_dev/debug` |
| Documentation-only changes | `git diff --check` and targeted path/link review | Repository docs and changed files |
| CPU-only environments | Run relevant pytest command on CPU; do not force CUDA paths | Debug tests are repository-local pytest paths |
| CUDA-available environments | Check CUDA availability before running CUDA-specific behavior; report CUDA-specific skips separately | No mandatory full-GPU validation command is configured here |
| Optional dependency checks | Import or test only the optional backend directly affected by the change, and report missing optional packages without installing them unless explicitly requested | Optional extras are declared in `pyproject.toml` |

## Stable package checks

A dedicated pytest suite for the stable `enn_torch/**` package is not currently identified in the repository. For changes under `enn_torch/__init__.py`, `enn_torch/core`, `enn_torch/data`, `enn_torch/nn`, or `enn_torch/runtime`, run a targeted import/API smoke check that imports the changed public module or helper. If the smoke check cannot run because required dependencies are missing, do not install packages unless explicitly requested; report the missing dependency clearly.

When a stable-package change overlaps behavior covered by existing development debug tests, also run the relevant `enn_torch_dev/debug` pytest command from the table above. Do not invent CI, lint, format, or type-check commands.

## Baseline debug commands

These commands correspond to existing paths in the repository:

```bash
python -m pytest enn_torch_dev/debug/data -q
python -m pytest enn_torch_dev/debug/executor -q
python -m pytest enn_torch_dev/debug/nn -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_source_factory.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_integration.py -q
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug/executor/test_public_api_exports.py -q
python -m pytest enn_torch_dev/debug -q
```

## Documentation-only baseline

For documentation-only changes, run at least:

```bash
git diff --check
git status --short
```

Also verify that changed files are documentation-focused, production Python code is unchanged, dependencies and lockfiles are unchanged, README and LICENSE bodies are unchanged, forbidden AI-document paths were not created, package-internal Markdown instruction files were not created, and repository-relative links point to real files.
