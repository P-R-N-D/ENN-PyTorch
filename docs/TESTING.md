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
| `enn_torch_dev/data/**` | `python -m pytest enn_torch_dev/debug/data -q` | `enn_torch_dev/debug/data` |
| `enn_torch_dev/executor/**` | `python -m pytest enn_torch_dev/debug/executor -q` | `enn_torch_dev/debug/executor` |
| `enn_torch_dev/nn/**` | `python -m pytest enn_torch_dev/debug/nn -q` | `enn_torch_dev/debug/nn` |
| `enn_torch_dev/runtime/**` | `python -m pytest enn_torch_dev/debug/runtime -q` | `enn_torch_dev/debug/runtime` |
| Public executor exports | `python -m pytest enn_torch_dev/debug/executor/test_public_api_exports.py -q` | `enn_torch_dev/debug/executor/test_public_api_exports.py` |
| Cross-area changes | `python -m pytest enn_torch_dev/debug -q` | `enn_torch_dev/debug` |
| Documentation-only changes | `git diff --check` and targeted path/link review | Repository docs and changed files |
| CPU-only environments | Run relevant pytest command on CPU; do not force CUDA paths | Debug tests are repository-local pytest paths |
| CUDA-available environments | Check CUDA availability before running CUDA-specific behavior; report CUDA-specific skips separately | No mandatory full-GPU validation command is configured here |
| Optional dependency checks | Import or test only the optional backend directly affected by the change, and report missing optional packages without installing them unless explicitly requested | Optional extras are declared in `pyproject.toml` |

## Baseline debug commands

These commands correspond to existing paths in the repository:

```bash
python -m pytest enn_torch_dev/debug/data -q
python -m pytest enn_torch_dev/debug/executor -q
python -m pytest enn_torch_dev/debug/nn -q
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
