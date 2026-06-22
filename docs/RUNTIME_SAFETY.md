# ENN-PyTorch Runtime and Artifact Safety

Use this document before running code that can allocate accelerator memory, write files, spawn workers, download data, train, export, checkpoint, or create artifacts.

## Default execution rules

- Use small synthetic tensors as default test inputs.
- Do not overwrite real data or artifacts in the repository.
- Run staging tests only in temporary directories.
- Use `overwrite=True` only for temporary paths created by the agent for the current task.
- Do not use full notebook execution as baseline validation.
- Do not use long-running training as baseline validation.
- Do not run CUDA code without checking CUDA availability first.
- Clean up distributed workers and child processes after tests.
- Do not automatically download external data or models.
- Do not assume TensorRT, CoreML, TensorFlow, ExecuTorch, NVIDIA Transformer Engine, Intel Extension for PyTorch, pandas, polars, Spark, safetensors, or other optional backends are installed.
- Do not add checkpoints, exports, predictions, memmaps, or temporary artifacts to Git.
- Do not use `sudo`, administrator elevation, or destructive root-level operations for repository tasks.
- Root-based container workspaces are allowed for safe repository-local checks such as `git diff --check`, `python -m py_compile`, and `python -m pytest`, as long as the task does not change system paths, install global packages, modify ownership/permissions, or touch data outside the workspace.
- Do not record secrets, tokens, API keys, or real credentials.
- Do not create `.env` files.

## Artifact handling

Keep generated outputs outside tracked repository paths when possible. If a test must write files, use a temporary directory and remove it when finished. Before final reporting, check `git status --short` for accidental artifacts.
