# Root-level cleanup guide

The root directory still carries many upstream llama.cpp artifacts and paper-supporting assets. To shrink or simplify the repository, use the lists below.

## Core pieces to keep

- **Build + runtime:** `CMakeLists.txt`, `cmake/`, `common/`, `ggml/`, `include/`, `src/`, `scripts/`, `gguf-py/`. Keep `tests/` for validation/CI even if not required to run inference binaries.
- **Conversion utilities:** keep `convert_hf_to_gguf.py`, `convert_hf_to_gguf_update.py`, `convert_hf_to_gguf_vlut.py`, `convert_llama_ggml_to_gguf.py`, and `convert_lora_to_gguf.py` (they rely on `requirements/` and `requirements.txt`).

Removing any of the above will break builds, conversion, or tests.

## Removable / optional at the root

Use your own needs to decide; deleting these will not stop the core C++ build or CLI inference.

- **Sample assets:** `models/` (sample vocabs), `data/`, `media/` (figures).
- **Evaluation and demos:** `evaluation/` guides and scripts, `examples/`, `pocs/`, `prompts/`, `grammars/`, `demo.py`, `demo2.py`.
- **Alternate tooling & packaging:**
  - Swift: `Package.swift`, `Sources/`.
  - Python/Poetry: `poetry.lock`, `pyproject.toml`.
  - Nix: `flake.nix`, `flake.lock`.
  - Dev tooling configs: `.pre-commit-config.yaml`, `.clang-format`, `.clang-tidy`, `.flake8`, `.ecrc`, `pyrightconfig.json`, `mypy.ini`.
- **Repo meta/CI:** `ci/` (GitHub workflows), plus `CODEOWNERS`, `AUTHORS`, `CONTRIBUTING.md`, `SECURITY.md` (governance and contributor guidance). Remove only if you are intentionally stripping repo metadata and automation.

Removing the sample assets alone frees roughly ~55 MB (`models/` ~47 MB, `examples/` ~6 MB, misc data/media/eval files under 1 MB each). If you only need the minimal binary/tooling, keeping the core set above and removing the optional items will leave a compact root layout.
