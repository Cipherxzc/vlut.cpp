# Root-level cleanup guide

The root directory still carries many upstream llama.cpp artifacts and paper-supporting assets. To shrink or simplify the checkout, use the lists below.

## Core pieces to keep

- **Build + runtime:** `CMakeLists.txt`, `cmake/`, `common/`, `ggml/`, `include/`, `src/`, `scripts/`, `gguf-py/`, `tests/`.
- **Conversion utilities:** the `convert_*.py` scripts rely on `requirements/` and `requirements.txt`.

Removing any of the above will break builds, conversion, or tests.

## Removable / optional at the root

Use your own needs to decide; deleting these will not stop the core C++ build or CLI inference.

- **Large artifacts & samples:** `models/` (sample vocabs), `data/`, `media/` (figures), `evaluation/` guides and scripts, `examples/`, `pocs/`, `prompts/`, `grammars/`, `demo.py` / `demo2.py`.
- **Alternate tooling & packaging:** `Package.swift` + `Sources/` (Swift SPM), `poetry.lock` / `pyproject.toml` (Poetry), `flake.nix` / `flake.lock` (Nix), `.pre-commit-config.yaml`, `.clang-format`, `.clang-tidy`, `.flake8`, `.ecrc`, `pyrightconfig.json`, `mypy.ini`.
- **Repo meta/CI:** `ci/`, `CODEOWNERS`, `AUTHORS`, `CONTRIBUTING.md`, `SECURITY.md` — safe to drop if you do not need GitHub workflows or contributor docs.

If you only need the minimal binary/tooling, keeping the core set above and removing the optional items will leave a compact root layout.
