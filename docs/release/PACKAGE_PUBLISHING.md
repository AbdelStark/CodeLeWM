# Package Publishing Gate

This gate covers Python package artifacts only. It does not publish model,
dataset, checkpoint, or benchmark artifacts, and it does not change Hugging Face
repository visibility.

## Status

Package publication is manually gated until the final release freeze and public
visibility review. CI builds and installs wheel/sdist artifacts on every pull
request, but no workflow uploads to TestPyPI or PyPI. A maintainer must run the
upload commands from a clean release commit only after #126 is complete and the
current claim boundary allows publication.

## Build And Check

Run from a clean checkout:

```bash
rm -rf dist .artifacts/package-gate
uv sync --group dev --group release
uv build --sdist --wheel --out-dir .artifacts/package-gate/dist --clear
uv run twine check .artifacts/package-gate/dist/*
```

The build must produce exactly one source distribution and one wheel. The wheel
must include:

- `codelewm/py.typed`;
- `codelewm = codelewm.harness.cli:main` in entry points;
- `README.md` as the rendered long description;
- `LICENSE` as the license file;
- `License-Expression: MIT`;
- supported Python classifiers and `Requires-Python: >=3.10`.

## Clean Install Smoke

Install the built wheel in a fresh virtual environment and run the console
script:

```bash
uv venv .artifacts/package-gate/venv --python 3.13
uv pip install \
  --python .artifacts/package-gate/venv/bin/python \
  .artifacts/package-gate/dist/codelewm-*.whl
.artifacts/package-gate/venv/bin/codelewm --help
.artifacts/package-gate/venv/bin/python -c "import codelewm; print(codelewm.__version__)"
```

The installed package must expose the same CLI surface tested by
`tests/api/test_cli_contract.py`.

## Manual Publication

Do not upload automatically from pull-request CI. After #126 signs off the
release and the public visibility gate is satisfied, publish with
repository-scoped credentials in a clean shell:

```bash
uv run twine upload --repository testpypi .artifacts/package-gate/dist/*
```

After TestPyPI install smoke passes, publish to PyPI:

```bash
uv run twine upload .artifacts/package-gate/dist/*
```

Token values must come from the shell or a local secret manager. Do not write
tokens to `.env`, logs, PR bodies, release notes, or benchmark artifacts.

## Release Preconditions

- CI package-build job passes on the release commit.
- `uv lock --check` passes.
- `docs/release/DEPENDENCY_PROVENANCE.md` audit and provenance gates pass.
- Full test suite passes.
- Manifest and secret gates pass.
- Release checklist references the exact wheel and sdist filenames.
- The benchmark report states the final model-quality claim boundary.
