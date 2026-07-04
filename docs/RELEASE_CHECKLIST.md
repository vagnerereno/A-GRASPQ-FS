# Release Checklist

Use this checklist before publishing a new release.

## Local validation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python examples/quickstart.py
python examples/pipeline_example.py
agraspqfs --version
python -m build
twine check dist/*
```

## Metadata

- [ ] Update `src/agraspqfs/_version.py`.
- [ ] Update `pyproject.toml` version.
- [ ] Update `CHANGELOG.md`.
- [ ] Update `CITATION.cff` release date/version.
- [ ] Update Zenodo metadata if needed.
- [ ] Confirm `LICENSE` and citation instructions are still correct.

## GitHub

```bash
git tag -a v1.0.0 -m "A-GRASPQ-FS v1.0.0"
git push origin v1.0.0
```

Create a GitHub Release from the tag. The `publish.yml` workflow is prepared for PyPI Trusted Publishing.

## Zenodo

After the GitHub release is created, archive the release on Zenodo to obtain a software DOI. Then update `CITATION.cff` and `docs/CITATION.md` with the DOI.
