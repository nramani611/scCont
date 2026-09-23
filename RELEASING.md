# Releasing `sccont` to PyPI

## One-time setup (trusted publishing, no tokens in the repo)

1. Create an account at <https://pypi.org> (and optionally <https://test.pypi.org>).
2. On PyPI go to **Your account → Publishing → Add a new pending publisher** and enter:
   - PyPI project name: `sccont`
   - Owner: `nramani611`
   - Repository: `scCont`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
3. In the GitHub repo go to **Settings → Environments → New environment**, name it `pypi`.
   (Optionally add yourself as a required reviewer so a release needs a manual approval.)

## Cutting a release

1. Bump `version` in `pyproject.toml` (e.g. `0.1.0` → `0.1.1`) and commit.
2. Run the tests locally: `pip install -e ".[dev]" && pytest`.
3. Tag and push: `git tag v0.1.1 && git push origin main --tags`.
4. On GitHub, **Releases → Draft a new release**, pick the tag, write notes, **Publish release**.
   The `Publish to PyPI` workflow builds the sdist + wheel and uploads them.
5. Check <https://pypi.org/project/sccont/> and `pip install sccont==0.1.1` in a fresh venv.

## Manual fallback

```bash
pip install build twine
rm -rf dist/ && python -m build
twine check dist/*
twine upload dist/*            # prompts for a PyPI API token (username: __token__)
```

## Notes

- The wheel only contains `src/sccont/`; the dataset folders (`A549_EGF/`, `MCF10A_TGFB1/`, ...)
  and the notebook are never packaged.
- `requirements.txt` is the frozen environment used for the paper and is kept for reproducibility
  only; the package's runtime dependencies live in `pyproject.toml`.
- `pip install sccont` pulls a CPU build of PyTorch on most platforms. For GPU training install
  torch first from <https://pytorch.org/get-started/locally/> and then `pip install sccont`.
