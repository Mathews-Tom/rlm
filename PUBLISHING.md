# Publishing to PyPI

This guide covers the process of building and publishing the `rlm` package to PyPI.

## Prerequisites

1. **PyPI Account**: Create accounts on both:
   - [TestPyPI](https://test.pypi.org/account/register/) (for testing)
   - [PyPI](https://pypi.org/account/register/) (for production)

2. **API Tokens**: Generate API tokens for both:
   - TestPyPI: https://test.pypi.org/manage/account/token/
   - PyPI: https://pypi.org/manage/account/token/

3. **Configure credentials** in `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR-PRODUCTION-TOKEN

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR-TEST-TOKEN
```

## Build Process

### 1. Pre-build Checks

Run quality checks before building:

```bash
# Type checking
uv run mypy src/

# Linting
uv run ruff check .

# Tests
uv run pytest

# Coverage
uv run pytest --cov=src --cov-fail-under=80
```

### 2. Clean Previous Builds

```bash
# Remove old build artifacts
rm -rf dist/ build/ src/*.egg-info
```

### 3. Build the Package

```bash
# Install build tools
uv pip install build twine

# Build both wheel and source distribution
uv run python -m build

# Verify build artifacts
ls -lh dist/
```

Expected output:

```
dist/
├── rlm-0.1.0-py3-none-any.whl
└── rlm-0.1.0.tar.gz
```

### 4. Verify Package Contents

```bash
# Check wheel contents
unzip -l dist/rlm-0.1.0-py3-none-any.whl

# Check tarball contents
tar -tzf dist/rlm-0.1.0.tar.gz

# Verify package metadata
uv run twine check dist/*
```

## Publishing

### Test on TestPyPI First

**Always test on TestPyPI before publishing to production PyPI:**

```bash
# Upload to TestPyPI
uv run twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --no-deps rlm

# Test the package
python -c "from rlm import RecursiveEngine; print('Success!')"
```

### Publish to PyPI

**Once verified on TestPyPI:**

```bash
# Upload to production PyPI
uv run twine upload dist/*

# Verify on PyPI
open https://pypi.org/project/rlm/

# Test installation
pip install rlm
```

## Version Management

### Semantic Versioning

Follow [SemVer](https://semver.org/):

- **MAJOR** (1.0.0): Breaking changes
- **MINOR** (0.1.0): New features (backward compatible)
- **PATCH** (0.1.1): Bug fixes (backward compatible)

### Update Version

Edit `pyproject.toml`:

```toml
[project]
version = "0.2.0"  # Update version
```

### Tag Release

```bash
# Create git tag
git tag -a v0.2.0 -m "Release version 0.2.0"

# Push tag to GitHub
git push origin v0.2.0

# Create GitHub release
gh release create v0.2.0 --title "v0.2.0" --notes "Release notes here"
```

## Automated Publishing with GitHub Actions

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    permissions:
      id-token: write  # Required for trusted publishing

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install uv
        run: pip install uv

      - name: Install dependencies
        run: uv sync

      - name: Run tests
        run: uv run pytest

      - name: Build package
        run: uv run python -m build

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
```

## Post-Publication

### 1. Verify Installation

```bash
# Install from PyPI
pip install rlm

# Test basic functionality
python -c "from rlm import RecursiveEngine; print('Installed successfully')"
```

### 2. Update Documentation

- Update README.md with installation instructions
- Update CHANGELOG.md with release notes
- Create GitHub release with detailed notes

### 3. Announce Release

- GitHub Discussions
- Social media
- Relevant communities

## Troubleshooting

### Build Fails

```bash
# Check pyproject.toml syntax
uv run python -m build --help

# Verify package structure
tree src/
```

### Upload Fails

```bash
# Check credentials
cat ~/.pypirc

# Verify token permissions
# Tokens must have "upload" scope

# Check if package name is taken
curl https://pypi.org/pypi/rlm/json
```

### Package Not Found After Upload

- Wait 1-5 minutes for PyPI indexing
- Check package page: https://pypi.org/project/rlm/
- Clear pip cache: `pip cache purge`

## Checklist

Before publishing:

- [ ] All tests pass (`uv run pytest`)
- [ ] Type checking clean (`uv run mypy src/`)
- [ ] Linting clean (`uv run ruff check .`)
- [ ] Version updated in `pyproject.toml`
- [ ] CHANGELOG.md updated
- [ ] README.md reflects current features
- [ ] Git tag created (`git tag v0.1.0`)
- [ ] Tested on TestPyPI
- [ ] Package builds successfully (`uv run python -m build`)
- [ ] Metadata verified (`uv run twine check dist/*`)

After publishing:

- [ ] Installation verified (`pip install rlm`)
- [ ] GitHub release created
- [ ] Documentation updated
- [ ] Announcement made

## Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [Semantic Versioning](https://semver.org/)
- [PyPI Help](https://pypi.org/help/)
- [TestPyPI](https://test.pypi.org/)
- [Twine Documentation](https://twine.readthedocs.io/)

---

**Package Name:** `rlm`
**PyPI URL:** https://pypi.org/project/rlm/ (after first publish)
**Installation:** `pip install rlm`
