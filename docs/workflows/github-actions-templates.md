# GitHub Actions Workflow Templates

This document provides complete workflow templates for the photonic neuromorphics project. Copy these files to `.github/workflows/` directory.

## 🔄 Complete CI Workflow (.github/workflows/ci.yml)

```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 6 * * 1'  # Weekly Monday 6 AM

env:
  PYTHON_VERSION_DEFAULT: "3.11"

jobs:
  lint:
    name: Code Quality Checks
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION_DEFAULT }}
          
      - name: Cache pip dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: pip-${{ runner.os }}-${{ hashFiles('**/requirements*.txt') }}
          restore-keys: |
            pip-${{ runner.os }}-
            
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev,test]"
          
      - name: Run Black formatter check
        run: black --check --diff src tests
        
      - name: Run Ruff linter
        run: ruff check src tests
        
      - name: Run MyPy type checker
        run: mypy src
        
      - name: Run Bandit security linter
        run: bandit -r src/ -f json -o bandit-report.json
        continue-on-error: true
        
      - name: Upload Bandit results
        uses: actions/upload-artifact@v3
        with:
          name: bandit-results
          path: bandit-report.json

  test:
    name: Test Suite
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.9", "3.10", "3.11", "3.12"]
        
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
          
      - name: Cache pip dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: pip-${{ runner.os }}-${{ matrix.python-version }}-${{ hashFiles('**/requirements*.txt') }}
          
      - name: Install system dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y libhdf5-dev libopenblas-dev gfortran
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev,test]"
          
      - name: Run unit tests
        run: |
          pytest tests/unit/ -v \
            --cov=photonic_neuromorphics \
            --cov-report=xml \
            --cov-report=term-missing \
            --junit-xml=junit/test-results-${{ matrix.python-version }}.xml
            
      - name: Run integration tests
        run: |
          pytest tests/integration/ -v \
            --junit-xml=junit/integration-results-${{ matrix.python-version }}.xml
        continue-on-error: true
        
      - name: Upload coverage to Codecov
        if: matrix.python-version == env.PYTHON_VERSION_DEFAULT
        uses: codecov/codecov-action@v3
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
          file: ./coverage.xml
          flags: unittests
          name: codecov-umbrella
          
      - name: Upload test results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: test-results-${{ matrix.python-version }}
          path: junit/

  build:
    name: Build Package
    runs-on: ubuntu-latest
    needs: [lint, test]
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION_DEFAULT }}
          
      - name: Install build dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine
          
      - name: Build package
        run: python -m build
        
      - name: Check package
        run: twine check dist/*
        
      - name: Upload build artifacts
        uses: actions/upload-artifact@v3
        with:
          name: dist-packages
          path: dist/

  docker:
    name: Docker Build Test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
        
      - name: Build Docker image
        uses: docker/build-push-action@v4
        with:
          context: .
          target: production
          push: false
          tags: photonic-neuromorphics:test
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

## 📚 Documentation Workflow (.github/workflows/docs.yml)

```yaml
name: Documentation

on:
  push:
    branches: [ main ]
    paths: 
      - 'docs/**'
      - 'src/**/*.py'
      - 'README.md'
  pull_request:
    paths:
      - 'docs/**'
      - 'src/**/*.py'

jobs:
  build-docs:
    name: Build Documentation
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[docs]"
          
      - name: Build documentation
        run: |
          cd docs
          sphinx-build -b html . _build/html -W --keep-going
          
      - name: Check for broken links
        run: |
          cd docs
          sphinx-build -b linkcheck . _build/linkcheck
        continue-on-error: true
        
      - name: Upload documentation
        uses: actions/upload-artifact@v3
        with:
          name: documentation
          path: docs/_build/html/

  deploy-docs:
    name: Deploy to GitHub Pages
    runs-on: ubuntu-latest
    needs: build-docs
    if: github.ref == 'refs/heads/main'
    permissions:
      contents: read
      pages: write
      id-token: write
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Download documentation
        uses: actions/download-artifact@v3
        with:
          name: documentation
          path: ./docs
          
      - name: Setup Pages
        uses: actions/configure-pages@v3
        
      - name: Upload to GitHub Pages
        uses: actions/upload-pages-artifact@v2
        with:
          path: ./docs
          
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v2
```

## 🔒 Security Workflow (.github/workflows/security.yml)

```yaml
name: Security

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 6 * * 1'  # Weekly Monday 6 AM

jobs:
  dependency-scan:
    name: Dependency Security Scan
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"
          
      - name: Install Safety
        run: pip install safety
        
      - name: Run Safety check
        run: |
          safety check --json --output safety-report.json || true
          safety check
        continue-on-error: true
        
      - name: Upload Safety results
        uses: actions/upload-artifact@v3
        with:
          name: safety-results
          path: safety-report.json

  codeql-analysis:
    name: CodeQL Analysis
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write
    steps:
      - uses: actions/checkout@v4
      
      - name: Initialize CodeQL
        uses: github/codeql-action/init@v2
        with:
          languages: python
          queries: security-extended,security-and-quality
          
      - name: Autobuild
        uses: github/codeql-action/autobuild@v2
        
      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v2

  secret-scan:
    name: Secret Detection
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          
      - name: Run GitLeaks
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  sbom-generation:
    name: Generate SBOM
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .
          pip install cyclonedx-bom
          
      - name: Generate SBOM
        run: |
          cyclonedx-py --output-file sbom.json --format json
          cyclonedx-py --output-file sbom.xml --format xml
          
      - name: Upload SBOM
        uses: actions/upload-artifact@v3
        with:
          name: sbom
          path: |
            sbom.json
            sbom.xml
```

## 🚀 Release Workflow (.github/workflows/release.yml)

```yaml
name: Release

on:
  push:
    tags:
      - 'v*.*.*'

env:
  PYTHON_VERSION: "3.11"

jobs:
  validate-tag:
    name: Validate Release Tag
    runs-on: ubuntu-latest
    outputs:
      version: ${{ steps.get_version.outputs.version }}
    steps:
      - uses: actions/checkout@v4
      
      - name: Get version from tag
        id: get_version
        run: |
          VERSION=${GITHUB_REF#refs/tags/v}
          echo "version=$VERSION" >> $GITHUB_OUTPUT
          
      - name: Validate version format
        run: |
          if ! [[ "${{ steps.get_version.outputs.version }}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            echo "Invalid version format: ${{ steps.get_version.outputs.version }}"
            exit 1
          fi

  build-release:
    name: Build Release Packages
    runs-on: ubuntu-latest
    needs: validate-tag
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          
      - name: Install build dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine
          
      - name: Build packages
        run: python -m build
        
      - name: Check packages
        run: twine check dist/*
        
      - name: Upload packages
        uses: actions/upload-artifact@v3
        with:
          name: release-packages
          path: dist/

  test-release:
    name: Test Release Installation
    runs-on: ubuntu-latest
    needs: build-release
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11", "3.12"]
    steps:
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
          
      - name: Download packages
        uses: actions/download-artifact@v3
        with:
          name: release-packages
          path: dist/
          
      - name: Install from wheel
        run: |
          pip install dist/*.whl
          python -c "import photonic_neuromorphics; print(photonic_neuromorphics.__version__)"
          
      - name: Test CLI
        run: photonic-sim --help

  publish-pypi:
    name: Publish to PyPI
    runs-on: ubuntu-latest
    needs: [validate-tag, test-release]
    environment: release
    permissions:
      id-token: write  # IMPORTANT: this permission is mandatory for trusted publishing
    steps:
      - name: Download packages
        uses: actions/download-artifact@v3
        with:
          name: release-packages
          path: dist/
          
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}

  create-github-release:
    name: Create GitHub Release
    runs-on: ubuntu-latest
    needs: [validate-tag, publish-pypi]
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          
      - name: Generate changelog
        id: changelog
        run: |
          # Generate changelog between tags
          if [ $(git tag -l | wc -l) -gt 1 ]; then
            PREV_TAG=$(git tag --sort=-version:refname | head -2 | tail -1)
            git log --pretty=format:"- %s (%h)" $PREV_TAG..HEAD > changelog.md
          else
            echo "- Initial release" > changelog.md
          fi
          
      - name: Create Release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: ${{ github.ref }}
          release_name: Release ${{ needs.validate-tag.outputs.version }}
          body_path: changelog.md
          draft: false
          prerelease: false
```

## 🔧 Dependabot Configuration (.github/dependabot.yml)

```yaml
version: 2
updates:
  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "06:00"
    open-pull-requests-limit: 10
    reviewers:
      - "danieleschmidt"
    assignees:
      - "danieleschmidt"
    commit-message:
      prefix: "deps"
      include: "scope"
    
  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "06:00"
    open-pull-requests-limit: 5
    reviewers:
      - "danieleschmidt"
    commit-message:
      prefix: "ci"
      include: "scope"

  # Docker
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "06:00"
    open-pull-requests-limit: 5
```

## 📋 Implementation Instructions

### 1. Create Workflow Directory
```bash
mkdir -p .github/workflows
```

### 2. Copy Workflow Files
Copy each workflow template above into separate files in `.github/workflows/`:
- `ci.yml`
- `docs.yml` 
- `security.yml`
- `release.yml`

### 3. Copy Dependabot Config
```bash
cp dependabot.yml .github/dependabot.yml
```

### 4. Configure GitHub Secrets
Add these secrets in GitHub repository settings:
- `CODECOV_TOKEN`
- `PYPI_API_TOKEN`
- `SLACK_WEBHOOK` (optional)

### 5. Enable GitHub Pages
1. Go to repository Settings → Pages
2. Set source to "GitHub Actions"
3. Configure custom domain if needed

### 6. Configure Branch Protection
Set up branch protection rules for `main` branch with required status checks.

These workflows provide comprehensive CI/CD automation while maintaining security and quality standards for the photonic neuromorphics project.