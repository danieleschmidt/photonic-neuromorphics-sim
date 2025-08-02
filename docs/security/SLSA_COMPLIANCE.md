# SLSA Compliance Documentation

Supply-chain Levels for Software Artifacts (SLSA) compliance implementation for the photonic neuromorphics simulation platform.

## Overview

SLSA (Supply-chain Levels for Software Artifacts) is a security framework designed to prevent tampering, improve integrity, and secure packages and infrastructure. This document outlines our SLSA compliance implementation.

## SLSA Levels Implementation

### SLSA Level 1: Documentation of Build Process

**Status**: ✅ Implemented

#### Requirements Met:
- **Build Process Documentation**: Comprehensive build documentation in `docs/BUILD.md`
- **Version Control**: All source code tracked in Git with signed commits
- **Build Script**: Automated build process via Docker and GitHub Actions
- **Public Build Instructions**: Available in README and documentation

#### Evidence:
- Build process documented in workflows
- Version control history available
- Automated build via GitHub Actions
- Public repository with build instructions

### SLSA Level 2: Tamper-Resistant Build Service

**Status**: ✅ Implemented

#### Requirements Met:
- **Hosted Build Service**: GitHub Actions used for all builds
- **Source Integrity**: Git commit hashes and branch protection
- **Build Service Authentication**: GitHub Actions with OIDC tokens
- **Ephemeral Environment**: Clean build environment for each run
- **Parameterized Builds**: Configurable via workflow inputs

#### Implementation Details:

```yaml
# SLSA Level 2 compliance in CI workflow
name: SLSA Level 2 Build
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      id-token: write  # Required for SLSA provenance
      contents: read
      packages: write
    
    steps:
      - name: Checkout with full history
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for integrity
      
      - name: Verify commit signature
        run: git verify-commit HEAD
      
      - name: Build with provenance
        uses: slsa-framework/slsa-github-generator/.github/workflows/builder_go_slsa3.yml@v1.9.0
```

### SLSA Level 3: Hardened Build Platform

**Status**: 🚧 Partially Implemented

#### Requirements Met:
- **Isolated Build Environment**: GitHub Actions with controlled environment
- **Provenance Generation**: SLSA provenance generated and signed
- **Non-Falsifiable Provenance**: OIDC tokens and signed attestations
- **Standard Build Process**: Reproducible builds via Docker

#### Provenance Generation:

```yaml
# SLSA Level 3 provenance generation
provenance:
  name: Generate SLSA Provenance
  needs: [build]
  permissions:
    actions: read
    id-token: write
    contents: write
  uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v1.9.0
  with:
    base64-subjects: "${{ needs.build.outputs.hashes }}"
    upload-assets: true
```

#### Requirements In Progress:
- **Hardened Build Platform**: Transition to hardened runners (planned)
- **Isolated Networks**: Network isolation implementation (in progress)

### SLSA Level 4: Highest Integrity

**Status**: 📋 Planned

#### Requirements (Future Implementation):
- **Two-Party Review**: Require multiple approvers for changes
- **Hermetic Builds**: Completely isolated build environment
- **Reproducible Builds**: Bit-for-bit reproducible artifacts
- **Dependencies Tracked**: Complete dependency manifest

## Provenance Generation

### Build Provenance

Every build generates SLSA provenance containing:

```json
{
  "_type": "https://in-toto.io/Statement/v0.1",
  "predicateType": "https://slsa.dev/provenance/v0.2",
  "subject": [
    {
      "name": "photonic-neuromorphics-sim",
      "digest": {
        "sha256": "..."
      }
    }
  ],
  "predicate": {
    "builder": {
      "id": "https://github.com/slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@refs/tags/v1.9.0"
    },
    "buildType": "https://github.com/slsa-framework/slsa-github-generator/generic@v1",
    "invocation": {
      "configSource": {
        "uri": "git+https://github.com/danieleschmidt/photonic-neuromorphics-sim@refs/heads/main",
        "digest": {
          "sha1": "..."
        }
      }
    },
    "metadata": {
      "buildInvocationId": "...",
      "buildStartedOn": "2025-01-01T10:00:00Z",
      "buildFinishedOn": "2025-01-01T10:15:00Z",
      "completeness": {
        "parameters": true,
        "environment": false,
        "materials": true
      },
      "reproducible": false
    },
    "materials": [
      {
        "uri": "git+https://github.com/danieleschmidt/photonic-neuromorphics-sim@refs/heads/main",
        "digest": {
          "sha1": "..."
        }
      }
    ]
  }
}
```

### Verification Process

Users can verify SLSA provenance:

```bash
# Install slsa-verifier
go install github.com/slsa-framework/slsa-verifier/v2/cli/slsa-verifier@latest

# Verify package provenance
slsa-verifier verify-artifact \
  --provenance-path photonic-neuromorphics-sim.intoto.jsonl \
  --source-uri github.com/danieleschmidt/photonic-neuromorphics-sim \
  photonic-neuromorphics-sim.tar.gz
```

## Software Bill of Materials (SBOM)

### SBOM Generation

Automated SBOM generation for all dependencies:

```yaml
# SBOM generation in CI
- name: Generate SBOM
  run: |
    # Python dependencies
    pip install cyclonedx-bom
    cyclonedx-py -o sbom-python.json

    # Container dependencies  
    syft packages dir:. -o cyclonedx-json=sbom-container.json

    # Merge SBOMs
    merge-sboms --output sbom-complete.json sbom-python.json sbom-container.json
```

### SBOM Contents

Our SBOM includes:
- **Python packages**: All pip-installed dependencies
- **System packages**: OS-level dependencies in containers
- **Docker base images**: Base image dependencies
- **Build tools**: Tools used during build process

Example SBOM structure:
```json
{
  "bomFormat": "CycloneDX",
  "specVersion": "1.4",
  "serialNumber": "urn:uuid:...",
  "version": 1,
  "metadata": {
    "timestamp": "2025-01-01T10:00:00Z",
    "tools": [
      {
        "vendor": "CycloneDX",
        "name": "cyclonedx-python",
        "version": "3.11.0"
      }
    ],
    "component": {
      "type": "application",
      "name": "photonic-neuromorphics-sim",
      "version": "1.0.0"
    }
  },
  "components": [
    {
      "type": "library",
      "name": "torch",
      "version": "2.0.1",
      "scope": "required",
      "hashes": [
        {
          "alg": "SHA-256",
          "content": "..."
        }
      ],
      "licenses": [
        {
          "license": {
            "id": "BSD-3-Clause"
          }
        }
      ]
    }
  ]
}
```

## Dependency Management

### Dependency Pinning

All dependencies are pinned with exact versions:

```requirements.txt
# requirements.txt - Pinned dependencies
torch==2.0.1
numpy==1.24.3
scipy==1.10.1
matplotlib==3.7.1
```

### Dependency Verification

Automated verification of dependency integrity:

```yaml
# Dependency verification
- name: Verify dependency hashes
  run: |
    pip-audit --desc
    pip check
    python scripts/verify_dependency_hashes.py
```

### Vulnerability Scanning

Continuous dependency vulnerability scanning:

```yaml
# Security scanning
- name: Scan dependencies
  run: |
    # Python vulnerabilities
    safety check --json
    
    # Container vulnerabilities
    trivy fs --format sarif --output trivy-results.sarif .
```

## Reproducible Builds

### Build Environment

Reproducible build environment specification:

```dockerfile
# Dockerfile with pinned base image
FROM python:3.9.18-slim@sha256:f7318c7aa7d8ccee49ad2f1d2e7e22522e9bc31e0d2db7b93ee5bb626dd9f75b

# Install exact dependency versions
COPY requirements.txt .
RUN pip install --no-deps -r requirements.txt

# Build with fixed timestamps
ENV SOURCE_DATE_EPOCH=1640995200
```

### Build Process

Standardized build process:

```bash
#!/bin/bash
# build.sh - Reproducible build script

set -euo pipefail

# Set reproducible environment
export SOURCE_DATE_EPOCH=1640995200
export PYTHONHASHSEED=0
export TZ=UTC

# Clean build
rm -rf build/ dist/
python -m build --no-isolation

# Verify reproducibility
sha256sum dist/* > checksums.txt
```

## Attestation and Signing

### Code Signing

All releases are signed with GPG keys:

```bash
# Sign release artifacts
gpg --armor --detach-sign photonic-neuromorphics-sim-1.0.0.tar.gz
gpg --verify photonic-neuromorphics-sim-1.0.0.tar.gz.asc
```

### Container Signing

Container images signed with Cosign:

```yaml
# Container signing in CI
- name: Sign container image
  uses: sigstore/cosign-installer@v3.1.1

- name: Sign image with Cosign
  run: |
    cosign sign --yes ghcr.io/org/photonic-neuromorphics-sim:${{ github.sha }}
```

### Verification

Users can verify signatures:

```bash
# Verify container signature
cosign verify ghcr.io/org/photonic-neuromorphics-sim:latest \
  --certificate-identity-regexp="^https://github.com/danieleschmidt/photonic-neuromorphics-sim" \
  --certificate-oidc-issuer="https://token.actions.githubusercontent.com"
```

## Supply Chain Security

### Source Code Integrity

- **Signed commits**: All commits signed with GPG
- **Branch protection**: Main branch protected with required reviews
- **Verified sources**: All dependencies from trusted sources

### Build Integrity

- **Isolated builds**: Each build in clean environment
- **Immutable artifacts**: Built artifacts are immutable
- **Audit trails**: Complete build audit trails maintained

### Distribution Security

- **Secure distribution**: Artifacts distributed via secure channels
- **Access controls**: Limited access to distribution infrastructure
- **Monitoring**: Distribution monitored for integrity

## Compliance Monitoring

### Automated Checks

Regular automated compliance verification:

```yaml
# SLSA compliance check
- name: SLSA Compliance Check
  run: |
    python scripts/slsa_compliance_check.py
    python scripts/verify_provenance.py
    python scripts/audit_dependencies.py
```

### Compliance Reports

Monthly compliance reports generated:

- **SLSA level compliance status**
- **Provenance verification results**
- **Dependency security status**
- **Build reproducibility metrics**

### Continuous Improvement

- **Quarterly reviews**: SLSA compliance assessment
- **Security updates**: Regular security patching
- **Process improvements**: Continuous process enhancement

## Documentation and Training

### Compliance Documentation

- **SLSA implementation guide**
- **Verification procedures**
- **Incident response plans**
- **Audit procedures**

### Team Training

- **SLSA framework training**
- **Security best practices**
- **Incident response procedures**
- **Compliance requirements**

## Incident Response

### Security Incident Procedures

1. **Detection**: Automated monitoring and alerts
2. **Assessment**: Rapid security assessment
3. **Containment**: Immediate containment measures
4. **Investigation**: Thorough investigation
5. **Recovery**: Secure recovery procedures
6. **Lessons Learned**: Post-incident review

### Compliance Violations

1. **Immediate containment**
2. **Impact assessment**
3. **Corrective actions**
4. **Process improvements**
5. **Stakeholder notification**

## Roadmap

### Short Term (Q1 2025)
- [ ] Complete SLSA Level 3 implementation
- [ ] Enhance reproducible builds
- [ ] Implement hermetic builds

### Medium Term (Q2-Q3 2025)
- [ ] SLSA Level 4 planning
- [ ] Advanced threat detection
- [ ] Supply chain risk assessment

### Long Term (Q4 2025+)
- [ ] SLSA Level 4 implementation
- [ ] Zero-trust supply chain
- [ ] Advanced compliance automation

## References

- [SLSA Framework](https://slsa.dev/)
- [SLSA Specification](https://slsa.dev/spec/)
- [GitHub SLSA Generator](https://github.com/slsa-framework/slsa-github-generator)
- [Supply Chain Security Best Practices](https://csrc.nist.gov/publications/detail/sp/800-161/rev-1/final)