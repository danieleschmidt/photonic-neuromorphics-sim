# Claude Code Memory File

This file contains important context and configuration information for Claude Code to remember about this project.

## Project Overview

**Name**: Photonic Neuromorphics Simulation Platform  
**Repository**: danieleschmidt/photonic-neuromorphics-sim  
**Type**: Scientific simulation software for silicon-photonic spiking neural networks  
**Primary Language**: Python  
**Target**: MPW tape-out ready RTL generation for photonic neuromorphic processors  

## Development Environment

### Python Environment
- **Python Version**: 3.9+
- **Package Manager**: pip with pip-tools for dependency management
- **Virtual Environment**: Recommended (conda or venv)
- **Key Dependencies**: PyTorch, NumPy, SciPy, matplotlib

### Code Quality Tools
- **Formatter**: Black (line length: 88)
- **Import Sorting**: isort
- **Linting**: flake8 with specific configuration
- **Type Checking**: mypy for static type analysis
- **Pre-commit**: Configured with quality checks

### Testing Framework
- **Test Runner**: pytest
- **Coverage**: pytest-cov with 90% target coverage
- **Test Types**: unit, integration, e2e, performance, contract, security
- **Configuration**: pytest.ini, tox.ini for multi-environment testing

## Project Structure

```
src/photonic_neuromorphics/    # Main package source code
tests/                         # Comprehensive test suite
docs/                         # Documentation (guides, ADRs, specs)
scripts/                      # Automation and utility scripts
monitoring/                   # Observability configuration
.github/                      # GitHub workflows and project metrics
```

## Key Commands

### Development
```bash
# Setup environment
pip install -r requirements-dev.txt
pip install -e .

# Code quality
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/

# Testing
pytest tests/
pytest --cov=src --cov-report=html
tox  # Multi-environment testing

# Build
docker build -t photonic-neuromorphics .
python -m build
```

### Automation Scripts
```bash
# Metrics collection
python scripts/metrics/collect_metrics.py --report

# Dependency updates
python scripts/automation/dependency_updater.py --type security --create-pr

# Health check
python scripts/maintenance/repository_health_check.py --format markdown

# Release automation
python scripts/automation/release_automation.py --bump minor

# Quality monitoring
python scripts/automation/code_quality_monitor.py --report --dashboard
```

## Documentation Standards

### Architecture Decision Records (ADRs)
- Location: `docs/adr/`
- Template: `docs/adr/0000-adr-template.md`
- Naming: `NNNN-decision-title.md`

### Code Documentation
- **Docstring Style**: Google/NumPy style
- **API Documentation**: Auto-generated from docstrings
- **Coverage Target**: 90% docstring coverage for public APIs

### Commit Convention
- **Format**: `type(scope): description`
- **Types**: feat, fix, docs, style, refactor, test, chore, security
- **Breaking Changes**: Include `!` or `BREAKING CHANGE:` in footer

## Security and Compliance

### Security Scanning
- **SAST**: bandit for Python security analysis
- **Dependencies**: safety, pip-audit for vulnerability scanning
- **Secrets**: detect-secrets, trufflehog for secrets detection
- **Containers**: trivy for container vulnerability scanning

### Compliance Standards
- **SLSA**: Supply chain security framework (Level 3 target)
- **SOC 2**: Security controls framework
- **GDPR**: Data protection compliance
- **NIST Cybersecurity Framework**: Primary security framework

## Monitoring and Observability

### Metrics Stack
- **Collection**: Prometheus, OpenTelemetry Collector
- **Storage**: Prometheus, InfluxDB (long-term)
- **Visualization**: Grafana dashboards
- **Alerting**: AlertManager with Slack/email notifications

### Logging
- **Aggregation**: Loki
- **Collection**: Promtail
- **Structured Logging**: JSON format with correlation IDs
- **Retention**: 365 days for logs, 7 years for compliance data

### Key Metrics
- **Code Quality**: Test coverage, complexity, maintainability
- **Security**: Vulnerability count, secrets detection
- **Performance**: Build time, test execution, simulation time
- **Business**: User adoption, simulation success rate, model accuracy

## Automation and CI/CD

### GitHub Workflows (Manual Setup Required)
- **CI**: `docs/workflows/examples/ci.yml`
- **CD**: `docs/workflows/examples/cd.yml`
- **Security**: `docs/workflows/examples/security-scan.yml`
- **Dependencies**: `docs/workflows/examples/dependency-update.yml`

### Deployment Strategy
- **Staging**: Automatic deployment on main branch
- **Production**: Manual approval required
- **Rollback**: Automated rollback capability
- **Blue-Green**: Production deployment strategy

## Performance Targets

### Build Performance
- **Build Time**: < 5 minutes (target: 3 minutes)
- **Test Execution**: < 2 minutes (target: 1 minute)
- **Docker Build**: < 3 minutes (target: 2 minutes)

### Application Performance
- **Simulation Time**: < 5 minutes for standard models
- **Memory Usage**: < 2GB peak usage
- **Startup Time**: < 30 seconds
- **Response Time**: < 500ms (95th percentile)

## Quality Targets

### Code Quality
- **Test Coverage**: ≥90% (current measurement via pytest-cov)
- **Code Complexity**: ≤5 average cyclomatic complexity
- **Maintainability**: ≥80 maintainability index
- **Documentation**: ≥90% public API documentation

### Security
- **Vulnerabilities**: 0 critical, 0 high severity
- **Dependencies**: All up-to-date with security patches
- **Secrets**: 0 exposed secrets in code
- **Compliance**: SLSA Level 3, SOC 2 ready

## Release Management

### Versioning
- **Scheme**: Semantic Versioning (MAJOR.MINOR.PATCH)
- **Automation**: Automated version bumping based on commit analysis
- **Changelog**: Auto-generated from commit messages
- **Tags**: Annotated git tags for releases

### Release Process
1. **Pre-release**: Automated checks (tests, security, build)
2. **Version Bump**: Automated based on commit types
3. **Changelog**: Generated from commits since last release
4. **Git Tag**: Annotated tag with release notes
5. **GitHub Release**: Automated release creation
6. **PyPI**: Optional publishing to Python Package Index

## Known Issues and Limitations

### GitHub App Permissions
- **Limitation**: Cannot create GitHub workflows due to app permissions
- **Workaround**: Manual creation of workflows from templates in `docs/workflows/examples/`
- **Required Action**: Repository maintainers must manually set up workflows

### External Dependencies
- **Monitoring**: Requires external Prometheus/Grafana setup
- **Notifications**: Requires Slack webhook configuration
- **Cloud Services**: AWS credentials needed for deployment workflows

## Environment Variables

### Required for Automation
```bash
GITHUB_TOKEN              # GitHub API access
GITHUB_REPOSITORY         # Repository identifier
DEPENDENCY_UPDATE_TOKEN   # GitHub PAT for dependency PRs
PYPI_API_TOKEN           # PyPI publishing (optional)
SLACK_WEBHOOK_URL        # Slack notifications (optional)
```

### Optional for Enhanced Features
```bash
PROMETHEUS_URL           # Prometheus endpoint for metrics export
SNYK_TOKEN              # Snyk security scanning
CODECOV_TOKEN           # Codecov integration
AWS_ACCESS_KEY_ID       # AWS deployment
AWS_SECRET_ACCESS_KEY   # AWS deployment
```

## Team Workflow

### Development Workflow
1. **Feature Development**: Create feature branch from main
2. **Code Quality**: Pre-commit hooks enforce quality standards
3. **Testing**: Comprehensive test suite validation
4. **Pull Request**: Required for all changes to main
5. **Review**: Code review and approval required
6. **Merge**: Automated after approval and checks pass

### Maintenance Workflow
1. **Daily**: Automated metrics collection and health checks
2. **Weekly**: Dependency updates and quality reports
3. **Monthly**: Comprehensive security scans and compliance reviews
4. **Quarterly**: Architecture reviews and process improvements

## Contact and Support

### Project Maintainers
- **Primary**: Repository owner
- **Security**: security@photonic-neuromorphics.com
- **Support**: GitHub issues and discussions

### Documentation
- **Primary**: Repository README and docs/ directory
- **Architecture**: docs/ARCHITECTURE.md
- **API**: Auto-generated from code docstrings
- **Runbooks**: docs/runbooks/ for operational procedures

---

*This file is automatically updated by automation scripts. Last updated by SDLC implementation on 2025-01-01.*