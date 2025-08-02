# GitHub Workflows Documentation

This directory contains documentation and templates for GitHub Actions workflows that need to be manually created by repository maintainers due to GitHub App permission limitations.

## ⚠️ MANUAL ACTION REQUIRED

The GitHub App used by Claude Code does not have permissions to create or modify GitHub workflows. Repository maintainers must manually create these workflow files in the `.github/workflows/` directory.

## Available Workflows

### 1. Continuous Integration (CI)
**File**: `.github/workflows/ci.yml`  
**Template**: [examples/ci.yml](examples/ci.yml)

Comprehensive CI pipeline that includes:
- Code quality checks (linting, type checking)
- Security scanning (Bandit, CodeQL, dependency audit)
- Testing (unit, integration, contract, performance)
- Build verification and Docker image creation
- Documentation generation
- SLSA provenance generation

**Key Features**:
- Multi-Python version matrix testing
- Comprehensive security scanning
- Performance regression detection
- Docker multi-platform builds
- Automated dependency vulnerability checking

### 2. Continuous Deployment (CD)
**File**: `.github/workflows/cd.yml`  
**Template**: [examples/cd.yml](examples/cd.yml)

Production-ready deployment pipeline featuring:
- Blue-green deployments
- Automated rollback capabilities
- Environment-specific configurations
- Database migrations
- Post-deployment verification
- PyPI package publishing

**Deployment Targets**:
- Staging environment (automatic on main branch)
- Production environment (manual approval required)
- Documentation deployment to GitHub Pages

### 3. Security Scanning
**File**: `.github/workflows/security-scan.yml`  
**Template**: [examples/security-scan.yml](examples/security-scan.yml)

Comprehensive security analysis including:
- Static Application Security Testing (SAST)
- Dependency vulnerability scanning
- Container security analysis
- Secrets detection
- Infrastructure security checks
- License compliance validation
- Malware scanning

**Security Tools Integrated**:
- Bandit, Semgrep, CodeQL for SAST
- pip-audit, Safety for dependency scanning
- Trivy, Snyk for container scanning
- TruffleHog, GitLeaks for secrets detection

### 4. Dependency Updates
**File**: `.github/workflows/dependency-update.yml`  
**Template**: [examples/dependency-update.yml](examples/dependency-update.yml)

Automated dependency management system:
- Weekly scheduled updates
- Security vulnerability prioritization
- Python, Docker, and GitHub Actions updates
- Automated testing and validation
- Consolidated update PRs

**Update Types**:
- Security updates (immediate priority)
- Minor and patch updates
- Docker base image updates
- GitHub Actions version updates

## Setup Instructions

### 1. Copy Workflow Templates

```bash
# Create workflows directory
mkdir -p .github/workflows

# Copy templates (choose what you need)
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/cd.yml .github/workflows/
cp docs/workflows/examples/security-scan.yml .github/workflows/
cp docs/workflows/examples/dependency-update.yml .github/workflows/
```

### 2. Configure Repository Secrets

Add the following secrets in GitHub repository settings:

#### Required Secrets
```bash
# Docker/Container Registry
GITHUB_TOKEN                    # Automatic (for GHCR)

# AWS Deployment (if using AWS)
AWS_ACCESS_KEY_ID              # AWS access key for staging
AWS_SECRET_ACCESS_KEY          # AWS secret key for staging
AWS_ACCESS_KEY_ID_PROD         # AWS access key for production
AWS_SECRET_ACCESS_KEY_PROD     # AWS secret key for production

# Database
DATABASE_URL                   # Database connection string

# Notifications
SLACK_WEBHOOK_URL             # Slack webhook for notifications
SECURITY_SLACK_WEBHOOK        # Dedicated security notifications

# Security Scanning
SNYK_TOKEN                    # Snyk API token
CODECOV_TOKEN                 # Codecov upload token

# Dependency Updates
DEPENDENCY_UPDATE_TOKEN       # GitHub token with write permissions

# PyPI Publishing
PYPI_API_TOKEN               # PyPI API token for package publishing
```

#### Optional Secrets
```bash
# External Services
PAGERDUTY_ROUTING_KEY        # PagerDuty integration
DATADOG_API_KEY              # Datadog monitoring
NEW_RELIC_LICENSE_KEY        # New Relic APM

# Container Scanning
DOCKER_SCOUT_TOKEN           # Docker Scout (if available)

# Compliance
SONARCLOUD_TOKEN             # SonarCloud integration
```

### 3. Configure Branch Protection

Set up branch protection rules for the main branch:

```yaml
# Branch protection configuration
required_status_checks:
  strict: true
  contexts:
    - "CI Status"
    - "Security Scanning"
    - "Build & Package"

enforce_admins: true
required_pull_request_reviews:
  required_approving_review_count: 2
  dismiss_stale_reviews: true
  require_code_owner_reviews: true

restrictions:
  users: []
  teams: ["maintainers"]
```

### 4. Environment Configuration

Create deployment environments in GitHub:

#### Staging Environment
- **Name**: `staging`
- **URL**: `https://staging.photonic-neuromorphics.com`
- **Protection Rules**: None (auto-deploy)

#### Production Environment
- **Name**: `production`
- **URL**: `https://photonic-neuromorphics.com`
- **Protection Rules**: 
  - Required reviewers: 2
  - Wait timer: 5 minutes
  - Restrict to protected branches

## Workflow Configuration

### Environment Variables

Each workflow supports environment-specific configuration:

```yaml
env:
  PYTHON_VERSION: '3.9'
  NODE_VERSION: '18'
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}
```

### Matrix Testing

CI workflow includes matrix testing across:
- Python versions: 3.9, 3.10, 3.11
- Test types: unit, integration, contract
- Platforms: ubuntu-latest, windows-latest, macos-latest (optional)

### Security Configuration

Security scanning is configured with:
- SARIF uploads to GitHub Security tab
- Multiple security tool integration
- Automatic vulnerability detection
- License compliance checking

## Customization Guide

### Adding Custom Steps

To add custom steps to workflows:

1. **Before existing steps**:
```yaml
- name: Custom setup
  run: |
    echo "Custom setup commands"
```

2. **After existing steps**:
```yaml
- name: Custom cleanup
  if: always()
  run: |
    echo "Custom cleanup commands"
```

### Environment-Specific Configuration

Add environment-specific steps:

```yaml
- name: Production-only step
  if: github.ref == 'refs/heads/main'
  run: |
    echo "Production deployment commands"
```

### Custom Security Tools

Add additional security scanning tools:

```yaml
- name: Custom security scan
  uses: custom/security-action@v1
  with:
    config-file: .security-config.yml
```

## Monitoring and Alerting

### Workflow Monitoring

Monitor workflow health through:
- GitHub Actions dashboard
- Slack notifications
- Custom metrics collection
- Performance tracking

### Alert Configuration

Set up alerts for:
- Workflow failures
- Security scan findings
- Deployment issues
- Performance regressions

## Troubleshooting

### Common Issues

1. **Permission Errors**:
   - Verify GitHub token permissions
   - Check repository settings
   - Ensure secrets are configured

2. **Build Failures**:
   - Check dependency compatibility
   - Verify test configuration
   - Review error logs

3. **Deployment Issues**:
   - Validate AWS credentials
   - Check environment configuration
   - Verify Docker image builds

### Debug Mode

Enable debug logging:

```yaml
env:
  ACTIONS_STEP_DEBUG: true
  ACTIONS_RUNNER_DEBUG: true
```

### Getting Help

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax Reference](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions)
- [Security Best Practices](https://docs.github.com/en/actions/security-guides)

## Best Practices

### Security
- Use least privilege access
- Store secrets securely
- Validate all inputs
- Monitor for vulnerabilities

### Performance
- Use caching for dependencies
- Parallelize jobs when possible
- Optimize Docker builds
- Monitor workflow execution time

### Maintenance
- Regular dependency updates
- Workflow health monitoring
- Documentation updates
- Team training

## Migration Notes

When migrating from other CI/CD systems:

1. **Jenkins**: Map Jenkinsfile stages to GitHub Actions jobs
2. **GitLab CI**: Convert `.gitlab-ci.yml` to GitHub Actions syntax
3. **CircleCI**: Adapt `.circleci/config.yml` to workflow format
4. **Azure DevOps**: Transform pipeline YAML to GitHub Actions

## Compliance

These workflows are designed to meet:
- SOC 2 compliance requirements
- GDPR data protection standards
- SLSA supply chain security
- Industry security best practices

## Support

For workflow-related issues:
- Check [GitHub Status](https://www.githubstatus.com/)
- Review [Actions Community](https://github.community/c/github-actions)
- Contact repository maintainers
- File issues in this repository