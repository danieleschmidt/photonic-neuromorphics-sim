# Automation Scripts

This directory contains automation scripts for maintaining and monitoring the photonic neuromorphics simulation platform.

## Scripts Overview

### 1. Metrics Collection (`collect_metrics.py`)
Comprehensive metrics collection system that gathers code quality, security, performance, and business metrics.

**Features:**
- Code quality metrics (test coverage, complexity, maintainability)
- Security metrics (vulnerabilities, secrets detection)
- Performance metrics (build time, test execution time)
- Development metrics (commit frequency, PR cycle time)
- GitHub integration for deployment and build metrics
- Prometheus export capability

**Usage:**
```bash
# Collect all metrics
python scripts/metrics/collect_metrics.py

# Generate summary report
python scripts/metrics/collect_metrics.py --report

# Send to Prometheus
python scripts/metrics/collect_metrics.py --prometheus http://localhost:9091
```

### 2. Dependency Updater (`dependency_updater.py`)
Automated dependency management system for Python packages, Docker images, and GitHub Actions.

**Features:**
- Python package updates with security prioritization
- Docker base image updates
- GitHub Actions version updates
- Automated testing and validation
- Pull request creation
- Update type classification (major, minor, patch, security)

**Usage:**
```bash
# Check for all updates
python scripts/automation/dependency_updater.py --check-only

# Apply security updates only
python scripts/automation/dependency_updater.py --type security --create-pr

# Apply all updates
python scripts/automation/dependency_updater.py --type all --create-pr
```

### 3. Repository Health Check (`repository_health_check.py`)
Comprehensive repository health assessment tool that evaluates multiple aspects of repository quality.

**Features:**
- Code quality assessment
- Security posture evaluation
- Documentation completeness check
- Testing infrastructure analysis
- CI/CD pipeline health
- Dependency freshness
- Repository structure validation
- Performance metrics
- Compliance checking

**Usage:**
```bash
# Run full health check
python scripts/maintenance/repository_health_check.py

# Generate markdown report
python scripts/maintenance/repository_health_check.py --format markdown

# Save to custom location
python scripts/maintenance/repository_health_check.py --output custom_report.json
```

### 4. Release Automation (`release_automation.py`)
Automated release management system with semantic versioning and changelog generation.

**Features:**
- Automatic version bumping (major, minor, patch)
- Semantic commit analysis
- Changelog generation
- Git tag creation
- GitHub release creation
- PyPI publishing
- Pre-release validation
- Rollback capability

**Usage:**
```bash
# Auto-detect release type and create release
python scripts/automation/release_automation.py

# Specify release type
python scripts/automation/release_automation.py --bump minor

# Dry run to see what would happen
python scripts/automation/release_automation.py --dry-run

# Publish to PyPI
python scripts/automation/release_automation.py --publish-pypi

# Rollback a release
python scripts/automation/release_automation.py --rollback 1.2.3
```

### 5. Code Quality Monitor (`code_quality_monitor.py`)
Continuous code quality monitoring with trend analysis and reporting.

**Features:**
- Test coverage measurement
- Code complexity analysis
- Maintainability index calculation
- Code duplication detection
- Technical debt estimation
- Quality score calculation
- Trend analysis
- Visual dashboard creation
- Historical metrics tracking

**Usage:**
```bash
# Monitor quality metrics
python scripts/automation/code_quality_monitor.py

# Generate quality report
python scripts/automation/code_quality_monitor.py --report

# Create visual dashboard
python scripts/automation/code_quality_monitor.py --dashboard

# Analyze 60-day trends
python scripts/automation/code_quality_monitor.py --trends 60
```

## Configuration

### Environment Variables

All scripts support these common environment variables:

```bash
# GitHub integration
GITHUB_TOKEN=your_github_token
GITHUB_REPOSITORY=danieleschmidt/photonic-neuromorphics-sim

# Prometheus integration
PROMETHEUS_URL=http://localhost:9090

# Security scanning
SNYK_TOKEN=your_snyk_token

# PyPI publishing
PYPI_API_TOKEN=your_pypi_token

# Notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

### GitHub Secrets

Configure these secrets in your GitHub repository:

```bash
GITHUB_TOKEN          # For API access
DEPENDENCY_UPDATE_TOKEN # For dependency PRs
PYPI_API_TOKEN       # For package publishing
SLACK_WEBHOOK_URL    # For notifications
```

## Automation Workflows

### Daily Automation
Set up daily automation with cron or GitHub Actions:

```bash
# Daily metrics collection
0 2 * * * cd /path/to/repo && python scripts/metrics/collect_metrics.py

# Daily health check
0 3 * * * cd /path/to/repo && python scripts/maintenance/repository_health_check.py

# Daily quality monitoring
0 4 * * * cd /path/to/repo && python scripts/automation/code_quality_monitor.py
```

### Weekly Automation
```bash
# Weekly dependency updates
0 9 * * 1 cd /path/to/repo && python scripts/automation/dependency_updater.py --type all --create-pr

# Weekly quality reports
0 10 * * 1 cd /path/to/repo && python scripts/automation/code_quality_monitor.py --report --dashboard
```

### Release Automation
Integrate with GitHub Actions for automated releases:

```yaml
# .github/workflows/release.yml
name: Automated Release
on:
  push:
    tags: ['v*']
jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Create Release
        run: python scripts/automation/release_automation.py --publish-pypi
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          PYPI_API_TOKEN: ${{ secrets.PYPI_API_TOKEN }}
```

## Integration with CI/CD

### GitHub Actions Integration

Example workflow that runs automation scripts:

```yaml
name: Quality Monitoring
on:
  push:
    branches: [main]
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM

jobs:
  quality-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-dev.txt
      
      - name: Collect metrics
        run: python scripts/metrics/collect_metrics.py --report
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Health check
        run: python scripts/maintenance/repository_health_check.py --format markdown
      
      - name: Quality monitoring
        run: python scripts/automation/code_quality_monitor.py --report --dashboard
      
      - name: Upload reports
        uses: actions/upload-artifact@v3
        with:
          name: quality-reports
          path: |
            metrics_report.md
            health_report.md
            quality_report.md
            quality_dashboard.png
```

### Jenkins Integration

Example Jenkinsfile for automation:

```groovy
pipeline {
    agent any
    
    triggers {
        cron('H 2 * * *')  // Daily
    }
    
    stages {
        stage('Setup') {
            steps {
                sh 'python -m pip install --upgrade pip'
                sh 'pip install -r requirements-dev.txt'
            }
        }
        
        stage('Metrics Collection') {
            steps {
                sh 'python scripts/metrics/collect_metrics.py --report'
            }
        }
        
        stage('Health Check') {
            steps {
                sh 'python scripts/maintenance/repository_health_check.py'
            }
        }
        
        stage('Quality Monitoring') {
            steps {
                sh 'python scripts/automation/code_quality_monitor.py --report'
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: '*_report.md', allowEmptyArchive: true
            publishHTML([
                allowMissing: false,
                alwaysLinkToLastBuild: true,
                keepAll: true,
                reportDir: '.',
                reportFiles: 'quality_report.md',
                reportName: 'Quality Report'
            ])
        }
    }
}
```

## Monitoring and Alerting

### Prometheus Integration

Export metrics to Prometheus for monitoring:

```python
# Automated metrics export
python scripts/metrics/collect_metrics.py --prometheus http://prometheus:9091
```

### Slack Notifications

Configure Slack notifications for important events:

```bash
# Set webhook URL
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."

# Notifications will be sent automatically for:
# - Security vulnerabilities
# - Health check failures
# - Quality degradation
# - Release completions
```

### Email Reports

Set up email reports for weekly summaries:

```bash
# Configure SMTP settings
export SMTP_SERVER="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USERNAME="reports@yourcompany.com"
export SMTP_PASSWORD="your_app_password"
```

## Troubleshooting

### Common Issues

1. **Permission Errors**
   ```bash
   # Ensure proper permissions
   chmod +x scripts/automation/*.py
   
   # Check GitHub token permissions
   curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user
   ```

2. **Missing Dependencies**
   ```bash
   # Install all required dependencies
   pip install -r requirements-dev.txt
   
   # Install system dependencies
   sudo apt-get install cloc  # For line counting
   ```

3. **Git Configuration**
   ```bash
   # Configure git for automation
   git config user.name "Automation Bot"
   git config user.email "automation@yourcompany.com"
   ```

### Debug Mode

Enable debug output for troubleshooting:

```bash
# Set debug environment variable
export DEBUG=1

# Run scripts with verbose output
python scripts/automation/dependency_updater.py --check-only
```

### Log Files

Scripts generate log files in the `logs/` directory:

```bash
# View recent logs
tail -f logs/automation.log

# Check error logs
grep ERROR logs/automation.log
```

## Best Practices

### Security
- Store sensitive tokens in environment variables or secret management systems
- Use least-privilege access for GitHub tokens
- Regularly rotate API keys and tokens
- Validate all external inputs

### Reliability
- Implement retry logic for network operations
- Handle errors gracefully with meaningful messages
- Use atomic operations for file modifications
- Maintain backward compatibility for configuration changes

### Performance
- Cache expensive operations
- Use parallel processing where appropriate
- Implement rate limiting for API calls
- Optimize database queries and file operations

### Maintainability
- Follow consistent coding standards
- Document all configuration options
- Implement comprehensive error handling
- Write unit tests for critical functions

## Contributing

When adding new automation scripts:

1. Follow the existing code structure and patterns
2. Add comprehensive error handling
3. Include configuration documentation
4. Write unit tests
5. Update this README with usage examples
6. Add logging and monitoring capabilities

## Support

For issues with automation scripts:

1. Check the troubleshooting section above
2. Review log files for error messages
3. Verify configuration and environment variables
4. Test scripts in isolation to identify issues
5. Create GitHub issues with detailed error information