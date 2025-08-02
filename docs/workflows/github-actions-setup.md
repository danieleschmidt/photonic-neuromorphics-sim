# GitHub Actions Setup Guide

This guide provides step-by-step instructions for setting up GitHub Actions workflows for the photonic neuromorphics simulation platform.

## Prerequisites

- Repository admin access
- GitHub organization/personal account
- AWS account (for deployment workflows)
- Container registry access (GitHub Container Registry)

## 1. Repository Configuration

### Enable GitHub Actions

1. Go to repository **Settings** → **Actions** → **General**
2. Under "Actions permissions", select:
   - **Allow all actions and reusable workflows**
   - **Allow actions created by GitHub**
   - **Allow actions by Marketplace verified creators**

3. Under "Workflow permissions", select:
   - **Read and write permissions**
   - **Allow GitHub Actions to create and approve pull requests**

### Configure Branch Protection

1. Go to **Settings** → **Branches**
2. Add rule for `main` branch:

```yaml
Branch protection rule:
  - Require a pull request before merging
  - Require status checks to pass before merging
  - Require branches to be up to date before merging
  - Required status checks:
    * CI Status
    * Security Scanning  
    * Build & Package
  - Restrict pushes that create files (optional)
  - Do not allow bypassing the above settings
```

## 2. Secrets Configuration

### Repository Secrets

Go to **Settings** → **Secrets and variables** → **Actions** and add:

#### Core Secrets
```bash
# GitHub Token (automatic)
GITHUB_TOKEN                 # Auto-generated, no action needed

# Container Registry
REGISTRY_USERNAME           # GitHub username
REGISTRY_PASSWORD           # GitHub Personal Access Token
```

#### AWS Deployment Secrets
```bash
# Staging Environment
AWS_ACCESS_KEY_ID          # AWS IAM access key for staging
AWS_SECRET_ACCESS_KEY      # AWS IAM secret key for staging
AWS_REGION                 # us-west-2 (or your preferred region)

# Production Environment  
AWS_ACCESS_KEY_ID_PROD     # AWS IAM access key for production
AWS_SECRET_ACCESS_KEY_PROD # AWS IAM secret key for production
AWS_REGION_PROD           # us-west-2 (or your preferred region)
```

#### Database Secrets
```bash
# Database Configuration
DATABASE_URL              # postgresql://user:pass@host:port/db
DATABASE_URL_STAGING      # Staging database URL
DATABASE_URL_PROD         # Production database URL
```

#### Notification Secrets
```bash
# Slack Integration
SLACK_WEBHOOK_URL         # https://hooks.slack.com/services/...
SECURITY_SLACK_WEBHOOK    # Dedicated security notifications

# Email Configuration (optional)
SMTP_SERVER              # smtp.gmail.com
SMTP_PORT                # 587
SMTP_USERNAME            # notification email username
SMTP_PASSWORD            # notification email password
```

#### Security Tool Secrets
```bash
# Security Scanning
SNYK_TOKEN               # Snyk API token
CODECOV_TOKEN            # Codecov upload token
SONARCLOUD_TOKEN         # SonarCloud project token (optional)

# Dependency Updates
DEPENDENCY_UPDATE_TOKEN  # GitHub PAT with repo permissions
```

#### Publishing Secrets
```bash
# PyPI Publishing
PYPI_API_TOKEN          # PyPI API token for package publishing
TWINE_USERNAME          # __token__
TWINE_PASSWORD          # PyPI API token
```

### Environment Secrets

Create environments and add environment-specific secrets:

#### Staging Environment
1. Go to **Settings** → **Environments**
2. Create new environment: `staging`
3. Add environment secrets:
   ```bash
   DEPLOYMENT_URL=https://staging.photonic-neuromorphics.com
   ```

#### Production Environment
1. Create new environment: `production`
2. Configure protection rules:
   - **Required reviewers**: 2
   - **Wait timer**: 5 minutes
   - **Deployment branches**: Protected branches only
3. Add environment secrets:
   ```bash
   DEPLOYMENT_URL=https://photonic-neuromorphics.com
   ```

## 3. Workflow Files Setup

### Create Workflow Directory
```bash
mkdir -p .github/workflows
```

### Copy and Customize Workflows

#### 1. Continuous Integration
```bash
cp docs/workflows/examples/ci.yml .github/workflows/ci.yml
```

**Customizations needed**:
- Update Python version matrix if needed
- Modify test commands for your project
- Adjust security scanning configuration
- Update Docker registry settings

#### 2. Continuous Deployment
```bash
cp docs/workflows/examples/cd.yml .github/workflows/cd.yml
```

**Customizations needed**:
- Update AWS ECS cluster names
- Modify deployment commands
- Adjust health check URLs
- Update notification channels

#### 3. Security Scanning
```bash
cp docs/workflows/examples/security-scan.yml .github/workflows/security-scan.yml
```

**Customizations needed**:
- Configure security tool tokens
- Adjust scan schedules
- Modify notification settings
- Update compliance requirements

#### 4. Dependency Updates
```bash
cp docs/workflows/examples/dependency-update.yml .github/workflows/dependency-update.yml
```

**Customizations needed**:
- Set update schedule preferences
- Configure notification channels
- Adjust auto-merge settings

## 4. AWS Infrastructure Setup

### Create IAM Roles

#### Staging Role
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecs:UpdateService",
        "ecs:DescribeServices",
        "ecs:DescribeTaskDefinition",
        "ecs:RegisterTaskDefinition",
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage"
      ],
      "Resource": "*"
    }
  ]
}
```

#### Production Role
Similar to staging but with additional security restrictions:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecs:UpdateService",
        "ecs:DescribeServices",
        "ecs:DescribeTaskDefinition",
        "ecs:RegisterTaskDefinition"
      ],
      "Resource": "arn:aws:ecs:*:*:service/photonic-production/*"
    }
  ]
}
```

### ECS Configuration

#### Create ECS Clusters
```bash
# Staging cluster
aws ecs create-cluster --cluster-name photonic-staging

# Production cluster  
aws ecs create-cluster --cluster-name photonic-production
```

#### Task Definition Template
```json
{
  "family": "photonic-neuromorphics",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "photonic-app",
      "image": "ghcr.io/your-org/photonic-neuromorphics-sim:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/photonic-neuromorphics",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

## 5. Container Registry Setup

### GitHub Container Registry

1. Enable GitHub Container Registry:
   - Go to **Settings** → **Developer settings** → **Personal access tokens**
   - Create token with `write:packages` scope

2. Configure package visibility:
   - Go to **Packages** tab in repository
   - Set package visibility to private/public as needed

### Alternative: Docker Hub
```bash
# Add Docker Hub secrets
DOCKER_USERNAME=your-dockerhub-username
DOCKER_PASSWORD=your-dockerhub-token
```

## 6. Notification Setup

### Slack Integration

1. Create Slack App:
   - Go to [Slack API](https://api.slack.com/apps)
   - Create new app
   - Add Incoming Webhooks feature
   - Create webhooks for channels:
     - `#ci-cd` (general notifications)
     - `#security-alerts` (security notifications)
     - `#deployments` (deployment notifications)

2. Configure webhook URLs in repository secrets

### Email Notifications (Optional)

Set up SMTP configuration for email alerts:
```bash
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=notifications@yourcompany.com
SMTP_PASSWORD=app-specific-password
```

## 7. Security Configuration

### Enable Security Features

1. **Dependency Graph**: Settings → Security & analysis → Dependency graph (Enable)
2. **Dependabot alerts**: Enable Dependabot security updates
3. **Code scanning**: Enable CodeQL analysis
4. **Secret scanning**: Enable secret scanning alerts

### Configure SARIF Uploads

Ensure security tools upload SARIF results to GitHub Security tab:
```yaml
- name: Upload security results
  uses: github/codeql-action/upload-sarif@v2
  with:
    sarif_file: security-results.sarif
```

## 8. Testing the Setup

### Trigger Initial Workflows

1. **Push to main branch** - Should trigger CI and CD workflows
2. **Create pull request** - Should trigger CI workflow
3. **Manual dispatch** - Test workflow_dispatch triggers

### Verify Workflow Execution

1. Go to **Actions** tab
2. Monitor workflow runs
3. Check for any failures or warnings
4. Verify secrets are being used correctly

### Test Notifications

1. Trigger a workflow failure (intentionally)
2. Verify Slack notifications
3. Check email notifications (if configured)
4. Validate security alerts

## 9. Monitoring and Maintenance

### Workflow Health Monitoring

Set up monitoring for:
- Workflow success/failure rates
- Execution duration trends
- Resource usage patterns
- Security scan results

### Regular Maintenance Tasks

1. **Weekly**: Review failed workflows
2. **Monthly**: Update workflow dependencies
3. **Quarterly**: Review and update secrets
4. **Annually**: Security audit of workflows

### Performance Optimization

Monitor and optimize:
- Workflow execution time
- Resource usage
- Cache hit rates
- Parallel job efficiency

## 10. Troubleshooting

### Common Issues

#### Permission Errors
```
Error: Resource not accessible by integration
```
**Solution**: Check token permissions and repository settings

#### Secret Access Issues
```
Error: Secret not found
```
**Solution**: Verify secret names and scope (repository vs environment)

#### AWS Deployment Failures
```
Error: Unable to assume role
```
**Solution**: Check IAM role configuration and trust relationships

### Debug Mode

Enable debug logging by adding to workflow:
```yaml
env:
  ACTIONS_STEP_DEBUG: true
  ACTIONS_RUNNER_DEBUG: true
```

### Getting Help

- GitHub Actions [documentation](https://docs.github.com/en/actions)
- GitHub [community forum](https://github.community/)
- Repository issues and discussions
- Team Slack channels

## 11. Security Best Practices

### Secrets Management
- Use environment-specific secrets
- Rotate secrets regularly
- Minimize secret scope
- Audit secret access

### Workflow Security
- Pin action versions
- Use official actions when possible
- Validate external inputs
- Minimize permissions

### Access Control
- Use branch protection rules
- Require pull request reviews
- Enable status checks
- Restrict force pushes

## Next Steps

After completing the setup:

1. **Test all workflows** with sample changes
2. **Train team members** on workflow usage
3. **Document custom procedures** specific to your organization
4. **Set up monitoring** and alerting
5. **Plan regular maintenance** schedules

This completes the GitHub Actions setup for your photonic neuromorphics simulation platform. The workflows will provide comprehensive CI/CD, security scanning, and automation capabilities.