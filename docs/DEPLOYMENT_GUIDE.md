# Photonic Neuromorphics - Production Deployment Guide

## Overview

This guide covers the complete production deployment of the Photonic Neuromorphics Simulation Platform using Kubernetes, Docker, and Terraform on AWS infrastructure.

## Prerequisites

### Required Tools
- Docker Desktop or Docker Engine
- kubectl (v1.28+)
- Terraform (v1.0+)
- AWS CLI v2
- Helm (v3.10+)
- Git

### AWS Requirements
- AWS Account with appropriate permissions
- VPC and subnets configured
- EKS cluster creation permissions
- RDS and ElastiCache permissions
- Route53 DNS management (optional)

## Architecture Overview

The deployment consists of:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CloudFront    │    │  Application    │    │    Database     │
│   (Global CDN)  │────│  Load Balancer  │────│   (PostgreSQL)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                      ┌─────────────────┐
                      │  EKS Cluster    │
                      │  (Kubernetes)   │
                      └─────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
    │   Photonic    │   │   Monitoring  │   │     Cache     │
    │ Neuromorphics │   │  (Prometheus  │   │    (Redis)    │
    │   Pods        │   │   & Grafana)  │   │   Cluster     │
    └───────────────┘   └───────────────┘   └───────────────┘
```

## Deployment Steps

### 1. Infrastructure Setup with Terraform

#### Initialize Terraform
```bash
cd deployment/terraform
terraform init
```

#### Configure Variables
Create `terraform.tfvars`:
```hcl
environment = "production"
project_name = "photonic-neuromorphics"
aws_region = "us-west-2"

# Node group configuration
node_groups = {
  general = {
    instance_types = ["c5.2xlarge"]
    min_size       = 2
    max_size       = 10
    desired_size   = 3
    capacity_type  = "ON_DEMAND"
  }
  compute = {
    instance_types = ["c5.4xlarge"]
    min_size       = 1
    max_size       = 20
    desired_size   = 2
    capacity_type  = "SPOT"
  }
  gpu = {
    instance_types = ["p3.2xlarge"]
    min_size       = 0
    max_size       = 5
    desired_size   = 1
    capacity_type  = "ON_DEMAND"
  }
}

# Database configuration
database_config = {
  instance_class    = "db.r6g.2xlarge"
  allocated_storage = 1000
  engine_version   = "16.1"
  backup_retention  = 30
}

enable_gpu_nodes = true
domain_name = "your-domain.com"
```

#### Deploy Infrastructure
```bash
terraform plan -var-file=terraform.tfvars
terraform apply -var-file=terraform.tfvars
```

### 2. Docker Image Build and Registry

#### Build Production Image
```bash
cd ../..
docker build -f deployment/docker/Dockerfile -t photonic-neuromorphics:latest .
```

#### Tag and Push to Registry
```bash
# For AWS ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-west-2.amazonaws.com

docker tag photonic-neuromorphics:latest <account-id>.dkr.ecr.us-west-2.amazonaws.com/photonic-neuromorphics:latest
docker push <account-id>.dkr.ecr.us-west-2.amazonaws.com/photonic-neuromorphics:latest
```

### 3. Kubernetes Deployment

#### Configure kubectl
```bash
aws eks update-kubeconfig --region us-west-2 --name photonic-neuromorphics-production-cluster
```

#### Create Secrets
```bash
# Database credentials
kubectl create secret generic photonic-secrets \
  --from-literal=postgres-url="postgresql://photonic_admin:<password>@<rds-endpoint>:5432/photonic_neuromorphics" \
  --from-literal=redis-url="redis://<redis-endpoint>:6379" \
  --from-literal=quantum-encryption-key="<your-quantum-key>" \
  -n photonic-neuromorphics
```

#### Deploy Application
```bash
cd deployment/kubernetes

# Create namespace
kubectl apply -f namespace.yaml

# Apply configurations
kubectl apply -f configmap.yaml
kubectl apply -f service.yaml
kubectl apply -f deployment.yaml
```

#### Verify Deployment
```bash
kubectl get pods -n photonic-neuromorphics
kubectl get services -n photonic-neuromorphics
kubectl logs -f deployment/photonic-sim-deployment -n photonic-neuromorphics
```

### 4. Monitoring Setup

#### Deploy Prometheus
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --values deployment/monitoring/prometheus-values.yaml
```

#### Configure Grafana Dashboard
```bash
# Import dashboard
kubectl create configmap grafana-dashboard \
  --from-file=deployment/monitoring/grafana-dashboard.json \
  -n monitoring
```

### 5. SSL Certificate Setup

#### Using AWS Certificate Manager
```bash
aws acm request-certificate \
  --domain-name your-domain.com \
  --subject-alternative-names *.your-domain.com \
  --validation-method DNS \
  --region us-west-2
```

#### Configure Route53 DNS (if using)
```bash
# Get load balancer DNS name
kubectl get service photonic-sim-service -n photonic-neuromorphics -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'

# Create CNAME record in Route53 pointing to load balancer
```

## Environment-Specific Configurations

### Production Environment Variables
```bash
PHOTONIC_ENV=production
PHOTONIC_LOG_LEVEL=INFO
PHOTONIC_REDIS_URL=redis://production-redis:6379
PHOTONIC_POSTGRES_URL=postgresql://user:pass@production-db:5432/photonic_db
PHOTONIC_QUANTUM_SECURITY=enabled
PHOTONIC_CACHE_TTL=3600
PHOTONIC_METRICS_ENABLED=true
```

### Staging Environment Variables
```bash
PHOTONIC_ENV=staging
PHOTONIC_LOG_LEVEL=DEBUG
PHOTONIC_REDIS_URL=redis://staging-redis:6379
PHOTONIC_POSTGRES_URL=postgresql://user:pass@staging-db:5432/photonic_db_staging
PHOTONIC_QUANTUM_SECURITY=enabled
PHOTONIC_CACHE_TTL=1800
PHOTONIC_METRICS_ENABLED=true
```

## Health Checks and Monitoring

### Application Health Endpoints
- `/health` - Basic health check
- `/ready` - Readiness probe
- `/startup` - Startup probe
- `/metrics` - Prometheus metrics

### Monitoring Dashboards
- Grafana: `http://<grafana-url>:3000`
- Prometheus: `http://<prometheus-url>:9090`
- Application Metrics: `http://<app-url>/metrics`

### Key Metrics to Monitor
- Request rate and response time
- Quantum processing performance
- Photonic simulation metrics
- Database connection pool
- Cache hit rates
- Error rates and security events

## Scaling and Performance

### Horizontal Pod Autoscaler
```bash
kubectl autoscale deployment photonic-sim-deployment \
  --cpu-percent=70 \
  --min=3 \
  --max=20 \
  -n photonic-neuromorphics
```

### Cluster Autoscaler
The EKS cluster is configured with auto-scaling groups that will automatically add/remove nodes based on demand.

### Performance Tuning
- Adjust resource requests/limits based on actual usage
- Configure appropriate JVM heap sizes for Java components
- Optimize database connection pool settings
- Tune Redis cache eviction policies

## Security Considerations

### Network Security
- All traffic encrypted in transit (TLS 1.2+)
- VPC with private subnets for application components
- Security groups restricting access to necessary ports only
- WAF rules for application layer protection

### Data Security
- Database encryption at rest and in transit
- Redis AUTH and TLS encryption
- Kubernetes secrets for sensitive configuration
- Regular security scans and vulnerability assessments

### Access Control
- RBAC configured for Kubernetes access
- IAM roles with least privilege access
- Multi-factor authentication for administrative access
- Audit logging enabled for all API calls

## Backup and Disaster Recovery

### Database Backups
- Automated daily backups with 30-day retention
- Point-in-time recovery enabled
- Cross-region backup replication for critical data

### Application State
- Persistent volumes for stateful components
- Regular snapshots of EBS volumes
- Configuration stored in version control

### Disaster Recovery Plan
1. Infrastructure as Code enables rapid rebuilding
2. Database restoration from automated backups
3. Application deployment from container registry
4. DNS failover to secondary region (if configured)

## Troubleshooting

### Common Issues

#### Pod Startup Failures
```bash
kubectl describe pod <pod-name> -n photonic-neuromorphics
kubectl logs <pod-name> -n photonic-neuromorphics
```

#### Database Connection Issues
```bash
# Test database connectivity
kubectl exec -it <pod-name> -n photonic-neuromorphics -- psql $PHOTONIC_POSTGRES_URL -c "SELECT 1;"
```

#### Performance Issues
```bash
# Check resource usage
kubectl top pods -n photonic-neuromorphics
kubectl top nodes

# Check application metrics
curl http://<app-url>/metrics
```

### Log Analysis
```bash
# Application logs
kubectl logs -f deployment/photonic-sim-deployment -n photonic-neuromorphics

# System logs
journalctl -u kubelet -f

# Database logs (if accessible)
aws rds describe-db-log-files --db-instance-identifier photonic-neuromorphics-db
```

## Maintenance and Updates

### Application Updates
1. Build new container image
2. Push to registry with new tag
3. Update Kubernetes deployment
4. Monitor rollout and health checks

### Infrastructure Updates
1. Update Terraform configurations
2. Plan and apply changes
3. Validate infrastructure health
4. Update monitoring and alerting

### Security Updates
1. Regularly scan container images
2. Apply OS and package updates
3. Rotate credentials and certificates
4. Review and update security policies

## Cost Optimization

### Resource Right-Sizing
- Monitor actual resource usage vs. requests/limits
- Adjust instance types based on workload patterns
- Use Spot instances for batch processing workloads

### Storage Optimization
- Implement lifecycle policies for S3 storage
- Use appropriate storage classes for different data types
- Regular cleanup of old logs and temporary files

### Monitoring Costs
- Set up billing alerts and budgets
- Use AWS Cost Explorer to analyze spending
- Tag resources for cost allocation tracking

## Support and Contact

For deployment issues or questions:
- Create GitHub issue: https://github.com/danieleschmidt/photonic-neuromorphics-sim/issues
- Email: support@terragon-labs.ai
- Documentation: https://docs.photonic-neuromorphics.ai

## Next Steps

After successful deployment:
1. Set up CI/CD pipelines for automated deployments
2. Implement comprehensive monitoring and alerting
3. Configure backup and disaster recovery procedures
4. Plan for capacity scaling based on usage patterns
5. Set up regular security audits and compliance checks