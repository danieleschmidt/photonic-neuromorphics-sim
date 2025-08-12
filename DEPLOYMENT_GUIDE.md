# Production Deployment Guide

## 🚀 Photonic Neuromorphics Simulation Framework - Production Deployment

This guide provides comprehensive instructions for deploying the photonic neuromorphics simulation framework in production environments.

## 📋 Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 4 cores, 2.0 GHz
- RAM: 8 GB
- Storage: 20 GB SSD
- OS: Linux (Ubuntu 20.04+), macOS, or Windows 10+
- Python: 3.9+

**Recommended for Production:**
- CPU: 16+ cores, 3.0+ GHz
- RAM: 32+ GB
- Storage: 100+ GB NVMe SSD
- GPU: NVIDIA RTX 3080+ or equivalent (optional, for acceleration)
- Network: High-speed internet for distributed computing

### Software Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Optional: GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Optional: Additional scientific libraries
pip install cupy-cuda11x  # For CUDA acceleration
pip install ray  # For distributed computing
```

## 🏗️ Installation Options

### Option 1: Local Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/photonic-neuromorphics-sim.git
cd photonic-neuromorphics-sim

# Create virtual environment
python -m venv photonic_env
source photonic_env/bin/activate  # On Windows: photonic_env\Scripts\activate

# Install in development mode
pip install -e .

# Verify installation
python -c "import photonic_neuromorphics; print('✓ Installation successful')"
```

### Option 2: Docker Deployment

```bash
# Build the Docker image
docker build -t photonic-neuromorphics:latest .

# Run with basic configuration
docker run -p 8080:8080 -v $(pwd)/data:/app/data photonic-neuromorphics:latest

# Run with GPU support
docker run --gpus all -p 8080:8080 -v $(pwd)/data:/app/data photonic-neuromorphics:latest
```

### Option 3: Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Scale the deployment
kubectl scale deployment photonic-neuromorphics --replicas=5

# Check status
kubectl get pods -l app=photonic-neuromorphics
```

## ⚙️ Configuration

### Security Configuration

Create `photonic_config.yaml`:

```yaml
# Security settings
enable_input_validation: true
enable_output_sanitization: true
max_simulation_time: 1.0e-6  # 1 microsecond
max_memory_usage: 8589934592  # 8 GB
rate_limit_requests: 10000  # per hour
enable_audit_logging: true
require_authentication: true
session_timeout: 3600  # 1 hour

# Allowed file types
allowed_file_types:
  - .gds
  - .sp
  - .v
  - .sv
  - .json
  - .yaml

# Logging configuration
log_level: INFO
log_retention_days: 30
structured_logging: true
```

### Performance Configuration

Create `scaling_config.yaml`:

```yaml
# Scaling configuration
auto_scaling: true
min_workers: 2
max_workers: 16
scaling_factor: 1.5
load_threshold: 0.8
scale_down_delay: 300.0

# GPU configuration
gpu_enabled: true
gpu_memory_fraction: 0.8

# Distributed computing
distributed_enabled: true
cluster_nodes:
  - host: node1.cluster.local
    port: 8080
    capabilities:
      max_workers: 8
      memory_gb: 32
      gpu_available: true
  - host: node2.cluster.local
    port: 8080
    capabilities:
      max_workers: 16
      memory_gb: 64
      gpu_available: false
```

## 🔧 Environment Setup

### Environment Variables

```bash
# Required
export PHOTONIC_CONFIG_PATH=/path/to/photonic_config.yaml
export PHOTONIC_LOG_LEVEL=INFO
export PHOTONIC_LOG_DIR=/var/log/photonic

# Optional
export PHOTONIC_GPU_ENABLED=true
export PHOTONIC_DISTRIBUTED_ENABLED=true
export PHOTONIC_MAX_WORKERS=16

# Security
export PHOTONIC_SECRET_KEY=your-secret-key-here
export PHOTONIC_AUTH_TOKEN=your-auth-token

# Database (if using persistent storage)
export PHOTONIC_DB_URL=postgresql://user:pass@localhost/photonic_db

# Monitoring
export PROMETHEUS_URL=http://localhost:9090
export GRAFANA_URL=http://localhost:3000
```

### Systemd Service (Linux)

Create `/etc/systemd/system/photonic-neuromorphics.service`:

```ini
[Unit]
Description=Photonic Neuromorphics Simulation Framework
After=network.target

[Service]
Type=simple
User=photonic
Group=photonic
WorkingDirectory=/opt/photonic-neuromorphics
Environment=PHOTONIC_CONFIG_PATH=/etc/photonic/config.yaml
Environment=PHOTONIC_LOG_DIR=/var/log/photonic
ExecStart=/opt/photonic-neuromorphics/venv/bin/python -m photonic_neuromorphics.server
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable photonic-neuromorphics
sudo systemctl start photonic-neuromorphics
sudo systemctl status photonic-neuromorphics
```

## 📊 Monitoring and Observability

### Prometheus Metrics

The framework automatically exposes metrics on `/metrics` endpoint:

- `photonic_simulations_total` - Total number of simulations
- `photonic_simulation_duration_seconds` - Simulation duration histogram
- `photonic_errors_total` - Error count by type
- `photonic_memory_usage_bytes` - Memory usage
- `photonic_worker_pool_size` - Current worker pool size
- `photonic_queue_size` - Work queue size

### Grafana Dashboard

Import the provided dashboard from `monitoring/grafana/dashboards/application-overview.json`:

Key panels:
- Simulation throughput
- Error rates
- Memory and CPU usage
- Response times
- Worker utilization

### Log Aggregation

Logs are structured JSON with correlation IDs:

```json
{
  "timestamp": 1640995200.123,
  "level": "INFO",
  "message": "Simulation completed successfully",
  "correlation_id": "abc123",
  "session_id": "session456",
  "component": "simulation",
  "operation": "run_photonic_snn",
  "duration_ms": 1234.5,
  "metadata": {
    "simulation_type": "multiwavelength",
    "wavelength_channels": 8
  }
}
```

Configure log shipping to your preferred aggregation system (ELK, Loki, etc.).

## 🔒 Security Hardening

### Network Security

```bash
# Configure firewall (Ubuntu/Debian)
sudo ufw allow 8080/tcp  # Application port
sudo ufw allow 9090/tcp  # Metrics port (internal only)
sudo ufw enable

# Use TLS in production
# Configure reverse proxy (nginx/Apache) with SSL certificates
```

### Application Security

1. **Authentication & Authorization**
   ```python
   # Enable authentication in config
   require_authentication: true
   
   # Set strong session timeout
   session_timeout: 1800  # 30 minutes
   ```

2. **Input Validation**
   ```python
   # Strict validation enabled by default
   enable_input_validation: true
   max_simulation_time: 1.0e-6  # Prevent resource exhaustion
   ```

3. **Rate Limiting**
   ```python
   # Prevent abuse
   rate_limit_requests: 1000  # per hour per user
   ```

4. **Audit Logging**
   ```python
   # Track all activities
   enable_audit_logging: true
   audit_log_retention: 365  # days
   ```

## 🚀 Performance Optimization

### CPU Optimization

```bash
# Set CPU affinity for better performance
taskset -c 0-7 python -m photonic_neuromorphics.server

# Enable CPU governor for performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### Memory Optimization

```python
# Configure memory limits
max_memory_usage: 34359738368  # 32 GB
enable_memory_monitoring: true
memory_cleanup_threshold: 0.8
```

### GPU Optimization

```python
# GPU configuration
gpu_memory_fraction: 0.9  # Use 90% of GPU memory
enable_mixed_precision: true
gpu_memory_growth: true  # Dynamic allocation
```

### Distributed Computing

```yaml
# Cluster configuration
distributed_enabled: true
load_balancing_strategy: "round_robin"
heartbeat_interval: 30  # seconds
node_failure_timeout: 120  # seconds
```

## 📈 Scaling Strategies

### Horizontal Scaling

1. **Auto-scaling based on queue size**
   ```python
   auto_scaling: true
   scale_up_threshold: 10  # queue items per worker
   scale_down_threshold: 2
   ```

2. **Kubernetes HPA**
   ```yaml
   apiVersion: autoscaling/v2
   kind: HorizontalPodAutoscaler
   metadata:
     name: photonic-neuromorphics-hpa
   spec:
     scaleTargetRef:
       apiVersion: apps/v1
       kind: Deployment
       name: photonic-neuromorphics
     minReplicas: 3
     maxReplicas: 20
     metrics:
     - type: Resource
       resource:
         name: cpu
         target:
           type: Utilization
           averageUtilization: 70
   ```

### Vertical Scaling

```bash
# Increase memory limits
docker run -m 64g photonic-neuromorphics:latest

# Increase CPU limits
docker run --cpus="16.0" photonic-neuromorphics:latest
```

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory**
   ```bash
   # Check memory usage
   docker stats
   
   # Increase memory limit
   docker run -m 32g photonic-neuromorphics:latest
   ```

2. **GPU Not Detected**
   ```bash
   # Verify GPU support
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Install CUDA drivers
   sudo apt install nvidia-driver-470
   ```

3. **Slow Performance**
   ```bash
   # Check CPU usage
   htop
   
   # Profile application
   python -m cProfile -o profile.stats main.py
   ```

4. **Connection Issues**
   ```bash
   # Check network connectivity
   telnet node1.cluster.local 8080
   
   # Verify firewall rules
   sudo ufw status
   ```

### Debug Mode

```bash
# Enable debug logging
export PHOTONIC_LOG_LEVEL=DEBUG

# Enable profiling
export PHOTONIC_ENABLE_PROFILING=true

# Enable memory tracking
export PHOTONIC_TRACK_MEMORY=true
```

### Log Analysis

```bash
# View recent errors
grep "ERROR\|CRITICAL" /var/log/photonic/photonic_neuromorphics.log | tail -100

# Analyze performance
python scripts/analyze_logs.py --log-file /var/log/photonic/photonic_neuromorphics.log --metric duration
```

## 📚 Best Practices

### Deployment

1. **Use Infrastructure as Code**
   - Terraform for cloud resources
   - Ansible for configuration management
   - GitOps for continuous deployment

2. **Implement Blue-Green Deployment**
   ```bash
   # Deploy to staging
   kubectl apply -f k8s/ --namespace=staging
   
   # Test thoroughly
   ./scripts/integration_tests.sh staging
   
   # Switch traffic
   kubectl patch service photonic-neuromorphics -p '{"spec":{"selector":{"version":"v2"}}}'
   ```

3. **Database Migrations**
   ```bash
   # Run migrations before deployment
   python manage.py migrate
   
   # Backup before major changes
   pg_dump photonic_db > backup_$(date +%Y%m%d).sql
   ```

### Operations

1. **Health Checks**
   ```bash
   # Kubernetes health checks
   livenessProbe:
     httpGet:
       path: /health
       port: 8080
     initialDelaySeconds: 30
     periodSeconds: 10
   ```

2. **Alerting**
   ```yaml
   # Prometheus alerts
   groups:
   - name: photonic-neuromorphics
     rules:
     - alert: HighErrorRate
       expr: rate(photonic_errors_total[5m]) > 0.1
       for: 5m
       annotations:
         summary: High error rate detected
   ```

3. **Backup Strategy**
   ```bash
   # Daily backups
   0 2 * * * /opt/photonic/scripts/backup.sh
   
   # Test recovery procedures monthly
   0 0 1 * * /opt/photonic/scripts/test_recovery.sh
   ```

## 🏭 Production Checklist

### Pre-Deployment

- [ ] Security configuration reviewed
- [ ] Performance testing completed
- [ ] Monitoring configured
- [ ] Backup strategy implemented
- [ ] Documentation updated
- [ ] Team training completed

### Deployment

- [ ] Infrastructure provisioned
- [ ] Application deployed
- [ ] Health checks passing
- [ ] Monitoring active
- [ ] Alerts configured
- [ ] Load testing passed

### Post-Deployment

- [ ] Performance monitoring active
- [ ] Error rates within SLA
- [ ] User acceptance testing
- [ ] Documentation updated
- [ ] Support procedures tested
- [ ] Incident response ready

## 📞 Support

### Getting Help

1. **Documentation**: Check the comprehensive docs in `/docs`
2. **Issues**: Report bugs on GitHub Issues
3. **Discussions**: Community support on GitHub Discussions
4. **Enterprise**: Contact support@terragon.ai for enterprise support

### Maintenance

- **Regular Updates**: Keep dependencies updated monthly
- **Security Patches**: Apply security updates immediately
- **Performance Review**: Monthly performance analysis
- **Capacity Planning**: Quarterly scaling assessment

---

*This deployment guide is maintained by the Terragon Labs team. Last updated: 2025-01-01*