# Runbook: Simulation Performance Issues

## Overview
Troubleshooting guide for photonic neuromorphics simulation performance issues.

## Symptoms
- Simulations taking longer than expected (>5 minutes)
- High memory usage during simulation
- CPU utilization consistently above 90%
- Simulation failures due to resource exhaustion

## Alert Triggers
- `SimulationDurationHigh`: Simulation duration > 300 seconds
- `MemoryUsageHigh`: Memory usage > 8GB
- `CPUUtilizationHigh`: CPU utilization > 90%

## Immediate Actions

### 1. Check System Resources
```bash
# Check current resource usage
htop
free -h
df -h

# Check running simulations
ps aux | grep photonic
docker stats
```

### 2. Review Recent Simulations
```bash
# Check simulation logs
tail -f /var/log/photonic/performance.log

# Query recent simulation metrics
curl "http://localhost:9090/api/v1/query?query=simulation_duration_seconds"
```

### 3. Identify Resource Bottlenecks
```bash
# Check memory usage by component
curl "http://localhost:9090/api/v1/query?query=memory_usage_bytes"

# Check CPU usage patterns
curl "http://localhost:9090/api/v1/query?query=cpu_utilization_percent"
```

## Diagnostic Steps

### Performance Analysis
1. **Check simulation parameters**:
   - Neuron count
   - Simulation duration
   - Model complexity
   - Batch size

2. **Review resource allocation**:
   - Available memory
   - CPU cores
   - GPU availability
   - Storage I/O

3. **Analyze bottlenecks**:
   - Memory allocation patterns
   - CPU-intensive operations
   - I/O wait times
   - Network latency

### Common Causes

#### High Memory Usage
- Large neural network models
- Memory leaks in simulation code
- Insufficient garbage collection
- Large dataset loading

#### CPU Bottlenecks
- Inefficient algorithms
- Single-threaded operations
- Context switching overhead
- Insufficient parallelization

#### I/O Issues
- Large file writes/reads
- Network storage latency
- Disk space exhaustion
- Concurrent access conflicts

## Resolution Steps

### 1. Optimize Simulation Parameters
```python
# Reduce model complexity
model = PhotonicSNN(
    topology=[784, 128, 10],  # Reduced from [784, 256, 128, 10]
    batch_size=32,           # Reduced from 64
    precision='float16'      # Reduced from float32
)

# Enable memory optimization
simulator = PhotonicSimulator(
    memory_optimization=True,
    lazy_loading=True,
    checkpoint_interval=1000
)
```

### 2. Scale Resources
```bash
# Increase container memory limit
docker update --memory=16g photonic-app

# Add CPU resources
docker update --cpus=8 photonic-app

# Scale horizontally
docker-compose up --scale photonic-worker=4
```

### 3. Enable Performance Monitoring
```python
# Add performance profiling
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run simulation
result = run_simulation()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(20)
```

### 4. Optimize Memory Usage
```python
# Use memory mapping for large datasets
import mmap

with open('large_dataset.bin', 'rb') as f:
    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        # Process data without loading into memory
        process_data(mm)

# Enable memory pooling
from photonic_neuromorphics.utils import MemoryPool

memory_pool = MemoryPool(size_gb=4)
with memory_pool:
    run_simulation()
```

### 5. Parallel Processing
```python
# Enable multiprocessing
from multiprocessing import Pool
import numpy as np

def parallel_simulation(neuron_batch):
    return simulate_neurons(neuron_batch)

# Split neurons across processes
neuron_batches = np.array_split(neurons, num_processes)
with Pool(processes=num_processes) as pool:
    results = pool.map(parallel_simulation, neuron_batches)
```

## Prevention

### 1. Performance Testing
```bash
# Run performance benchmarks
python scripts/benchmark_simulation.py \
  --neurons 1000 \
  --duration 100ms \
  --profile

# Monitor baseline performance
python scripts/performance_baseline.py
```

### 2. Resource Monitoring
```yaml
# docker-compose.yml
services:
  photonic-app:
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
        reservations:
          memory: 4G
          cpus: '2'
```

### 3. Code Optimization
- Use vectorized operations (NumPy, PyTorch)
- Implement memory pooling
- Enable just-in-time compilation (Numba)
- Profile code regularly

### 4. Configuration Tuning
```yaml
# config/performance.yaml
simulation:
  max_memory_gb: 6
  cpu_threads: 4
  gpu_enabled: true
  batch_size: 32
  
optimization:
  enable_caching: true
  lazy_loading: true
  memory_mapping: true
```

## Escalation

### Level 1: Automatic Recovery
- Restart simulation with reduced parameters
- Clear memory caches
- Scale resources automatically

### Level 2: Manual Intervention
- Review and optimize simulation code
- Adjust resource allocation
- Implement performance improvements

### Level 3: Engineering Review
- Architecture review for scalability
- Hardware upgrade evaluation
- Algorithm optimization

## Monitoring

### Key Metrics to Watch
- `simulation_duration_seconds` - Simulation execution time
- `memory_usage_bytes` - Memory consumption
- `cpu_utilization_percent` - CPU usage
- `gpu_utilization_percent` - GPU usage (if applicable)

### Performance Dashboards
- System Resource Dashboard
- Simulation Performance Dashboard
- Application Performance Monitoring

### Alerting Thresholds
```yaml
alerts:
  simulation_slow:
    threshold: 300s
    severity: warning
  
  memory_high:
    threshold: 6GB
    severity: critical
  
  cpu_high:
    threshold: 90%
    duration: 5m
    severity: warning
```

## Recovery Procedures

### Graceful Degradation
1. Reduce simulation complexity
2. Enable checkpointing
3. Split large simulations
4. Use approximation algorithms

### Emergency Actions
1. Kill long-running simulations
2. Clear memory caches
3. Restart services
4. Scale down concurrent operations

## Documentation Links
- [Performance Optimization Guide](../guides/performance_optimization.md)
- [Memory Management Best Practices](../guides/memory_management.md)
- [Simulation Configuration Reference](../reference/simulation_config.md)
- [Monitoring Dashboards](../monitoring/dashboards.md)

## Contact Information
- **On-Call Engineer**: oncall@photonic-neuromorphics.com
- **Performance Team**: performance@photonic-neuromorphics.com
- **Slack Channel**: #performance-alerts
- **Escalation**: CTO office