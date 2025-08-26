# Photonic Neuromorphics API Documentation

## Overview

The Photonic Neuromorphics Simulation Platform provides a comprehensive REST API for quantum-photonic neural network operations, including temporal entanglement, metamaterial simulation, and enterprise-grade security features.

**Base URL**: `https://api.photonic-neuromorphics.ai/v1`  
**Version**: 1.0.0  
**Authentication**: JWT Bearer Token with Quantum Key Distribution  

## Authentication

### JWT Token Authentication
```http
Authorization: Bearer <jwt_token>
```

### Quantum Key Distribution (QKD)
For enhanced security, quantum keys can be exchanged:
```http
X-Quantum-Key: <quantum_key_id>
X-Quantum-Signature: <quantum_signature>
```

### API Key Authentication
```http
X-API-Key: <your_api_key>
```

## Core Endpoints

### 1. Quantum Temporal Entanglement

#### Process Quantum Temporal Entanglement
```http
POST /quantum/temporal/entanglement
```

**Request Body:**
```json
{
  "spike_train": [[0.1, 0.2, 0.0], [0.3, 0.1, 0.4]],
  "temporal_window": 0.001,
  "entanglement_depth": 10,
  "coherence_time": 0.1,
  "fidelity_threshold": 0.95
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "entanglement_id": "ent_7f8a9b2c3d4e5f6g",
    "quantum_states": [
      {
        "amplitude": 0.707,
        "phase": 1.57,
        "entanglement_measure": 0.89
      }
    ],
    "temporal_correlations": {
      "past_correlations": [[0.23, 0.45], [0.12, 0.67]],
      "future_predictions": [[0.34, 0.56], [0.78, 0.23]]
    },
    "processing_time": 0.045,
    "quantum_fidelity": 0.97
  }
}
```

#### Get Quantum State
```http
GET /quantum/temporal/entanglement/{entanglement_id}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "entanglement_id": "ent_7f8a9b2c3d4e5f6g",
    "state": "entangled",
    "coherence_time_remaining": 0.089,
    "entanglement_strength": 0.92,
    "last_measurement": "2024-01-15T14:30:25Z"
  }
}
```

### 2. Neuromorphic Photonic Metamaterials

#### Create Metamaterial Structure
```http
POST /neuromorphic/metamaterials
```

**Request Body:**
```json
{
  "name": "adaptive_neural_metamaterial_v1",
  "base_refractive_index": 2.4,
  "neural_activity_threshold": 0.5,
  "adaptation_rate": 0.1,
  "structure_type": "honeycomb",
  "dimensions": {
    "width": 100e-6,
    "height": 100e-6,
    "depth": 10e-6
  },
  "optimization_target": "spike_rate_enhancement"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "metamaterial_id": "meta_9a8b7c6d5e4f3g2h",
    "structure": {
      "unit_cells": 10000,
      "effective_index": 2.47,
      "resonant_wavelength": 1550e-9,
      "q_factor": 1247.5
    },
    "adaptation_parameters": {
      "response_time": 1.2e-12,
      "adaptation_efficiency": 0.87,
      "power_consumption": 2.3e-15
    },
    "simulation_id": "sim_1a2b3c4d5e6f7g8h"
  }
}
```

#### Update Metamaterial Configuration
```http
PUT /neuromorphic/metamaterials/{metamaterial_id}
```

**Request Body:**
```json
{
  "neural_activity": [[0.1, 0.3, 0.7], [0.2, 0.9, 0.1]],
  "environmental_conditions": {
    "temperature": 300,
    "optical_power": 1e-3
  }
}
```

### 3. Enterprise Quantum Security

#### Initialize Quantum Key Distribution
```http
POST /security/quantum/qkd/initialize
```

**Request Body:**
```json
{
  "protocol": "BB84",
  "key_length": 256,
  "error_correction": "cascade",
  "privacy_amplification": true
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "session_id": "qkd_session_123456",
    "alice_basis": "random_basis_alice_789",
    "public_parameters": {
      "polarization_states": ["H", "V", "D", "A"],
      "detection_efficiency": 0.8,
      "error_rate_threshold": 0.11
    },
    "expires_at": "2024-01-15T15:30:25Z"
  }
}
```

#### Detect Security Threats
```http
POST /security/threats/analyze
```

**Request Body:**
```json
{
  "network_traffic": {
    "source_ip": "192.168.1.100",
    "destination_ip": "10.0.0.50",
    "packet_size": 1500,
    "protocol": "TCP",
    "timestamp": "2024-01-15T14:25:30Z"
  },
  "behavioral_patterns": ["unusual_request_rate", "anomalous_data_access"],
  "quantum_signatures": ["entanglement_anomaly", "coherence_breach"]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "threat_assessment": {
      "risk_level": "medium",
      "confidence": 0.84,
      "threat_types": ["quantum_eavesdropping", "side_channel_attack"],
      "mitigation_recommended": true
    },
    "quantum_integrity": {
      "key_security": "intact",
      "entanglement_preserved": true,
      "decoherence_detected": false
    },
    "response_actions": [
      "rotate_quantum_keys",
      "increase_monitoring_frequency",
      "alert_security_team"
    ]
  }
}
```

### 4. Real-time Adaptive Monitoring

#### Submit System Metrics
```http
POST /monitoring/metrics
```

**Request Body:**
```json
{
  "timestamp": "2024-01-15T14:30:00Z",
  "metrics": {
    "cpu_usage": 0.65,
    "memory_usage": 0.78,
    "network_latency": 0.012,
    "quantum_fidelity": 0.94,
    "photonic_efficiency": 0.87
  },
  "component": "quantum_processor_unit_1",
  "environment": "production"
}
```

#### Get Predictive Analytics
```http
GET /monitoring/predictions/{component_id}
```

**Query Parameters:**
- `horizon`: Time horizon for predictions (default: 3600 seconds)
- `confidence`: Confidence interval (default: 0.95)

**Response:**
```json
{
  "success": true,
  "data": {
    "component_id": "quantum_processor_unit_1",
    "predictions": {
      "cpu_usage": {
        "predicted_values": [0.72, 0.68, 0.75, 0.81],
        "confidence_intervals": [[0.65, 0.79], [0.61, 0.75]],
        "anomaly_probability": 0.15
      },
      "quantum_fidelity": {
        "predicted_values": [0.92, 0.89, 0.91, 0.88],
        "trend": "declining",
        "maintenance_recommended": true
      }
    },
    "alerts": [
      {
        "type": "performance_degradation",
        "severity": "warning",
        "message": "Quantum fidelity trending downward",
        "eta": "2024-01-15T16:45:00Z"
      }
    ]
  }
}
```

### 5. Quantum Accelerated Optimization

#### Run QAOA Optimization
```http
POST /optimization/qaoa
```

**Request Body:**
```json
{
  "problem_type": "max_cut",
  "graph": {
    "vertices": 10,
    "edges": [[0, 1, 0.5], [1, 2, 0.8], [2, 3, 0.3]]
  },
  "optimization_params": {
    "layers": 5,
    "max_iterations": 100,
    "convergence_threshold": 1e-6
  },
  "quantum_backend": "quantum_simulator"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "optimization_id": "opt_qaoa_789abc",
    "optimal_solution": {
      "parameters": [1.23, 0.89, 1.45, 0.67, 1.12],
      "objective_value": -4.67,
      "probability": 0.87
    },
    "execution_stats": {
      "iterations": 67,
      "convergence_achieved": true,
      "quantum_shots": 10000,
      "execution_time": 12.45
    },
    "quantum_metrics": {
      "fidelity": 0.94,
      "gate_count": 245,
      "depth": 67
    }
  }
}
```

#### Run VQE Optimization
```http
POST /optimization/vqe
```

**Request Body:**
```json
{
  "hamiltonian": {
    "terms": [
      {"pauli": "ZZII", "coefficient": 0.5},
      {"pauli": "ZIZI", "coefficient": 0.3},
      {"pauli": "IIZZ", "coefficient": -0.2}
    ]
  },
  "ansatz": "hardware_efficient",
  "optimizer": "COBYLA",
  "convergence_threshold": 1e-8
}
```

### 6. Ultra High Performance Caching

#### Cache Data
```http
POST /cache/store
```

**Request Body:**
```json
{
  "key": "quantum_state_snapshot_001",
  "data": {
    "quantum_amplitudes": [0.707, 0.0, 0.707, 0.0],
    "phase_information": [0, 0, 1.57, 0],
    "timestamp": "2024-01-15T14:30:00Z"
  },
  "metadata": {
    "experiment_id": "exp_123456",
    "fidelity": 0.95,
    "coherence_time": 0.1
  },
  "ttl": 3600,
  "cache_level": "L1"
}
```

#### Retrieve Cached Data
```http
GET /cache/{key}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "key": "quantum_state_snapshot_001",
    "data": {
      "quantum_amplitudes": [0.707, 0.0, 0.707, 0.0],
      "phase_information": [0, 0, 1.57, 0],
      "timestamp": "2024-01-15T14:30:00Z"
    },
    "metadata": {
      "cache_level": "L1",
      "hit_count": 47,
      "last_accessed": "2024-01-15T14:35:12Z",
      "expiry": "2024-01-15T15:30:00Z"
    },
    "performance": {
      "retrieval_time": 0.002,
      "compression_ratio": 0.34
    }
  }
}
```

## Health and Status Endpoints

### System Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T14:30:00Z",
  "version": "1.0.0",
  "components": {
    "quantum_processor": "healthy",
    "photonic_simulator": "healthy",
    "database": "healthy",
    "cache": "healthy",
    "security_module": "healthy"
  },
  "metrics": {
    "uptime": 86400,
    "total_requests": 1234567,
    "average_response_time": 0.045,
    "quantum_fidelity": 0.94
  }
}
```

### Readiness Check
```http
GET /ready
```

### Prometheus Metrics
```http
GET /metrics
```

## Error Handling

### Error Response Format
```json
{
  "success": false,
  "error": {
    "code": "QUANTUM_DECOHERENCE_ERROR",
    "message": "Quantum state coherence lost during processing",
    "details": {
      "coherence_time_remaining": 0.001,
      "decoherence_source": "thermal_noise",
      "suggested_action": "increase_isolation"
    },
    "timestamp": "2024-01-15T14:30:00Z",
    "request_id": "req_123abc456def"
  }
}
```

### Common Error Codes
- `AUTHENTICATION_FAILED`: Invalid or expired authentication token
- `QUANTUM_DECOHERENCE_ERROR`: Quantum state coherence lost
- `PHOTONIC_SIMULATION_FAILED`: Photonic simulation encountered error
- `CACHE_MISS`: Requested data not found in cache
- `RATE_LIMIT_EXCEEDED`: API rate limit exceeded
- `VALIDATION_ERROR`: Request validation failed
- `SECURITY_THREAT_DETECTED`: Security threat identified
- `OPTIMIZATION_TIMEOUT`: Optimization algorithm timed out

## Rate Limiting

API requests are rate-limited per authentication token:
- **Free Tier**: 100 requests/minute, 1000 requests/hour
- **Professional**: 1000 requests/minute, 10000 requests/hour  
- **Enterprise**: 10000 requests/minute, unlimited hourly

Rate limit headers:
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642251600
```

## SDKs and Libraries

### Python SDK
```bash
pip install photonic-neuromorphics-sdk
```

```python
from photonic_neuromorphics import PhotonicClient

client = PhotonicClient(api_key="your_api_key")

# Quantum temporal entanglement
result = client.quantum.process_entanglement(
    spike_train=[[0.1, 0.2], [0.3, 0.1]],
    temporal_window=0.001
)

print(f"Entanglement ID: {result.entanglement_id}")
print(f"Quantum Fidelity: {result.quantum_fidelity}")
```

### JavaScript SDK
```bash
npm install photonic-neuromorphics-js
```

```javascript
const { PhotonicClient } = require('photonic-neuromorphics-js');

const client = new PhotonicClient({ apiKey: 'your_api_key' });

// Neuromorphic metamaterials
const metamaterial = await client.neuromorphic.createMetamaterial({
  name: 'adaptive_neural_metamaterial_v1',
  baseRefractiveIndex: 2.4,
  neuralActivityThreshold: 0.5
});

console.log(`Metamaterial ID: ${metamaterial.metamaterialId}`);
```

## Webhooks

### Webhook Configuration
```http
POST /webhooks
```

**Request Body:**
```json
{
  "url": "https://your-app.com/webhooks/photonic",
  "events": [
    "quantum.entanglement.created",
    "security.threat.detected",
    "optimization.completed"
  ],
  "secret": "webhook_secret_key"
}
```

### Webhook Payload Example
```json
{
  "event": "quantum.entanglement.created",
  "timestamp": "2024-01-15T14:30:00Z",
  "data": {
    "entanglement_id": "ent_7f8a9b2c3d4e5f6g",
    "quantum_fidelity": 0.97,
    "processing_time": 0.045
  },
  "signature": "sha256=d2f3e4a5b6c7d8e9f0a1b2c3d4e5f6"
}
```

## Support and Resources

- **Documentation**: https://docs.photonic-neuromorphics.ai
- **API Explorer**: https://api.photonic-neuromorphics.ai/explorer
- **Status Page**: https://status.photonic-neuromorphics.ai
- **Support**: support@terragon-labs.ai
- **GitHub**: https://github.com/danieleschmidt/photonic-neuromorphics-sim

## Changelog

### Version 1.0.0 (2024-01-15)
- Initial API release
- Quantum temporal entanglement endpoints
- Neuromorphic photonic metamaterials
- Enterprise quantum security
- Real-time adaptive monitoring
- Quantum accelerated optimization
- Ultra high performance caching