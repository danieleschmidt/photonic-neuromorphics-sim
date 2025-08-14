"""
Production Deployment Suite for Photonic Neuromorphic Systems

Comprehensive production deployment configuration including containerization,
orchestration, monitoring, security, and scalability for enterprise environments.
"""

import os
import json
import yaml
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class DeploymentEnvironment(Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DISASTER_RECOVERY = "disaster_recovery"


class ScalingStrategy(Enum):
    """Auto-scaling strategies."""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    ELASTIC = "elastic"
    PREDICTIVE = "predictive"


@dataclass
class ResourceRequirements:
    """Resource requirements for deployment."""
    cpu_cores: int = 4
    memory_gb: int = 8
    storage_gb: int = 100
    gpu_count: int = 0
    network_bandwidth_mbps: int = 1000
    
    # Photonic-specific requirements
    optical_channels: int = 8
    wavelength_range_nm: tuple = (1500, 1600)
    power_budget_mw: int = 100


@dataclass
class SecurityConfiguration:
    """Security configuration for deployment."""
    enable_tls: bool = True
    tls_version: str = "1.3"
    enable_encryption_at_rest: bool = True
    enable_network_policies: bool = True
    enable_rbac: bool = True
    enable_pod_security: bool = True
    enable_audit_logging: bool = True
    secrets_backend: str = "kubernetes"
    
    # Photonic-specific security
    optical_isolation: bool = True
    quantum_key_distribution: bool = False
    secure_multiparty_computation: bool = False


@dataclass
class MonitoringConfiguration:
    """Monitoring and observability configuration."""
    enable_metrics: bool = True
    enable_logging: bool = True
    enable_tracing: bool = True
    enable_alerts: bool = True
    
    metrics_retention_days: int = 30
    logs_retention_days: int = 90
    traces_retention_days: int = 7
    
    # Alert thresholds
    cpu_threshold_percent: int = 80
    memory_threshold_percent: int = 85
    storage_threshold_percent: int = 90
    error_rate_threshold_percent: int = 5
    
    # Photonic-specific monitoring
    optical_power_monitoring: bool = True
    wavelength_drift_monitoring: bool = True
    quantum_decoherence_monitoring: bool = True


class ProductionDeploymentGenerator:
    """Generates production deployment configurations."""
    
    def __init__(self, environment: DeploymentEnvironment):
        self.environment = environment
        self.base_config = self._get_base_configuration()
    
    def _get_base_configuration(self) -> Dict[str, Any]:
        """Get base configuration for environment."""
        configs = {
            DeploymentEnvironment.DEVELOPMENT: {
                'replicas': 1,
                'resources': ResourceRequirements(cpu_cores=2, memory_gb=4, storage_gb=50),
                'security': SecurityConfiguration(enable_tls=False, enable_rbac=False),
                'monitoring': MonitoringConfiguration(enable_tracing=False)
            },
            DeploymentEnvironment.STAGING: {
                'replicas': 2,
                'resources': ResourceRequirements(cpu_cores=4, memory_gb=8, storage_gb=100),
                'security': SecurityConfiguration(),
                'monitoring': MonitoringConfiguration()
            },
            DeploymentEnvironment.PRODUCTION: {
                'replicas': 3,
                'resources': ResourceRequirements(cpu_cores=8, memory_gb=16, storage_gb=200),
                'security': SecurityConfiguration(enable_encryption_at_rest=True, enable_audit_logging=True),
                'monitoring': MonitoringConfiguration(enable_alerts=True)
            },
            DeploymentEnvironment.DISASTER_RECOVERY: {
                'replicas': 2,
                'resources': ResourceRequirements(cpu_cores=4, memory_gb=8, storage_gb=200),
                'security': SecurityConfiguration(enable_encryption_at_rest=True),
                'monitoring': MonitoringConfiguration(metrics_retention_days=90)
            }
        }
        
        return configs[self.environment]
    
    def generate_kubernetes_manifests(self) -> Dict[str, str]:
        """Generate Kubernetes deployment manifests."""
        manifests = {}
        
        # Namespace
        manifests['namespace.yaml'] = self._generate_namespace()
        
        # ConfigMap
        manifests['configmap.yaml'] = self._generate_configmap()
        
        # Secret
        manifests['secret.yaml'] = self._generate_secret()
        
        # Deployment
        manifests['deployment.yaml'] = self._generate_deployment()
        
        # Service
        manifests['service.yaml'] = self._generate_service()
        
        # Ingress
        manifests['ingress.yaml'] = self._generate_ingress()
        
        # HorizontalPodAutoscaler
        manifests['hpa.yaml'] = self._generate_hpa()
        
        # NetworkPolicy
        manifests['networkpolicy.yaml'] = self._generate_network_policy()
        
        # ServiceMonitor (Prometheus)
        manifests['servicemonitor.yaml'] = self._generate_service_monitor()
        
        return manifests
    
    def _generate_namespace(self) -> str:
        """Generate Kubernetes namespace manifest."""
        namespace = {
            'apiVersion': 'v1',
            'kind': 'Namespace',
            'metadata': {
                'name': f'photonic-neuromorphics-{self.environment.value}',
                'labels': {
                    'app': 'photonic-neuromorphics',
                    'environment': self.environment.value,
                    'version': 'v1.0.0'
                }
            }
        }
        
        return yaml.dump(namespace, default_flow_style=False)
    
    def _generate_configmap(self) -> str:
        """Generate Kubernetes ConfigMap manifest."""
        config_data = {
            'photonic_config.yaml': yaml.dump({
                'photonic_system': {
                    'wavelength_channels': self.base_config['resources'].optical_channels,
                    'wavelength_range': {
                        'min_nm': self.base_config['resources'].wavelength_range_nm[0],
                        'max_nm': self.base_config['resources'].wavelength_range_nm[1]
                    },
                    'power_budget_mw': self.base_config['resources'].power_budget_mw,
                    'optimization_level': 2 if self.environment == DeploymentEnvironment.PRODUCTION else 1
                },
                'neural_network': {
                    'default_layers': [784, 256, 128, 10],
                    'activation_function': 'mach_zehnder',
                    'learning_rate': 0.001,
                    'batch_size': 32
                },
                'monitoring': {
                    'metrics_interval_seconds': 10,
                    'health_check_interval_seconds': 30,
                    'log_level': 'INFO' if self.environment == DeploymentEnvironment.PRODUCTION else 'DEBUG'
                }
            }),
            'prometheus.yml': yaml.dump({
                'global': {
                    'scrape_interval': '15s',
                    'evaluation_interval': '15s'
                },
                'scrape_configs': [
                    {
                        'job_name': 'photonic-neuromorphics',
                        'static_configs': [
                            {'targets': ['photonic-neuromorphics-service:8080']}
                        ]
                    }
                ]
            })
        }
        
        configmap = {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': 'photonic-neuromorphics-config',
                'namespace': f'photonic-neuromorphics-{self.environment.value}'
            },
            'data': config_data
        }
        
        return yaml.dump(configmap, default_flow_style=False)
    
    def _generate_secret(self) -> str:
        """Generate Kubernetes Secret manifest."""
        import base64
        
        # Encode secrets
        secrets = {
            'api_key': base64.b64encode(b'photonic-api-key-placeholder').decode(),
            'database_password': base64.b64encode(b'secure-database-password').decode(),
            'jwt_secret': base64.b64encode(b'jwt-signing-secret-key').decode(),
            'optical_calibration_key': base64.b64encode(b'optical-system-calibration-key').decode()
        }
        
        secret = {
            'apiVersion': 'v1',
            'kind': 'Secret',
            'metadata': {
                'name': 'photonic-neuromorphics-secrets',
                'namespace': f'photonic-neuromorphics-{self.environment.value}'
            },
            'type': 'Opaque',
            'data': secrets
        }
        
        return yaml.dump(secret, default_flow_style=False)
    
    def _generate_deployment(self) -> str:
        """Generate Kubernetes Deployment manifest."""
        resources = self.base_config['resources']
        security = self.base_config['security']
        
        deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': 'photonic-neuromorphics',
                'namespace': f'photonic-neuromorphics-{self.environment.value}',
                'labels': {
                    'app': 'photonic-neuromorphics',
                    'environment': self.environment.value
                }
            },
            'spec': {
                'replicas': self.base_config['replicas'],
                'selector': {
                    'matchLabels': {
                        'app': 'photonic-neuromorphics'
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': 'photonic-neuromorphics',
                            'environment': self.environment.value
                        }
                    },
                    'spec': {
                        'containers': [
                            {
                                'name': 'photonic-neuromorphics',
                                'image': 'photonic-neuromorphics:latest',
                                'ports': [
                                    {'containerPort': 8080, 'name': 'http'},
                                    {'containerPort': 8090, 'name': 'metrics'},
                                    {'containerPort': 9000, 'name': 'optical'}
                                ],
                                'env': [
                                    {'name': 'ENVIRONMENT', 'value': self.environment.value},
                                    {'name': 'LOG_LEVEL', 'value': 'INFO'},
                                    {'name': 'PYTHONPATH', 'value': '/app/src'},
                                    {
                                        'name': 'API_KEY',
                                        'valueFrom': {
                                            'secretKeyRef': {
                                                'name': 'photonic-neuromorphics-secrets',
                                                'key': 'api_key'
                                            }
                                        }
                                    }
                                ],
                                'resources': {
                                    'requests': {
                                        'cpu': f'{resources.cpu_cores // 2}',
                                        'memory': f'{resources.memory_gb // 2}Gi'
                                    },
                                    'limits': {
                                        'cpu': f'{resources.cpu_cores}',
                                        'memory': f'{resources.memory_gb}Gi'
                                    }
                                },
                                'volumeMounts': [
                                    {
                                        'name': 'config-volume',
                                        'mountPath': '/app/config'
                                    },
                                    {
                                        'name': 'data-volume',
                                        'mountPath': '/app/data'
                                    }
                                ],
                                'livenessProbe': {
                                    'httpGet': {
                                        'path': '/health',
                                        'port': 8080
                                    },
                                    'initialDelaySeconds': 30,
                                    'periodSeconds': 10
                                },
                                'readinessProbe': {
                                    'httpGet': {
                                        'path': '/ready',
                                        'port': 8080
                                    },
                                    'initialDelaySeconds': 10,
                                    'periodSeconds': 5
                                }
                            }
                        ],
                        'volumes': [
                            {
                                'name': 'config-volume',
                                'configMap': {
                                    'name': 'photonic-neuromorphics-config'
                                }
                            },
                            {
                                'name': 'data-volume',
                                'persistentVolumeClaim': {
                                    'claimName': 'photonic-neuromorphics-pvc'
                                }
                            }
                        ],
                        'securityContext': {
                            'runAsNonRoot': True,
                            'runAsUser': 1000,
                            'fsGroup': 1000
                        } if security.enable_pod_security else {}
                    }
                }
            }
        }
        
        return yaml.dump(deployment, default_flow_style=False)
    
    def _generate_service(self) -> str:
        """Generate Kubernetes Service manifest."""
        service = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': 'photonic-neuromorphics-service',
                'namespace': f'photonic-neuromorphics-{self.environment.value}',
                'labels': {
                    'app': 'photonic-neuromorphics'
                }
            },
            'spec': {
                'selector': {
                    'app': 'photonic-neuromorphics'
                },
                'ports': [
                    {
                        'name': 'http',
                        'port': 80,
                        'targetPort': 8080,
                        'protocol': 'TCP'
                    },
                    {
                        'name': 'metrics',
                        'port': 8090,
                        'targetPort': 8090,
                        'protocol': 'TCP'
                    },
                    {
                        'name': 'optical',
                        'port': 9000,
                        'targetPort': 9000,
                        'protocol': 'TCP'
                    }
                ],
                'type': 'ClusterIP'
            }
        }
        
        return yaml.dump(service, default_flow_style=False)
    
    def _generate_ingress(self) -> str:
        """Generate Kubernetes Ingress manifest."""
        security = self.base_config['security']
        
        ingress = {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'Ingress',
            'metadata': {
                'name': 'photonic-neuromorphics-ingress',
                'namespace': f'photonic-neuromorphics-{self.environment.value}',
                'annotations': {
                    'kubernetes.io/ingress.class': 'nginx',
                    'cert-manager.io/cluster-issuer': 'letsencrypt-prod' if security.enable_tls else '',
                    'nginx.ingress.kubernetes.io/rate-limit': '100',
                    'nginx.ingress.kubernetes.io/rate-limit-window': '1m'
                }
            },
            'spec': {
                'rules': [
                    {
                        'host': f'photonic-{self.environment.value}.example.com',
                        'http': {
                            'paths': [
                                {
                                    'path': '/',
                                    'pathType': 'Prefix',
                                    'backend': {
                                        'service': {
                                            'name': 'photonic-neuromorphics-service',
                                            'port': {
                                                'number': 80
                                            }
                                        }
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }
        
        if security.enable_tls:
            ingress['spec']['tls'] = [
                {
                    'hosts': [f'photonic-{self.environment.value}.example.com'],
                    'secretName': 'photonic-neuromorphics-tls'
                }
            ]
        
        return yaml.dump(ingress, default_flow_style=False)
    
    def _generate_hpa(self) -> str:
        """Generate Horizontal Pod Autoscaler manifest."""
        hpa = {
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': 'photonic-neuromorphics-hpa',
                'namespace': f'photonic-neuromorphics-{self.environment.value}'
            },
            'spec': {
                'scaleTargetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': 'photonic-neuromorphics'
                },
                'minReplicas': self.base_config['replicas'],
                'maxReplicas': self.base_config['replicas'] * 3,
                'metrics': [
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'cpu',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': 70
                            }
                        }
                    },
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'memory',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': 80
                            }
                        }
                    }
                ]
            }
        }
        
        return yaml.dump(hpa, default_flow_style=False)
    
    def _generate_network_policy(self) -> str:
        """Generate Kubernetes NetworkPolicy manifest."""
        network_policy = {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'NetworkPolicy',
            'metadata': {
                'name': 'photonic-neuromorphics-netpol',
                'namespace': f'photonic-neuromorphics-{self.environment.value}'
            },
            'spec': {
                'podSelector': {
                    'matchLabels': {
                        'app': 'photonic-neuromorphics'
                    }
                },
                'policyTypes': ['Ingress', 'Egress'],
                'ingress': [
                    {
                        'from': [
                            {
                                'namespaceSelector': {
                                    'matchLabels': {
                                        'name': 'ingress-nginx'
                                    }
                                }
                            }
                        ],
                        'ports': [
                            {'protocol': 'TCP', 'port': 8080}
                        ]
                    },
                    {
                        'from': [
                            {
                                'namespaceSelector': {
                                    'matchLabels': {
                                        'name': 'monitoring'
                                    }
                                }
                            }
                        ],
                        'ports': [
                            {'protocol': 'TCP', 'port': 8090}
                        ]
                    }
                ],
                'egress': [
                    {
                        'to': [],
                        'ports': [
                            {'protocol': 'TCP', 'port': 53},
                            {'protocol': 'UDP', 'port': 53},
                            {'protocol': 'TCP', 'port': 443},
                            {'protocol': 'TCP', 'port': 80}
                        ]
                    }
                ]
            }
        }
        
        return yaml.dump(network_policy, default_flow_style=False)
    
    def _generate_service_monitor(self) -> str:
        """Generate Prometheus ServiceMonitor manifest."""
        service_monitor = {
            'apiVersion': 'monitoring.coreos.com/v1',
            'kind': 'ServiceMonitor',
            'metadata': {
                'name': 'photonic-neuromorphics-monitor',
                'namespace': f'photonic-neuromorphics-{self.environment.value}',
                'labels': {
                    'app': 'photonic-neuromorphics'
                }
            },
            'spec': {
                'selector': {
                    'matchLabels': {
                        'app': 'photonic-neuromorphics'
                    }
                },
                'endpoints': [
                    {
                        'port': 'metrics',
                        'interval': '30s',
                        'path': '/metrics'
                    }
                ]
            }
        }
        
        return yaml.dump(service_monitor, default_flow_style=False)
    
    def generate_docker_configuration(self) -> Dict[str, str]:
        """Generate Docker configuration files."""
        configs = {}
        
        # Dockerfile
        configs['Dockerfile'] = self._generate_dockerfile()
        
        # Docker Compose
        configs['docker-compose.yml'] = self._generate_docker_compose()
        
        # .dockerignore
        configs['.dockerignore'] = self._generate_dockerignore()
        
        return configs
    
    def _generate_dockerfile(self) -> str:
        """Generate optimized Dockerfile."""
        dockerfile = """# Multi-stage build for photonic neuromorphic systems
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Create non-root user
RUN groupadd -r photonic && useradd -r -g photonic photonic

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /home/photonic/.local

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY docs/ ./docs/

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs && \\
    chown -R photonic:photonic /app

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV PATH=/home/photonic/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

# Switch to non-root user
USER photonic

# Expose ports
EXPOSE 8080 8090 9000

# Default command
CMD ["python", "-m", "photonic_neuromorphics.cli", "--mode", "server"]
"""
        
        return dockerfile
    
    def _generate_docker_compose(self) -> str:
        """Generate Docker Compose configuration."""
        resources = self.base_config['resources']
        monitoring = self.base_config['monitoring']
        
        compose_config = {
            'version': '3.8',
            'services': {
                'photonic-neuromorphics': {
                    'build': {
                        'context': '.',
                        'dockerfile': 'Dockerfile',
                        'target': 'production'
                    },
                    'ports': [
                        '8080:8080',
                        '8090:8090',
                        '9000:9000'
                    ],
                    'environment': [
                        f'ENVIRONMENT={self.environment.value}',
                        'LOG_LEVEL=INFO',
                        'PYTHONPATH=/app/src'
                    ],
                    'volumes': [
                        './data:/app/data',
                        './logs:/app/logs',
                        './config:/app/config'
                    ],
                    'deploy': {
                        'replicas': self.base_config['replicas'],
                        'resources': {
                            'limits': {
                                'cpus': str(resources.cpu_cores),
                                'memory': f'{resources.memory_gb}G'
                            },
                            'reservations': {
                                'cpus': str(resources.cpu_cores // 2),
                                'memory': f'{resources.memory_gb // 2}G'
                            }
                        },
                        'restart_policy': {
                            'condition': 'on-failure',
                            'delay': '5s',
                            'max_attempts': 3
                        }
                    },
                    'healthcheck': {
                        'test': ['CMD', 'curl', '-f', 'http://localhost:8080/health'],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3,
                        'start_period': '60s'
                    }
                }
            }
        }
        
        # Add monitoring services if enabled
        if monitoring.enable_metrics:
            compose_config['services']['prometheus'] = {
                'image': 'prom/prometheus:latest',
                'ports': ['9090:9090'],
                'volumes': [
                    './monitoring/prometheus:/etc/prometheus',
                    'prometheus_data:/prometheus'
                ],
                'command': [
                    '--config.file=/etc/prometheus/prometheus.yml',
                    '--storage.tsdb.path=/prometheus',
                    '--web.console.libraries=/etc/prometheus/console_libraries',
                    '--web.console.templates=/etc/prometheus/consoles'
                ]
            }
            
            compose_config['services']['grafana'] = {
                'image': 'grafana/grafana:latest',
                'ports': ['3000:3000'],
                'volumes': [
                    './monitoring/grafana:/etc/grafana',
                    'grafana_data:/var/lib/grafana'
                ],
                'environment': [
                    'GF_SECURITY_ADMIN_PASSWORD=admin'
                ]
            }
        
        # Add volumes
        volumes = {}
        if monitoring.enable_metrics:
            volumes.update({
                'prometheus_data': {},
                'grafana_data': {}
            })
        
        if volumes:
            compose_config['volumes'] = volumes
        
        return yaml.dump(compose_config, default_flow_style=False)
    
    def _generate_dockerignore(self) -> str:
        """Generate .dockerignore file."""
        dockerignore = """# Git
.git
.gitignore

# Python
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.venv
venv/

# Development
.pytest_cache
.mypy_cache
.vscode
.idea
*.swp
*.swo
*~

# Documentation
docs/_build
*.pdf

# Test files
tests/
test_*.py
*_test.py

# Build artifacts
build/
dist/
*.egg-info/

# Deployment
deployment/
k8s/
terraform/

# Data and logs
data/
logs/
*.log

# Temporary files
tmp/
temp/
*.tmp
"""
        return dockerignore
    
    def generate_terraform_configuration(self) -> Dict[str, str]:
        """Generate Terraform infrastructure configuration."""
        configs = {}
        
        # Main Terraform configuration
        configs['main.tf'] = self._generate_terraform_main()
        
        # Variables
        configs['variables.tf'] = self._generate_terraform_variables()
        
        # Outputs
        configs['outputs.tf'] = self._generate_terraform_outputs()
        
        return configs
    
    def _generate_terraform_main(self) -> str:
        """Generate main Terraform configuration."""
        terraform_config = f"""# Terraform configuration for Photonic Neuromorphic Systems
terraform {{
  required_version = ">= 1.0"
  required_providers {{
    kubernetes = {{
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }}
    helm = {{
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }}
  }}
}}

# Provider configurations
provider "kubernetes" {{
  config_path = var.kubeconfig_path
}}

provider "helm" {{
  kubernetes {{
    config_path = var.kubeconfig_path
  }}
}}

# Namespace
resource "kubernetes_namespace" "photonic_neuromorphics" {{
  metadata {{
    name = "photonic-neuromorphics-{self.environment.value}"
    labels = {{
      app         = "photonic-neuromorphics"
      environment = "{self.environment.value}"
      managed-by  = "terraform"
    }}
  }}
}}

# Persistent Volume Claim
resource "kubernetes_persistent_volume_claim" "photonic_data" {{
  metadata {{
    name      = "photonic-neuromorphics-pvc"
    namespace = kubernetes_namespace.photonic_neuromorphics.metadata[0].name
  }}
  spec {{
    access_modes = ["ReadWriteOnce"]
    resources {{
      requests = {{
        storage = "{self.base_config['resources'].storage_gb}Gi"
      }}
    }}
    storage_class_name = var.storage_class
  }}
}}

# Monitoring namespace
resource "kubernetes_namespace" "monitoring" {{
  metadata {{
    name = "monitoring"
    labels = {{
      name = "monitoring"
    }}
  }}
}}

# Prometheus Helm release
resource "helm_release" "prometheus" {{
  count = var.enable_monitoring ? 1 : 0
  
  name       = "prometheus"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  namespace  = kubernetes_namespace.monitoring.metadata[0].name
  
  values = [
    yamlencode({{
      prometheus = {{
        prometheusSpec = {{
          retention = "{self.base_config['monitoring'].metrics_retention_days}d"
          storageSpec = {{
            volumeClaimTemplate = {{
              spec = {{
                accessModes = ["ReadWriteOnce"]
                resources = {{
                  requests = {{
                    storage = "50Gi"
                  }}
                }}
              }}
            }}
          }}
        }}
      }}
      grafana = {{
        adminPassword = var.grafana_admin_password
        persistence = {{
          enabled = true
          size    = "10Gi"
        }}
      }}
    }})
  ]
}}

# Ingress Controller
resource "helm_release" "nginx_ingress" {{
  name       = "nginx-ingress"
  repository = "https://kubernetes.github.io/ingress-nginx"
  chart      = "ingress-nginx"
  namespace  = "ingress-nginx"
  
  create_namespace = true
  
  values = [
    yamlencode({{
      controller = {{
        replicaCount = var.ingress_replicas
        service = {{
          type = "LoadBalancer"
        }}
      }}
    }})
  ]
}}

# Cert Manager for TLS
resource "helm_release" "cert_manager" {{
  count = var.enable_tls ? 1 : 0
  
  name       = "cert-manager"
  repository = "https://charts.jetstack.io"
  chart      = "cert-manager"
  namespace  = "cert-manager"
  
  create_namespace = true
  
  set {{
    name  = "installCRDs"
    value = "true"
  }}
}}
"""
        
        return terraform_config
    
    def _generate_terraform_variables(self) -> str:
        """Generate Terraform variables."""
        variables = """# Terraform variables for Photonic Neuromorphic Systems

variable "kubeconfig_path" {
  description = "Path to the kubeconfig file"
  type        = string
  default     = "~/.kube/config"
}

variable "storage_class" {
  description = "Storage class for persistent volumes"
  type        = string
  default     = "fast-ssd"
}

variable "enable_monitoring" {
  description = "Enable monitoring stack (Prometheus/Grafana)"
  type        = bool
  default     = true
}

variable "grafana_admin_password" {
  description = "Admin password for Grafana"
  type        = string
  sensitive   = true
  default     = "admin"
}

variable "enable_tls" {
  description = "Enable TLS/SSL certificates"
  type        = bool
  default     = true
}

variable "ingress_replicas" {
  description = "Number of ingress controller replicas"
  type        = number
  default     = 2
}

variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = "photonic.example.com"
}
"""
        
        return variables
    
    def _generate_terraform_outputs(self) -> str:
        """Generate Terraform outputs."""
        outputs = """# Terraform outputs

output "namespace_name" {
  description = "Name of the created namespace"
  value       = kubernetes_namespace.photonic_neuromorphics.metadata[0].name
}

output "prometheus_endpoint" {
  description = "Prometheus endpoint URL"
  value       = var.enable_monitoring ? "http://prometheus.monitoring.svc.cluster.local:9090" : null
}

output "grafana_endpoint" {
  description = "Grafana endpoint URL"
  value       = var.enable_monitoring ? "http://grafana.monitoring.svc.cluster.local:3000" : null
}

output "application_url" {
  description = "Application URL"
  value       = "https://${var.domain_name}"
}
"""
        
        return outputs
    
    def generate_monitoring_configuration(self) -> Dict[str, str]:
        """Generate monitoring and observability configuration."""
        configs = {}
        
        # Prometheus configuration
        configs['prometheus.yml'] = self._generate_prometheus_config()
        
        # Grafana dashboard
        configs['grafana-dashboard.json'] = self._generate_grafana_dashboard()
        
        # Alert rules
        configs['alert-rules.yml'] = self._generate_alert_rules()
        
        return configs
    
    def _generate_prometheus_config(self) -> str:
        """Generate Prometheus configuration."""
        config = {
            'global': {
                'scrape_interval': '15s',
                'evaluation_interval': '15s'
            },
            'rule_files': [
                '/etc/prometheus/alert-rules.yml'
            ],
            'scrape_configs': [
                {
                    'job_name': 'photonic-neuromorphics',
                    'static_configs': [
                        {
                            'targets': ['photonic-neuromorphics-service:8090']
                        }
                    ],
                    'metrics_path': '/metrics',
                    'scrape_interval': '30s'
                },
                {
                    'job_name': 'kubernetes-apiservers',
                    'kubernetes_sd_configs': [
                        {
                            'role': 'endpoints'
                        }
                    ],
                    'scheme': 'https',
                    'tls_config': {
                        'ca_file': '/var/run/secrets/kubernetes.io/serviceaccount/ca.crt'
                    },
                    'bearer_token_file': '/var/run/secrets/kubernetes.io/serviceaccount/token',
                    'relabel_configs': [
                        {
                            'source_labels': ['__meta_kubernetes_namespace', '__meta_kubernetes_service_name', '__meta_kubernetes_endpoint_port_name'],
                            'action': 'keep',
                            'regex': 'default;kubernetes;https'
                        }
                    ]
                }
            ],
            'alerting': {
                'alertmanagers': [
                    {
                        'static_configs': [
                            {
                                'targets': ['alertmanager:9093']
                            }
                        ]
                    }
                ]
            }
        }
        
        return yaml.dump(config, default_flow_style=False)
    
    def _generate_grafana_dashboard(self) -> str:
        """Generate Grafana dashboard configuration."""
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "Photonic Neuromorphic Systems",
                "tags": ["photonic", "neuromorphic"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "System Overview",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "up{job='photonic-neuromorphics'}",
                                "legendFormat": "System Status"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "Optical Power Levels",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "photonic_optical_power_mw",
                                "legendFormat": "Channel {{channel}}"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    },
                    {
                        "id": 3,
                        "title": "Neural Network Performance",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "photonic_inference_time_seconds",
                                "legendFormat": "Inference Time"
                            },
                            {
                                "expr": "photonic_accuracy_ratio",
                                "legendFormat": "Accuracy"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
                    }
                ],
                "time": {
                    "from": "now-6h",
                    "to": "now"
                },
                "refresh": "30s"
            }
        }
        
        return json.dumps(dashboard, indent=2)
    
    def _generate_alert_rules(self) -> str:
        """Generate Prometheus alert rules."""
        monitoring = self.base_config['monitoring']
        
        rules = {
            'groups': [
                {
                    'name': 'photonic_neuromorphics_alerts',
                    'rules': [
                        {
                            'alert': 'PhotonicSystemDown',
                            'expr': 'up{job="photonic-neuromorphics"} == 0',
                            'for': '5m',
                            'labels': {
                                'severity': 'critical'
                            },
                            'annotations': {
                                'summary': 'Photonic neuromorphic system is down',
                                'description': 'The photonic neuromorphic system has been down for more than 5 minutes.'
                            }
                        },
                        {
                            'alert': 'HighCPUUsage',
                            'expr': f'100 * (1 - avg(rate(container_cpu_usage_seconds_total[5m]))) < {100 - monitoring.cpu_threshold_percent}',
                            'for': '10m',
                            'labels': {
                                'severity': 'warning'
                            },
                            'annotations': {
                                'summary': 'High CPU usage detected',
                                'description': f'CPU usage is above {monitoring.cpu_threshold_percent}% for more than 10 minutes.'
                            }
                        },
                        {
                            'alert': 'HighMemoryUsage',
                            'expr': f'(container_memory_usage_bytes / container_spec_memory_limit_bytes) * 100 > {monitoring.memory_threshold_percent}',
                            'for': '10m',
                            'labels': {
                                'severity': 'warning'
                            },
                            'annotations': {
                                'summary': 'High memory usage detected',
                                'description': f'Memory usage is above {monitoring.memory_threshold_percent}% for more than 10 minutes.'
                            }
                        },
                        {
                            'alert': 'OpticalPowerAnomaly',
                            'expr': 'photonic_optical_power_mw < 0.1 or photonic_optical_power_mw > 10',
                            'for': '2m',
                            'labels': {
                                'severity': 'critical'
                            },
                            'annotations': {
                                'summary': 'Optical power anomaly detected',
                                'description': 'Optical power levels are outside normal operating range.'
                            }
                        }
                    ]
                }
            ]
        }
        
        return yaml.dump(rules, default_flow_style=False)


def generate_complete_deployment_suite(environment: DeploymentEnvironment) -> Dict[str, Dict[str, str]]:
    """Generate complete deployment suite for specified environment."""
    generator = ProductionDeploymentGenerator(environment)
    
    deployment_suite = {
        'kubernetes': generator.generate_kubernetes_manifests(),
        'docker': generator.generate_docker_configuration(),
        'terraform': generator.generate_terraform_configuration(),
        'monitoring': generator.generate_monitoring_configuration()
    }
    
    return deployment_suite


def save_deployment_configurations(deployment_suite: Dict[str, Dict[str, str]], output_dir: str = "deployment_output"):
    """Save deployment configurations to files."""
    import os
    
    for category, configs in deployment_suite.items():
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        for filename, content in configs.items():
            filepath = os.path.join(category_dir, filename)
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Generated: {filepath}")


if __name__ == "__main__":
    print("🚀 Generating Production Deployment Suite")
    print("=" * 50)
    
    # Generate for all environments
    environments = [
        DeploymentEnvironment.DEVELOPMENT,
        DeploymentEnvironment.STAGING,
        DeploymentEnvironment.PRODUCTION
    ]
    
    for env in environments:
        print(f"\nGenerating deployment for {env.value} environment...")
        
        deployment_suite = generate_complete_deployment_suite(env)
        output_dir = f"deployment_{env.value}"
        
        save_deployment_configurations(deployment_suite, output_dir)
        
        print(f"✅ {env.value.title()} deployment suite generated in {output_dir}/")
    
    print("\n🎉 All deployment configurations generated successfully!")
    print("\nNext steps:")
    print("1. Review and customize configurations for your environment")
    print("2. Set up your Kubernetes cluster")
    print("3. Deploy using: kubectl apply -f deployment_production/kubernetes/")
    print("4. Set up monitoring: terraform apply -f deployment_production/terraform/")
    print("5. Verify deployment health and monitoring")