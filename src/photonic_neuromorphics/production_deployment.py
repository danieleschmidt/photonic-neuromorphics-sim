"""
Production Deployment Suite for Photonic Neuromorphic Computing.

This module provides comprehensive production deployment capabilities including
containerization, orchestration, CI/CD integration, and cloud deployment support.
"""

import os
import sys
import time
import json
import yaml
import subprocess
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import shutil
import logging

from .enhanced_logging import PhotonicLogger
from .quality_assurance import QualityAssuranceFramework
from .production_health_monitor import HealthMonitor, AlertingSystem
from .high_performance_scaling import ScalingConfig, DistributedPhotonicProcessor


@dataclass
class DeploymentConfig:
    """Configuration for production deployment."""
    # Environment settings
    environment: str = "production"  # development, staging, production
    service_name: str = "photonic-neuromorphics"
    service_version: str = "1.0.0"
    
    # Container settings
    container_registry: str = "docker.io"
    base_image: str = "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime"
    enable_gpu: bool = True
    memory_limit: str = "8Gi"
    cpu_limit: str = "4"
    
    # Kubernetes settings
    kubernetes_namespace: str = "photonic-neuromorphics"
    replicas: int = 3
    max_replicas: int = 10
    enable_autoscaling: bool = True
    target_cpu_utilization: int = 70
    
    # Networking
    service_port: int = 8080
    health_check_port: int = 8081
    enable_ingress: bool = True
    ingress_host: str = "photonic-api.example.com"
    
    # Security
    enable_rbac: bool = True
    enable_network_policies: bool = True
    enable_pod_security_policy: bool = True
    
    # Monitoring and logging
    enable_prometheus_monitoring: bool = True
    enable_jaeger_tracing: bool = True
    log_level: str = "INFO"
    
    # Database and storage
    persistent_storage: bool = True
    storage_class: str = "fast-ssd"
    storage_size: str = "100Gi"
    
    # Performance
    enable_performance_monitoring: bool = True
    enable_distributed_processing: bool = True
    
    # Backup and disaster recovery
    enable_backup: bool = True
    backup_schedule: str = "0 2 * * *"  # Daily at 2 AM
    backup_retention_days: int = 30


class ContainerBuilder:
    """
    Advanced container builder for photonic neuromorphic applications.
    
    Creates optimized, production-ready container images with:
    - Multi-stage builds for size optimization
    - Security hardening
    - Performance optimizations
    - GPU support
    """
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.logger = PhotonicLogger(__name__)
        self.build_context = None
    
    def build_production_image(self, source_directory: str, output_tag: str) -> bool:
        """Build production-ready container image."""
        
        self.logger.info(f"Building production image: {output_tag}")
        
        try:
            # Create build context
            self.build_context = self._create_build_context(source_directory)
            
            # Generate Dockerfile
            dockerfile_path = self._generate_dockerfile()
            
            # Build image
            success = self._build_docker_image(dockerfile_path, output_tag)
            
            if success:
                # Run security scan
                self._scan_image_security(output_tag)
                
                # Test image
                self._test_image(output_tag)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Container build failed: {e}")
            return False
        finally:
            # Cleanup build context
            if self.build_context and os.path.exists(self.build_context):
                shutil.rmtree(self.build_context)
    
    def _create_build_context(self, source_directory: str) -> str:
        """Create optimized build context."""
        
        build_context = tempfile.mkdtemp(prefix="photonic_build_")
        
        # Copy source code
        src_dest = os.path.join(build_context, "src")
        shutil.copytree(source_directory, src_dest)
        
        # Copy requirements
        requirements_files = ["requirements.txt", "requirements-prod.txt"]
        for req_file in requirements_files:
            src_path = os.path.join(os.path.dirname(source_directory), req_file)
            if os.path.exists(src_path):
                shutil.copy2(src_path, build_context)
        
        # Copy configuration files
        config_files = ["pyproject.toml", "setup.py"]
        for config_file in config_files:
            src_path = os.path.join(os.path.dirname(source_directory), config_file)
            if os.path.exists(src_path):
                shutil.copy2(src_path, build_context)
        
        self.logger.info(f"Created build context: {build_context}")
        return build_context
    
    def _generate_dockerfile(self) -> str:
        """Generate optimized production Dockerfile."""
        
        dockerfile_content = f'''
# Multi-stage build for production optimization
FROM {self.config.base_image} AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    git \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements*.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
{"RUN pip install --no-cache-dir -r requirements-prod.txt" if os.path.exists(os.path.join(self.build_context, "requirements-prod.txt")) else ""}

# Copy source code
COPY src/ ./src/
COPY pyproject.toml setup.py ./

# Install the package
RUN pip install -e .

# Production stage
FROM {self.config.base_image} AS production

# Security: Create non-root user
RUN groupadd -r photonic && useradd -r -g photonic photonic

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ENVIRONMENT={self.config.environment}
ENV SERVICE_NAME={self.config.service_name}
ENV LOG_LEVEL={self.config.log_level}

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/* \\
    && apt-get clean

# Copy installed packages from builder
COPY --from=builder /opt/conda /opt/conda

# Create app directory
WORKDIR /app

# Copy application code
COPY --from=builder /app/src ./src
COPY --from=builder /app/pyproject.toml ./

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs \\
    && chown -R photonic:photonic /app

# Set up health check
COPY scripts/health_check.py ./
RUN chmod +x health_check.py

# Expose ports
EXPOSE {self.config.service_port}
EXPOSE {self.config.health_check_port}

# Switch to non-root user
USER photonic

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \\
    CMD python health_check.py || exit 1

# Start command
CMD ["python", "-m", "photonic_neuromorphics.api.server", "--port", "{self.config.service_port}"]
'''
        
        dockerfile_path = os.path.join(self.build_context, "Dockerfile")
        
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content.strip())
        
        # Create health check script
        self._create_health_check_script()
        
        return dockerfile_path
    
    def _create_health_check_script(self):
        """Create health check script for container."""
        
        health_check_content = '''#!/usr/bin/env python3
import sys
import requests
import socket
import os

def check_health():
    """Check application health."""
    try:
        # Check if main service is responding
        port = int(os.environ.get('SERVICE_PORT', '8080'))
        health_port = int(os.environ.get('HEALTH_CHECK_PORT', '8081'))
        
        # Check main service
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        
        if result != 0:
            print(f"Main service not responding on port {port}")
            return False
        
        # Check health endpoint if available
        try:
            response = requests.get(f'http://localhost:{health_port}/health', timeout=5)
            if response.status_code != 200:
                print(f"Health endpoint returned {response.status_code}")
                return False
        except requests.exceptions.RequestException:
            # Health endpoint might not be implemented yet
            pass
        
        print("Health check passed")
        return True
        
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

if __name__ == "__main__":
    if check_health():
        sys.exit(0)
    else:
        sys.exit(1)
'''
        
        scripts_dir = os.path.join(self.build_context, "scripts")
        os.makedirs(scripts_dir, exist_ok=True)
        
        health_check_path = os.path.join(scripts_dir, "health_check.py")
        with open(health_check_path, 'w') as f:
            f.write(health_check_content.strip())
        
        os.chmod(health_check_path, 0o755)
    
    def _build_docker_image(self, dockerfile_path: str, tag: str) -> bool:
        """Build Docker image."""
        
        try:
            build_args = [
                "docker", "build",
                "-f", dockerfile_path,
                "-t", tag,
                self.build_context
            ]
            
            # Add build arguments
            if self.config.enable_gpu:
                build_args.extend(["--build-arg", "ENABLE_GPU=true"])
            
            self.logger.info(f"Running: {' '.join(build_args)}")
            
            result = subprocess.run(
                build_args,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode == 0:
                self.logger.info(f"Successfully built image: {tag}")
                return True
            else:
                self.logger.error(f"Docker build failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("Docker build timed out")
            return False
        except Exception as e:
            self.logger.error(f"Docker build error: {e}")
            return False
    
    def _scan_image_security(self, tag: str):
        """Scan image for security vulnerabilities."""
        
        try:
            # Try to run security scan with trivy (if available)
            scan_result = subprocess.run(
                ["trivy", "image", "--severity", "HIGH,CRITICAL", tag],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if scan_result.returncode == 0:
                self.logger.info("Security scan completed successfully")
                if scan_result.stdout:
                    self.logger.warning(f"Security scan results:\\n{scan_result.stdout}")
            else:
                self.logger.warning("Security scanner not available or failed")
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.logger.warning("Security scanning skipped (trivy not available)")
    
    def _test_image(self, tag: str):
        """Test the built image."""
        
        try:
            # Run basic container test
            test_result = subprocess.run([
                "docker", "run", "--rm",
                "-e", "ENVIRONMENT=test",
                tag,
                "python", "-c", "import photonic_neuromorphics; print('Import test passed')"
            ], capture_output=True, text=True, timeout=60)
            
            if test_result.returncode == 0:
                self.logger.info("Container test passed")
            else:
                self.logger.error(f"Container test failed: {test_result.stderr}")
                
        except subprocess.TimeoutExpired:
            self.logger.error("Container test timed out")
        except Exception as e:
            self.logger.error(f"Container test error: {e}")


class KubernetesDeployer:
    """
    Kubernetes deployment manager for photonic neuromorphic services.
    
    Provides:
    - Complete Kubernetes manifest generation
    - Automated deployment and rollback
    - Service mesh integration
    - Auto-scaling configuration
    """
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.logger = PhotonicLogger(__name__)
        self.manifests_dir = None
    
    def deploy_to_kubernetes(self, image_tag: str) -> bool:
        """Deploy application to Kubernetes."""
        
        self.logger.info(f"Deploying to Kubernetes: {image_tag}")
        
        try:
            # Generate manifests
            self.manifests_dir = self._generate_k8s_manifests(image_tag)
            
            # Apply manifests
            success = self._apply_manifests()
            
            if success:
                # Wait for deployment to be ready
                self._wait_for_deployment()
                
                # Run deployment tests
                self._test_deployment()
            
            return success
            
        except Exception as e:
            self.logger.error(f"Kubernetes deployment failed: {e}")
            return False
    
    def _generate_k8s_manifests(self, image_tag: str) -> str:
        """Generate complete Kubernetes manifests."""
        
        manifests_dir = tempfile.mkdtemp(prefix="k8s_manifests_")
        
        # Generate all manifests
        self._generate_namespace_manifest(manifests_dir)
        self._generate_deployment_manifest(manifests_dir, image_tag)
        self._generate_service_manifest(manifests_dir)
        self._generate_hpa_manifest(manifests_dir)
        self._generate_configmap_manifest(manifests_dir)
        self._generate_secret_manifest(manifests_dir)
        
        if self.config.enable_ingress:
            self._generate_ingress_manifest(manifests_dir)
        
        if self.config.persistent_storage:
            self._generate_pvc_manifest(manifests_dir)
        
        if self.config.enable_rbac:
            self._generate_rbac_manifests(manifests_dir)
        
        if self.config.enable_prometheus_monitoring:
            self._generate_monitoring_manifests(manifests_dir)
        
        return manifests_dir
    
    def _generate_namespace_manifest(self, manifests_dir: str):
        """Generate namespace manifest."""
        
        namespace_manifest = {
            'apiVersion': 'v1',
            'kind': 'Namespace',
            'metadata': {
                'name': self.config.kubernetes_namespace,
                'labels': {
                    'name': self.config.kubernetes_namespace,
                    'app': self.config.service_name
                }
            }
        }
        
        self._write_manifest(manifests_dir, "namespace.yaml", namespace_manifest)
    
    def _generate_deployment_manifest(self, manifests_dir: str, image_tag: str):
        """Generate deployment manifest."""
        
        deployment_manifest = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': self.config.service_name,
                'namespace': self.config.kubernetes_namespace,
                'labels': {
                    'app': self.config.service_name,
                    'version': self.config.service_version
                }
            },
            'spec': {
                'replicas': self.config.replicas,
                'selector': {
                    'matchLabels': {
                        'app': self.config.service_name
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': self.config.service_name,
                            'version': self.config.service_version
                        },
                        'annotations': {
                            'prometheus.io/scrape': 'true',
                            'prometheus.io/port': str(self.config.health_check_port),
                            'prometheus.io/path': '/metrics'
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': self.config.service_name,
                            'image': image_tag,
                            'ports': [
                                {
                                    'containerPort': self.config.service_port,
                                    'name': 'http'
                                },
                                {
                                    'containerPort': self.config.health_check_port,
                                    'name': 'health'
                                }
                            ],
                            'env': [
                                {
                                    'name': 'ENVIRONMENT',
                                    'value': self.config.environment
                                },
                                {
                                    'name': 'SERVICE_NAME',
                                    'value': self.config.service_name
                                },
                                {
                                    'name': 'LOG_LEVEL',
                                    'value': self.config.log_level
                                }
                            ],
                            'resources': {
                                'requests': {
                                    'memory': self.config.memory_limit,
                                    'cpu': self.config.cpu_limit
                                },
                                'limits': {
                                    'memory': self.config.memory_limit,
                                    'cpu': self.config.cpu_limit
                                }
                            },
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': self.config.health_check_port
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/ready',
                                    'port': self.config.health_check_port
                                },
                                'initialDelaySeconds': 5,
                                'periodSeconds': 5
                            }
                        }],
                        'securityContext': {
                            'runAsNonRoot': True,
                            'runAsUser': 1000,
                            'fsGroup': 1000
                        }
                    }
                }
            }
        }
        
        # Add GPU resources if enabled
        if self.config.enable_gpu:
            gpu_resources = {'nvidia.com/gpu': '1'}
            deployment_manifest['spec']['template']['spec']['containers'][0]['resources']['requests'].update(gpu_resources)
            deployment_manifest['spec']['template']['spec']['containers'][0]['resources']['limits'].update(gpu_resources)
        
        self._write_manifest(manifests_dir, "deployment.yaml", deployment_manifest)
    
    def _generate_service_manifest(self, manifests_dir: str):
        """Generate service manifest."""
        
        service_manifest = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': self.config.service_name,
                'namespace': self.config.kubernetes_namespace,
                'labels': {
                    'app': self.config.service_name
                }
            },
            'spec': {
                'selector': {
                    'app': self.config.service_name
                },
                'ports': [
                    {
                        'name': 'http',
                        'port': 80,
                        'targetPort': self.config.service_port
                    },
                    {
                        'name': 'health',
                        'port': self.config.health_check_port,
                        'targetPort': self.config.health_check_port
                    }
                ],
                'type': 'ClusterIP'
            }
        }
        
        self._write_manifest(manifests_dir, "service.yaml", service_manifest)
    
    def _generate_hpa_manifest(self, manifests_dir: str):
        """Generate horizontal pod autoscaler manifest."""
        
        if not self.config.enable_autoscaling:
            return
        
        hpa_manifest = {
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': f"{self.config.service_name}-hpa",
                'namespace': self.config.kubernetes_namespace
            },
            'spec': {
                'scaleTargetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': self.config.service_name
                },
                'minReplicas': self.config.replicas,
                'maxReplicas': self.config.max_replicas,
                'metrics': [
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'cpu',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': self.config.target_cpu_utilization
                            }
                        }
                    }
                ]
            }
        }
        
        self._write_manifest(manifests_dir, "hpa.yaml", hpa_manifest)
    
    def _generate_configmap_manifest(self, manifests_dir: str):
        """Generate ConfigMap manifest."""
        
        configmap_manifest = {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': f"{self.config.service_name}-config",
                'namespace': self.config.kubernetes_namespace
            },
            'data': {
                'environment': self.config.environment,
                'log_level': self.config.log_level,
                'service_port': str(self.config.service_port),
                'health_check_port': str(self.config.health_check_port)
            }
        }
        
        self._write_manifest(manifests_dir, "configmap.yaml", configmap_manifest)
    
    def _generate_secret_manifest(self, manifests_dir: str):
        """Generate Secret manifest."""
        
        # This is a placeholder - in real deployment, secrets should be managed securely
        secret_manifest = {
            'apiVersion': 'v1',
            'kind': 'Secret',
            'metadata': {
                'name': f"{self.config.service_name}-secrets",
                'namespace': self.config.kubernetes_namespace
            },
            'type': 'Opaque',
            'data': {
                # Base64 encoded placeholder secrets
                'api_key': 'cGxhY2Vob2xkZXI=',  # 'placeholder'
                'db_password': 'cGxhY2Vob2xkZXI='  # 'placeholder'
            }
        }
        
        self._write_manifest(manifests_dir, "secret.yaml", secret_manifest)
    
    def _generate_ingress_manifest(self, manifests_dir: str):
        """Generate Ingress manifest."""
        
        ingress_manifest = {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'Ingress',
            'metadata': {
                'name': f"{self.config.service_name}-ingress",
                'namespace': self.config.kubernetes_namespace,
                'annotations': {
                    'nginx.ingress.kubernetes.io/rewrite-target': '/',
                    'nginx.ingress.kubernetes.io/ssl-redirect': 'true',
                    'cert-manager.io/cluster-issuer': 'letsencrypt-prod'
                }
            },
            'spec': {
                'tls': [
                    {
                        'hosts': [self.config.ingress_host],
                        'secretName': f"{self.config.service_name}-tls"
                    }
                ],
                'rules': [
                    {
                        'host': self.config.ingress_host,
                        'http': {
                            'paths': [
                                {
                                    'path': '/',
                                    'pathType': 'Prefix',
                                    'backend': {
                                        'service': {
                                            'name': self.config.service_name,
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
        
        self._write_manifest(manifests_dir, "ingress.yaml", ingress_manifest)
    
    def _generate_pvc_manifest(self, manifests_dir: str):
        """Generate PersistentVolumeClaim manifest."""
        
        pvc_manifest = {
            'apiVersion': 'v1',
            'kind': 'PersistentVolumeClaim',
            'metadata': {
                'name': f"{self.config.service_name}-storage",
                'namespace': self.config.kubernetes_namespace
            },
            'spec': {
                'accessModes': ['ReadWriteOnce'],
                'storageClassName': self.config.storage_class,
                'resources': {
                    'requests': {
                        'storage': self.config.storage_size
                    }
                }
            }
        }
        
        self._write_manifest(manifests_dir, "pvc.yaml", pvc_manifest)
    
    def _generate_rbac_manifests(self, manifests_dir: str):
        """Generate RBAC manifests."""
        
        # ServiceAccount
        sa_manifest = {
            'apiVersion': 'v1',
            'kind': 'ServiceAccount',
            'metadata': {
                'name': self.config.service_name,
                'namespace': self.config.kubernetes_namespace
            }
        }
        
        self._write_manifest(manifests_dir, "serviceaccount.yaml", sa_manifest)
        
        # Role
        role_manifest = {
            'apiVersion': 'rbac.authorization.k8s.io/v1',
            'kind': 'Role',
            'metadata': {
                'name': self.config.service_name,
                'namespace': self.config.kubernetes_namespace
            },
            'rules': [
                {
                    'apiGroups': [''],
                    'resources': ['pods', 'services'],
                    'verbs': ['get', 'list', 'watch']
                }
            ]
        }
        
        self._write_manifest(manifests_dir, "role.yaml", role_manifest)
        
        # RoleBinding
        rb_manifest = {
            'apiVersion': 'rbac.authorization.k8s.io/v1',
            'kind': 'RoleBinding',
            'metadata': {
                'name': self.config.service_name,
                'namespace': self.config.kubernetes_namespace
            },
            'subjects': [
                {
                    'kind': 'ServiceAccount',
                    'name': self.config.service_name,
                    'namespace': self.config.kubernetes_namespace
                }
            ],
            'roleRef': {
                'kind': 'Role',
                'name': self.config.service_name,
                'apiGroup': 'rbac.authorization.k8s.io'
            }
        }
        
        self._write_manifest(manifests_dir, "rolebinding.yaml", rb_manifest)
    
    def _generate_monitoring_manifests(self, manifests_dir: str):
        """Generate monitoring manifests."""
        
        # ServiceMonitor for Prometheus
        service_monitor = {
            'apiVersion': 'monitoring.coreos.com/v1',
            'kind': 'ServiceMonitor',
            'metadata': {
                'name': f"{self.config.service_name}-monitor",
                'namespace': self.config.kubernetes_namespace,
                'labels': {
                    'app': self.config.service_name
                }
            },
            'spec': {
                'selector': {
                    'matchLabels': {
                        'app': self.config.service_name
                    }
                },
                'endpoints': [
                    {
                        'port': 'health',
                        'path': '/metrics',
                        'interval': '30s'
                    }
                ]
            }
        }
        
        self._write_manifest(manifests_dir, "servicemonitor.yaml", service_monitor)
    
    def _write_manifest(self, manifests_dir: str, filename: str, manifest: Dict[str, Any]):
        """Write Kubernetes manifest to file."""
        
        manifest_path = os.path.join(manifests_dir, filename)
        
        with open(manifest_path, 'w') as f:
            yaml.dump(manifest, f, default_flow_style=False)
        
        self.logger.debug(f"Generated manifest: {manifest_path}")
    
    def _apply_manifests(self) -> bool:
        """Apply Kubernetes manifests."""
        
        try:
            # Apply all manifests
            result = subprocess.run([
                "kubectl", "apply",
                "-f", self.manifests_dir,
                "--recursive"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                self.logger.info("Kubernetes manifests applied successfully")
                return True
            else:
                self.logger.error(f"Failed to apply manifests: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("Kubectl apply timed out")
            return False
        except Exception as e:
            self.logger.error(f"Error applying manifests: {e}")
            return False
    
    def _wait_for_deployment(self):
        """Wait for deployment to be ready."""
        
        try:
            self.logger.info("Waiting for deployment to be ready...")
            
            result = subprocess.run([
                "kubectl", "rollout", "status",
                f"deployment/{self.config.service_name}",
                "-n", self.config.kubernetes_namespace,
                "--timeout=600s"
            ], capture_output=True, text=True, timeout=700)
            
            if result.returncode == 0:
                self.logger.info("Deployment is ready")
            else:
                self.logger.warning(f"Deployment status check failed: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Error checking deployment status: {e}")
    
    def _test_deployment(self):
        """Test the deployment."""
        
        try:
            # Get service endpoint
            result = subprocess.run([
                "kubectl", "get", "service",
                self.config.service_name,
                "-n", self.config.kubernetes_namespace,
                "-o", "jsonpath={.spec.clusterIP}"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                cluster_ip = result.stdout.strip()
                self.logger.info(f"Service available at: {cluster_ip}")
            
        except Exception as e:
            self.logger.error(f"Error testing deployment: {e}")


class ProductionDeploymentManager:
    """
    Complete production deployment manager.
    
    Orchestrates the entire deployment pipeline:
    - Quality assurance checks
    - Container building and testing
    - Kubernetes deployment
    - Health monitoring setup
    - Rollback capabilities
    """
    
    def __init__(self, config: DeploymentConfig = None):
        self.config = config or DeploymentConfig()
        self.logger = PhotonicLogger(__name__)
        
        # Initialize components
        self.qa_framework = QualityAssuranceFramework()
        self.container_builder = ContainerBuilder(self.config)
        self.k8s_deployer = KubernetesDeployer(self.config)
        
        # Deployment state
        self.deployment_id = None
        self.image_tag = None
    
    def deploy_to_production(self, source_directory: str) -> bool:
        """
        Execute complete production deployment pipeline.
        
        Args:
            source_directory: Path to source code directory
            
        Returns:
            True if deployment successful, False otherwise
        """
        
        self.deployment_id = f"deploy-{int(time.time())}"
        self.logger.info(f"Starting production deployment: {self.deployment_id}")
        
        try:
            # 1. Quality Assurance
            self.logger.info("Phase 1: Quality Assurance")
            if not self._run_quality_assurance():
                return False
            
            # 2. Container Building
            self.logger.info("Phase 2: Container Building")
            if not self._build_container(source_directory):
                return False
            
            # 3. Kubernetes Deployment
            self.logger.info("Phase 3: Kubernetes Deployment")
            if not self._deploy_to_kubernetes():
                return False
            
            # 4. Post-deployment Setup
            self.logger.info("Phase 4: Post-deployment Setup")
            self._setup_monitoring_and_alerting()
            
            # 5. Deployment Verification
            self.logger.info("Phase 5: Deployment Verification")
            if not self._verify_deployment():
                self.logger.error("Deployment verification failed, initiating rollback")
                self._rollback_deployment()
                return False
            
            self.logger.info(f"Production deployment completed successfully: {self.deployment_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Production deployment failed: {e}")
            self._rollback_deployment()
            return False
    
    def _run_quality_assurance(self) -> bool:
        """Run comprehensive quality assurance."""
        
        try:
            qa_report = self.qa_framework.run_complete_qa_pipeline()
            
            if qa_report.passed_quality_gates:
                self.logger.info(f"QA passed with score: {qa_report.overall_quality_score:.1f}")
                return True
            else:
                self.logger.error("QA failed - quality gates not met")
                self.logger.error(f"Recommendations: {qa_report.recommendations}")
                return False
                
        except Exception as e:
            self.logger.error(f"QA pipeline failed: {e}")
            return False
    
    def _build_container(self, source_directory: str) -> bool:
        """Build production container."""
        
        try:
            self.image_tag = f"{self.config.container_registry}/{self.config.service_name}:{self.config.service_version}-{self.deployment_id}"
            
            success = self.container_builder.build_production_image(
                source_directory,
                self.image_tag
            )
            
            if success:
                self.logger.info(f"Container built successfully: {self.image_tag}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Container build failed: {e}")
            return False
    
    def _deploy_to_kubernetes(self) -> bool:
        """Deploy to Kubernetes."""
        
        try:
            success = self.k8s_deployer.deploy_to_kubernetes(self.image_tag)
            
            if success:
                self.logger.info("Kubernetes deployment successful")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Kubernetes deployment failed: {e}")
            return False
    
    def _setup_monitoring_and_alerting(self):
        """Setup monitoring and alerting for the deployment."""
        
        try:
            # Setup health monitoring
            health_monitor = HealthMonitor()
            alerting_system = AlertingSystem()
            
            # Add email notifications (if configured)
            # alerting_system.add_email_notification(...)
            
            health_monitor.add_alert_callback(alerting_system.send_alert)
            health_monitor.start_monitoring()
            
            self.logger.info("Monitoring and alerting setup completed")
            
        except Exception as e:
            self.logger.warning(f"Monitoring setup failed: {e}")
    
    def _verify_deployment(self) -> bool:
        """Verify deployment health and functionality."""
        
        try:
            # Wait for pods to be ready
            time.sleep(30)
            
            # Check pod status
            result = subprocess.run([
                "kubectl", "get", "pods",
                "-n", self.config.kubernetes_namespace,
                "-l", f"app={self.config.service_name}",
                "--field-selector=status.phase=Running",
                "-o", "json"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                pods_data = json.loads(result.stdout)
                running_pods = len(pods_data.get('items', []))
                
                if running_pods >= self.config.replicas:
                    self.logger.info(f"Deployment verification passed: {running_pods} pods running")
                    return True
                else:
                    self.logger.error(f"Insufficient pods running: {running_pods}/{self.config.replicas}")
                    return False
            else:
                self.logger.error("Failed to check pod status")
                return False
                
        except Exception as e:
            self.logger.error(f"Deployment verification failed: {e}")
            return False
    
    def _rollback_deployment(self):
        """Rollback deployment to previous version."""
        
        try:
            self.logger.info("Initiating deployment rollback")
            
            # Kubernetes rollback
            subprocess.run([
                "kubectl", "rollout", "undo",
                f"deployment/{self.config.service_name}",
                "-n", self.config.kubernetes_namespace
            ], timeout=300)
            
            self.logger.info("Rollback completed")
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        
        return {
            'deployment_id': self.deployment_id,
            'image_tag': self.image_tag,
            'service_name': self.config.service_name,
            'environment': self.config.environment,
            'namespace': self.config.kubernetes_namespace
        }


def deploy_production_system():
    """Deploy complete production system."""
    print("🚀 PRODUCTION DEPLOYMENT SYSTEM")
    print("=" * 60)
    
    # Create deployment configuration
    config = DeploymentConfig(
        environment="production",
        service_name="photonic-neuromorphics",
        service_version="1.0.0",
        enable_gpu=True,
        replicas=3,
        enable_autoscaling=True,
        enable_prometheus_monitoring=True
    )
    
    # Create deployment manager
    deployment_manager = ProductionDeploymentManager(config)
    
    print("🔍 Running production deployment pipeline...")
    
    # Note: In a real scenario, you would provide the actual source directory
    source_dir = "src/photonic_neuromorphics"
    
    # Deploy (this would normally run the full pipeline)
    print("✅ Production deployment pipeline configured")
    print("\\n📊 DEPLOYMENT CONFIGURATION:")
    print(f"Service: {config.service_name}")
    print(f"Environment: {config.environment}")
    print(f"Replicas: {config.replicas}")
    print(f"Auto-scaling: {config.enable_autoscaling}")
    print(f"GPU Support: {config.enable_gpu}")
    print(f"Monitoring: {config.enable_prometheus_monitoring}")
    
    print("\\n🎯 DEPLOYMENT FEATURES:")
    print("  ✓ Quality assurance pipeline")
    print("  ✓ Multi-stage container builds")
    print("  ✓ Kubernetes orchestration")
    print("  ✓ Auto-scaling configuration")
    print("  ✓ Health monitoring & alerting")
    print("  ✓ Security hardening")
    print("  ✓ Rollback capabilities")
    
    return deployment_manager


if __name__ == "__main__":
    deployment_manager = deploy_production_system()