"""
Security module for photonic neuromorphics simulation framework.

This module provides security controls, input validation, and secure configuration
management for the photonic simulation environment.
"""

import hashlib
import hmac
import secrets
import time
import json
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator, SecretStr
from pathlib import Path
import yaml
from functools import wraps

from .exceptions import SecurityError, ValidationError


@dataclass
class SecurityConfig:
    """Security configuration parameters."""
    enable_input_validation: bool = True
    enable_output_sanitization: bool = True
    max_simulation_time: float = 1e-9  # 1 ns max simulation
    max_memory_usage: int = 1024 * 1024 * 1024  # 1 GB
    max_file_size: int = 100 * 1024 * 1024  # 100 MB
    allowed_file_types: List[str] = field(default_factory=lambda: ['.gds', '.sp', '.v', '.sv', '.json', '.yaml'])
    rate_limit_requests: int = 1000  # requests per hour
    enable_audit_logging: bool = True
    require_authentication: bool = False
    session_timeout: int = 3600  # 1 hour


class SecureSimulationSession(BaseModel):
    """Secure session management for simulations."""
    
    session_id: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    user_id: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    last_activity: float = Field(default_factory=time.time)
    permissions: List[str] = Field(default_factory=list)
    is_active: bool = True
    max_simulation_time: float = 1e-9
    memory_limit: int = 1024 * 1024 * 1024
    
    def __init__(self, **data):
        super().__init__(**data)
        self._logger = logging.getLogger(__name__)
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = time.time()
    
    def is_expired(self, timeout: int = 3600) -> bool:
        """Check if session has expired."""
        return (time.time() - self.last_activity) > timeout
    
    def has_permission(self, permission: str) -> bool:
        """Check if session has specific permission."""
        return permission in self.permissions or 'admin' in self.permissions
    
    def validate_simulation_request(self, simulation_params: Dict[str, Any]) -> bool:
        """Validate simulation request against session limits."""
        # Check simulation time
        sim_time = simulation_params.get('simulation_time', 0)
        if sim_time > self.max_simulation_time:
            raise SecurityError(
                f"Simulation time {sim_time:.2e}s exceeds limit {self.max_simulation_time:.2e}s"
            )
        
        # Check memory requirements
        estimated_memory = simulation_params.get('estimated_memory', 0)
        if estimated_memory > self.memory_limit:
            raise SecurityError(
                f"Estimated memory {estimated_memory} exceeds limit {self.memory_limit}"
            )
        
        return True


class InputValidator:
    """Comprehensive input validation for photonic simulations."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self._logger = logging.getLogger(__name__)
    
    def validate_optical_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate optical simulation parameters."""
        if not self.config.enable_input_validation:
            return params
        
        validated = {}
        
        # Wavelength validation
        wavelength = params.get('wavelength', 1550e-9)
        if not (200e-9 <= wavelength <= 2000e-9):
            raise ValidationError(f"Wavelength {wavelength*1e9:.1f} nm outside valid range (200-2000 nm)")
        validated['wavelength'] = float(wavelength)
        
        # Power validation
        power = params.get('power', 1e-3)
        if not (1e-12 <= power <= 1.0):
            raise ValidationError(f"Power {power:.2e} W outside valid range (1 pW - 1 W)")
        validated['power'] = float(power)
        
        # Loss validation
        loss = params.get('loss', 0.1)
        if not (0.0 <= loss <= 1000.0):
            raise ValidationError(f"Loss {loss} dB/cm outside valid range (0-1000 dB/cm)")
        validated['loss'] = float(loss)
        
        # Validate complex parameters
        for key, value in params.items():
            if key not in validated:
                validated[key] = self._validate_numeric_parameter(key, value)
        
        self._logger.info(f"Validated optical parameters: {len(validated)} fields")
        return validated
    
    def validate_device_geometry(self, geometry: Dict[str, Any]) -> Dict[str, Any]:
        """Validate device geometry parameters."""
        if not self.config.enable_input_validation:
            return geometry
        
        validated = {}
        
        # Dimension validation
        for dim in ['width', 'height', 'length', 'thickness']:
            if dim in geometry:
                value = geometry[dim]
                if not (1e-9 <= value <= 1e-2):  # 1 nm to 1 cm
                    raise ValidationError(f"{dim} {value*1e6:.1f} μm outside valid range (0.001-10000 μm)")
                validated[dim] = float(value)
        
        # Position validation
        for pos in ['x', 'y', 'z']:
            if pos in geometry:
                value = geometry[pos]
                if not (-1e-2 <= value <= 1e-2):  # ±1 cm
                    raise ValidationError(f"Position {pos} {value*1e3:.1f} mm outside valid range (±10 mm)")
                validated[pos] = float(value)
        
        # Domain size validation
        if 'domain_size' in geometry:
            domain = geometry['domain_size']
            if isinstance(domain, (list, tuple)) and len(domain) == 3:
                for i, size in enumerate(domain):
                    if not (1e-6 <= size <= 1e-2):  # 1 μm to 1 cm
                        raise ValidationError(f"Domain size[{i}] {size*1e6:.1f} μm outside valid range")
                validated['domain_size'] = tuple(float(s) for s in domain)
        
        # Material property validation
        if 'refractive_index' in geometry:
            n = geometry['refractive_index']
            if not (1.0 <= n <= 10.0):  # Reasonable material range
                raise ValidationError(f"Refractive index {n} outside valid range (1.0-10.0)")
            validated['refractive_index'] = float(n)
        
        self._logger.info(f"Validated device geometry: {len(validated)} fields")
        return validated
    
    def validate_simulation_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate simulation control parameters."""
        if not self.config.enable_input_validation:
            return params
        
        validated = {}
        
        # Time validation
        sim_time = params.get('simulation_time', 100e-15)
        if not (1e-18 <= sim_time <= self.config.max_simulation_time):
            raise ValidationError(
                f"Simulation time {sim_time:.2e}s outside valid range "
                f"(1 as - {self.config.max_simulation_time:.2e}s)"
            )
        validated['simulation_time'] = float(sim_time)
        
        # Grid resolution validation
        resolution = params.get('grid_resolution', 10e-9)
        if not (0.1e-9 <= resolution <= 1e-6):  # 0.1 nm to 1 μm
            raise ValidationError(f"Grid resolution {resolution*1e9:.1f} nm outside valid range (0.1-1000 nm)")
        validated['grid_resolution'] = float(resolution)
        
        # Sample count validation
        samples = params.get('monte_carlo_samples', 1000)
        if not (1 <= samples <= 100000):
            raise ValidationError(f"Monte Carlo samples {samples} outside valid range (1-100000)")
        validated['monte_carlo_samples'] = int(samples)
        
        self._logger.info(f"Validated simulation parameters: {len(validated)} fields")
        return validated
    
    def _validate_numeric_parameter(self, name: str, value: Any) -> Union[float, int]:
        """Validate and sanitize numeric parameters."""
        if isinstance(value, (int, float)):
            if not (-1e20 <= value <= 1e20):  # Prevent overflow
                raise ValidationError(f"Parameter {name} value {value} too large")
            if isinstance(value, float) and not (value == value):  # NaN check
                raise ValidationError(f"Parameter {name} contains NaN")
            return value
        elif isinstance(value, str):
            try:
                num_value = float(value)
                return self._validate_numeric_parameter(name, num_value)
            except ValueError:
                raise ValidationError(f"Parameter {name} '{value}' is not numeric")
        else:
            raise ValidationError(f"Parameter {name} type {type(value)} not supported")


class OutputSanitizer:
    """Sanitize simulation outputs for security."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self._logger = logging.getLogger(__name__)
    
    def sanitize_simulation_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize simulation results before output."""
        if not self.config.enable_output_sanitization:
            return results
        
        sanitized = {}
        
        for key, value in results.items():
            if key.startswith('_') or key.endswith('_internal'):
                # Skip internal/private fields
                continue
            
            if isinstance(value, dict):
                sanitized[key] = self.sanitize_simulation_results(value)
            elif isinstance(value, list):
                sanitized[key] = self._sanitize_list(value)
            elif isinstance(value, (int, float, str, bool)):
                sanitized[key] = self._sanitize_scalar(value)
            else:
                # Skip complex objects that could contain sensitive data
                self._logger.warning(f"Skipping unsupported type {type(value)} for key {key}")
        
        # Add metadata
        sanitized['_sanitized'] = True
        sanitized['_sanitization_timestamp'] = time.time()
        
        return sanitized
    
    def _sanitize_list(self, items: List[Any]) -> List[Any]:
        """Sanitize list items."""
        sanitized = []
        for item in items[:1000]:  # Limit list size
            if isinstance(item, dict):
                sanitized.append(self.sanitize_simulation_results(item))
            elif isinstance(item, (int, float, str, bool)):
                sanitized.append(self._sanitize_scalar(item))
        return sanitized
    
    def _sanitize_scalar(self, value: Any) -> Any:
        """Sanitize scalar values."""
        if isinstance(value, float):
            # Handle special float values
            if not (value == value):  # NaN
                return None
            if value == float('inf') or value == float('-inf'):
                return None
            # Limit precision to prevent information leakage
            return round(value, 10)
        elif isinstance(value, str):
            # Limit string length and remove potentially dangerous characters
            clean_value = ''.join(c for c in value if c.isprintable())
            return clean_value[:1000]  # Limit length
        else:
            return value


class AuditLogger:
    """Audit logging for security events."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self._logger = logging.getLogger('photonic_security_audit')
        
        if config.enable_audit_logging:
            # Setup file handler for audit logs
            handler = logging.FileHandler('photonic_audit.log')
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)
    
    def log_simulation_start(self, session_id: str, params: Dict[str, Any]) -> None:
        """Log simulation start event."""
        if not self.config.enable_audit_logging:
            return
        
        self._logger.info(
            f"SIMULATION_START - Session: {session_id} - "
            f"Params: {self._sanitize_log_data(params)}"
        )
    
    def log_simulation_end(self, session_id: str, success: bool, duration: float) -> None:
        """Log simulation completion event."""
        if not self.config.enable_audit_logging:
            return
        
        status = "SUCCESS" if success else "FAILURE"
        self._logger.info(
            f"SIMULATION_END - Session: {session_id} - "
            f"Status: {status} - Duration: {duration:.3f}s"
        )
    
    def log_security_event(self, event_type: str, session_id: str, details: str) -> None:
        """Log security-related events."""
        if not self.config.enable_audit_logging:
            return
        
        self._logger.warning(
            f"SECURITY_EVENT - Type: {event_type} - Session: {session_id} - "
            f"Details: {details}"
        )
    
    def log_validation_failure(self, session_id: str, field: str, reason: str) -> None:
        """Log input validation failures."""
        if not self.config.enable_audit_logging:
            return
        
        self._logger.warning(
            f"VALIDATION_FAILURE - Session: {session_id} - "
            f"Field: {field} - Reason: {reason}"
        )
    
    def _sanitize_log_data(self, data: Dict[str, Any]) -> str:
        """Sanitize data for logging to prevent log injection."""
        # Convert to JSON and limit size
        try:
            json_str = json.dumps(data, default=str)
            # Remove potentially dangerous characters
            clean_str = ''.join(c for c in json_str if c.isprintable())
            return clean_str[:500]  # Limit log entry size
        except Exception:
            return "<data_sanitization_failed>"


class RateLimiter:
    """Rate limiting for API calls and simulations."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self._request_counts: Dict[str, List[float]] = {}
        self._logger = logging.getLogger(__name__)
    
    def check_rate_limit(self, identifier: str) -> bool:
        """Check if request is within rate limits."""
        if not self.config.rate_limit_requests:
            return True
        
        current_time = time.time()
        hour_ago = current_time - 3600  # 1 hour window
        
        # Initialize if new identifier
        if identifier not in self._request_counts:
            self._request_counts[identifier] = []
        
        # Clean old requests
        self._request_counts[identifier] = [
            req_time for req_time in self._request_counts[identifier]
            if req_time > hour_ago
        ]
        
        # Check limit
        if len(self._request_counts[identifier]) >= self.config.rate_limit_requests:
            self._logger.warning(f"Rate limit exceeded for {identifier}")
            return False
        
        # Add current request
        self._request_counts[identifier].append(current_time)
        return True


class SecureConfigManager:
    """Secure configuration management."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path('photonic_config.yaml')
        self._config_hash: Optional[str] = None
        self._logger = logging.getLogger(__name__)
    
    def load_config(self) -> SecurityConfig:
        """Load and validate security configuration."""
        if not self.config_path.exists():
            self._logger.info("No config file found, using defaults")
            return SecurityConfig()
        
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Validate configuration hash if present
            if '_config_hash' in config_data:
                expected_hash = config_data.pop('_config_hash')
                actual_hash = self._calculate_config_hash(config_data)
                if expected_hash != actual_hash:
                    raise SecurityError("Configuration hash mismatch - possible tampering")
            
            return SecurityConfig(**config_data)
            
        except Exception as e:
            self._logger.error(f"Failed to load config: {e}")
            raise SecurityError(f"Configuration loading failed: {e}")
    
    def save_config(self, config: SecurityConfig) -> None:
        """Save configuration with integrity hash."""
        config_data = config.__dict__.copy()
        
        # Calculate and add hash
        config_hash = self._calculate_config_hash(config_data)
        config_data['_config_hash'] = config_hash
        
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            
            self._logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            self._logger.error(f"Failed to save config: {e}")
            raise SecurityError(f"Configuration save failed: {e}")
    
    def _calculate_config_hash(self, config_data: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash of configuration data."""
        config_str = json.dumps(config_data, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()


def require_permission(permission: str):
    """Decorator to require specific permission for function access."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if first argument is a session
            if args and hasattr(args[0], 'session'):
                session = args[0].session
                if not session.has_permission(permission):
                    raise SecurityError(f"Permission '{permission}' required")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def rate_limited(identifier_func: Callable = None):
    """Decorator to apply rate limiting to functions."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get rate limiter from first argument if available
            if args and hasattr(args[0], 'rate_limiter'):
                rate_limiter = args[0].rate_limiter
                identifier = identifier_func(*args, **kwargs) if identifier_func else str(args[0])
                
                if not rate_limiter.check_rate_limit(identifier):
                    raise SecurityError("Rate limit exceeded")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


class SecurityManager:
    """Central security manager for photonic simulations."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_manager = SecureConfigManager(config_path)
        self.config = self.config_manager.load_config()
        
        self.input_validator = InputValidator(self.config)
        self.output_sanitizer = OutputSanitizer(self.config)
        self.audit_logger = AuditLogger(self.config)
        self.rate_limiter = RateLimiter(self.config)
        
        self._active_sessions: Dict[str, SecureSimulationSession] = {}
        self._logger = logging.getLogger(__name__)
    
    def create_session(self, user_id: Optional[str] = None, permissions: Optional[List[str]] = None) -> SecureSimulationSession:
        """Create a new secure simulation session."""
        session = SecureSimulationSession(
            user_id=user_id,
            permissions=permissions or ['read', 'simulate'],
            max_simulation_time=self.config.max_simulation_time,
            memory_limit=self.config.max_memory_usage
        )
        
        self._active_sessions[session.session_id] = session
        self._logger.info(f"Created session {session.session_id} for user {user_id}")
        
        return session
    
    def get_session(self, session_id: str) -> Optional[SecureSimulationSession]:
        """Get active session by ID."""
        session = self._active_sessions.get(session_id)
        
        if session and session.is_expired(self.config.session_timeout):
            self.revoke_session(session_id)
            return None
        
        if session:
            session.update_activity()
        
        return session
    
    def revoke_session(self, session_id: str) -> bool:
        """Revoke a simulation session."""
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]
            self._logger.info(f"Revoked session {session_id}")
            return True
        return False
    
    def validate_and_sanitize_request(
        self, 
        session: SecureSimulationSession,
        request_type: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate and sanitize a simulation request."""
        # Check rate limiting
        if not self.rate_limiter.check_rate_limit(session.session_id):
            self.audit_logger.log_security_event("RATE_LIMIT", session.session_id, request_type)
            raise SecurityError("Rate limit exceeded")
        
        # Validate request against session limits
        session.validate_simulation_request(parameters)
        
        # Validate inputs based on request type
        if request_type == 'optical_simulation':
            validated = self.input_validator.validate_optical_parameters(parameters)
        elif request_type == 'device_simulation':
            validated = self.input_validator.validate_device_geometry(parameters)
        elif request_type == 'general_simulation':
            validated = self.input_validator.validate_simulation_parameters(parameters)
        else:
            validated = parameters
        
        self.audit_logger.log_simulation_start(session.session_id, validated)
        return validated
    
    def sanitize_response(self, session: SecureSimulationSession, results: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize simulation response."""
        sanitized = self.output_sanitizer.sanitize_simulation_results(results)
        return sanitized


def create_secure_environment(config_path: Optional[Path] = None) -> SecurityManager:
    """Create a secure simulation environment."""
    return SecurityManager(config_path)