"""
Custom exceptions for photonic neuromorphics simulation framework.

Provides comprehensive error handling with detailed diagnostics for
photonic neural network simulation, RTL generation, and optical modeling.
"""

from typing import Optional, Dict, Any, List
import logging


class PhotonicNeuromorphicsException(Exception):
    """Base exception for photonic neuromorphics framework."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.logger = logging.getLogger(__name__)
        self.logger.error(f"{self.__class__.__name__}: {message}", extra=self.details)


class SimulationError(PhotonicNeuromorphicsException):
    """Errors during photonic simulation."""
    pass


class OpticalModelError(SimulationError):
    """Errors in optical component modeling."""
    
    def __init__(self, component: str, parameter: str, value: Any, message: str):
        self.component = component
        self.parameter = parameter
        self.value = value
        details = {
            "component": component,
            "parameter": parameter,
            "invalid_value": str(value),
            "error_type": "optical_model_error"
        }
        super().__init__(f"Optical model error in {component}.{parameter}: {message}", details)


class ConvergenceError(SimulationError):
    """Simulation convergence failures."""
    
    def __init__(self, iteration: int, tolerance: float, current_error: float):
        self.iteration = iteration
        self.tolerance = tolerance
        self.current_error = current_error
        details = {
            "iteration": iteration,
            "required_tolerance": tolerance,
            "current_error": current_error,
            "error_type": "convergence_error"
        }
        message = f"Simulation failed to converge after {iteration} iterations (error: {current_error:.2e}, tolerance: {tolerance:.2e})"
        super().__init__(message, details)


class NoiseModelError(SimulationError):
    """Errors in optical noise modeling."""
    pass


class RTLGenerationError(PhotonicNeuromorphicsException):
    """Errors during RTL generation."""
    pass


class SynthesisError(RTLGenerationError):
    """Synthesis-related errors."""
    
    def __init__(self, tool: str, error_message: str, log_file: Optional[str] = None):
        self.tool = tool
        self.log_file = log_file
        details = {
            "synthesis_tool": tool,
            "log_file": log_file,
            "error_type": "synthesis_error"
        }
        message = f"Synthesis failed using {tool}: {error_message}"
        super().__init__(message, details)


class ConstraintViolationError(RTLGenerationError):
    """Design constraint violations."""
    
    def __init__(self, constraint_type: str, required: float, actual: float, unit: str):
        self.constraint_type = constraint_type
        self.required = required
        self.actual = actual
        self.unit = unit
        details = {
            "constraint_type": constraint_type,
            "required_value": required,
            "actual_value": actual,
            "unit": unit,
            "violation_ratio": actual / required if required > 0 else float('inf'),
            "error_type": "constraint_violation"
        }
        message = f"Constraint violation: {constraint_type} = {actual:.3f} {unit} (required: ≤ {required:.3f} {unit})"
        super().__init__(message, details)


class NetworkTopologyError(PhotonicNeuromorphicsException):
    """Invalid network topology errors."""
    
    def __init__(self, topology: List[int], issue: str):
        self.topology = topology
        self.issue = issue
        details = {
            "topology": topology,
            "issue": issue,
            "error_type": "topology_error"
        }
        message = f"Invalid network topology {topology}: {issue}"
        super().__init__(message, details)


class WavelengthError(OpticalModelError):
    """Invalid wavelength specifications."""
    
    def __init__(self, wavelength: float, valid_range: tuple):
        self.wavelength = wavelength
        self.valid_range = valid_range
        details = {
            "wavelength_nm": wavelength * 1e9,
            "valid_range_nm": [r * 1e9 for r in valid_range],
            "error_type": "wavelength_error"
        }
        message = f"Wavelength {wavelength*1e9:.1f} nm outside valid range {valid_range[0]*1e9:.0f}-{valid_range[1]*1e9:.0f} nm"
        super().__init__("optical_system", "wavelength", wavelength, message)


class PowerBudgetError(OpticalModelError):
    """Optical power budget violations."""
    
    def __init__(self, required_power: float, available_power: float, loss_budget: Dict[str, float]):
        self.required_power = required_power
        self.available_power = available_power
        self.loss_budget = loss_budget
        details = {
            "required_power_mw": required_power * 1e3,
            "available_power_mw": available_power * 1e3,
            "loss_budget_db": loss_budget,
            "power_deficit_db": 10 * np.log10(required_power / available_power),
            "error_type": "power_budget_error"
        }
        message = f"Insufficient optical power: need {required_power*1e3:.1f} mW, have {available_power*1e3:.1f} mW"
        super().__init__("optical_system", "power_budget", required_power, message)


class TechnologyError(RTLGenerationError):
    """Unsupported technology node errors."""
    
    def __init__(self, requested_tech: str, supported_techs: List[str]):
        self.requested_tech = requested_tech
        self.supported_techs = supported_techs
        details = {
            "requested_technology": requested_tech,
            "supported_technologies": supported_techs,
            "error_type": "technology_error"
        }
        message = f"Unsupported technology '{requested_tech}'. Supported: {', '.join(supported_techs)}"
        super().__init__(message, details)


class MemoryError(RTLGenerationError):
    """Memory allocation/sizing errors."""
    
    def __init__(self, memory_type: str, required_bits: int, available_bits: int):
        self.memory_type = memory_type
        self.required_bits = required_bits
        self.available_bits = available_bits
        details = {
            "memory_type": memory_type,
            "required_bits": required_bits,
            "available_bits": available_bits,
            "utilization_percent": (required_bits / available_bits) * 100,
            "error_type": "memory_error"
        }
        message = f"Insufficient {memory_type} memory: need {required_bits} bits, have {available_bits} bits"
        super().__init__(message, details)


class ValidationError(PhotonicNeuromorphicsException):
    """Input validation errors."""
    
    def __init__(self, parameter: str, value: Any, expected_type: str, constraints: Optional[str] = None):
        self.parameter = parameter
        self.value = value
        self.expected_type = expected_type
        self.constraints = constraints
        details = {
            "parameter": parameter,
            "actual_value": str(value),
            "actual_type": type(value).__name__,
            "expected_type": expected_type,
            "constraints": constraints,
            "error_type": "validation_error"
        }
        message = f"Invalid {parameter}: got {value} ({type(value).__name__}), expected {expected_type}"
        if constraints:
            message += f" with constraints: {constraints}"
        super().__init__(message, details)


class SecurityError(PhotonicNeuromorphicsException):
    """Security-related errors."""
    
    def __init__(self, message: str, security_context: Optional[Dict[str, Any]] = None):
        details = security_context or {}
        details["error_type"] = "security_error"
        super().__init__(message, details)


class ConfigurationError(PhotonicNeuromorphicsException):
    """Configuration file or parameter errors."""
    
    def __init__(self, config_file: str, issue: str, suggestion: Optional[str] = None):
        self.config_file = config_file
        self.issue = issue
        self.suggestion = suggestion
        details = {
            "config_file": config_file,
            "issue": issue,
            "suggestion": suggestion,
            "error_type": "configuration_error"
        }
        message = f"Configuration error in {config_file}: {issue}"
        if suggestion:
            message += f". Suggestion: {suggestion}"
        super().__init__(message, details)


class ResourceExhaustionError(PhotonicNeuromorphicsException):
    """System resource exhaustion errors."""
    
    def __init__(self, resource_type: str, usage: float, limit: float, unit: str):
        self.resource_type = resource_type
        self.usage = usage
        self.limit = limit
        self.unit = unit
        details = {
            "resource_type": resource_type,
            "usage": usage,
            "limit": limit,
            "unit": unit,
            "usage_percent": (usage / limit) * 100,
            "error_type": "resource_exhaustion"
        }
        message = f"{resource_type} exhausted: using {usage:.1f} {unit} of {limit:.1f} {unit} limit"
        super().__init__(message, details)


class CompatibilityError(PhotonicNeuromorphicsException):
    """Version or compatibility errors."""
    
    def __init__(self, component: str, current_version: str, required_version: str):
        self.component = component
        self.current_version = current_version
        self.required_version = required_version
        details = {
            "component": component,
            "current_version": current_version,
            "required_version": required_version,
            "error_type": "compatibility_error"
        }
        message = f"Version incompatibility: {component} v{current_version} (required: v{required_version})"
        super().__init__(message, details)


def handle_exception_with_recovery(
    func,
    *args,
    max_retries: int = 3,
    recovery_strategies: Optional[List[callable]] = None,
    **kwargs
) -> Any:
    """
    Execute function with automatic recovery strategies.
    
    Args:
        func: Function to execute
        *args: Function arguments
        max_retries: Maximum number of retry attempts
        recovery_strategies: List of recovery functions to try
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
        
    Raises:
        PhotonicNeuromorphicsException: After all recovery attempts fail
    """
    logger = logging.getLogger(__name__)
    recovery_strategies = recovery_strategies or []
    
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except PhotonicNeuromorphicsException as e:
            last_exception = e
            logger.warning(f"Attempt {attempt + 1} failed: {e.message}")
            
            if attempt < max_retries:
                # Try recovery strategies
                for i, strategy in enumerate(recovery_strategies):
                    try:
                        logger.info(f"Attempting recovery strategy {i + 1}")
                        strategy(e, attempt)
                        break
                    except Exception as recovery_error:
                        logger.warning(f"Recovery strategy {i + 1} failed: {recovery_error}")
                        continue
            else:
                logger.error(f"All {max_retries + 1} attempts failed")
                break
        except Exception as e:
            # Convert unexpected exceptions to framework exceptions
            last_exception = PhotonicNeuromorphicsException(
                f"Unexpected error in {func.__name__}: {str(e)}",
                {"function": func.__name__, "attempt": attempt + 1, "original_error": str(e)}
            )
            if attempt >= max_retries:
                break
    
    raise last_exception


class ExceptionContext:
    """Context manager for enhanced exception handling."""
    
    def __init__(self, operation: str, **context_info):
        self.operation = operation
        self.context_info = context_info
        self.logger = logging.getLogger(__name__)
    
    def __enter__(self):
        self.logger.debug(f"Starting operation: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            if issubclass(exc_type, PhotonicNeuromorphicsException):
                # Add context information to existing exception
                exc_val.details.update(self.context_info)
                exc_val.details["operation"] = self.operation
            else:
                # Convert to framework exception with context
                context_details = self.context_info.copy()
                context_details.update({
                    "operation": self.operation,
                    "original_exception_type": exc_type.__name__,
                    "original_exception_message": str(exc_val)
                })
                
                new_exception = PhotonicNeuromorphicsException(
                    f"Error in {self.operation}: {str(exc_val)}",
                    context_details
                )
                
                # Preserve original traceback
                raise new_exception from exc_val
        
        self.logger.debug(f"Completed operation: {self.operation}")
        return False  # Don't suppress the exception


def validate_optical_parameters(
    wavelength: float,
    power: float,
    loss: float,
    efficiency: float
) -> None:
    """
    Validate optical parameters with comprehensive error checking.
    
    Args:
        wavelength: Operating wavelength in meters
        power: Optical power in watts
        loss: Loss coefficient in dB/cm
        efficiency: Coupling/detector efficiency (0-1)
        
    Raises:
        ValidationError: If any parameter is invalid
    """
    # Wavelength validation (common optical communication bands)
    if not (1260e-9 <= wavelength <= 1675e-9):  # O, E, S, C, L bands
        raise WavelengthError(wavelength, (1260e-9, 1675e-9))
    
    # Power validation
    if power <= 0 or power > 1.0:  # Max 1W for safety
        raise ValidationError(
            "optical_power", power, "positive float",
            "0 < power <= 1.0 W"
        )
    
    # Loss validation
    if loss < 0 or loss > 100:  # Reasonable loss range
        raise ValidationError(
            "propagation_loss", loss, "positive float",
            "0 <= loss <= 100 dB/cm"
        )
    
    # Efficiency validation
    if not (0 <= efficiency <= 1):
        raise ValidationError(
            "optical_efficiency", efficiency, "float",
            "0.0 <= efficiency <= 1.0"
        )


def validate_network_topology(topology: List[int]) -> None:
    """
    Validate neural network topology.
    
    Args:
        topology: List of layer sizes
        
    Raises:
        NetworkTopologyError: If topology is invalid
    """
    if len(topology) < 2:
        raise NetworkTopologyError(topology, "Need at least 2 layers (input and output)")
    
    if any(size <= 0 for size in topology):
        raise NetworkTopologyError(topology, "All layer sizes must be positive")
    
    if any(size > 10000 for size in topology):
        raise NetworkTopologyError(topology, "Layer size exceeds maximum (10000 neurons)")
    
    # Check for reasonable scaling
    max_size = max(topology)
    min_size = min(topology)
    if max_size / min_size > 1000:
        raise NetworkTopologyError(
            topology, 
            f"Extreme size ratio ({max_size}/{min_size} = {max_size/min_size:.0f}x) may cause numerical issues"
        )


# Import numpy for power budget calculations
try:
    import numpy as np
except ImportError:
    # Fallback for log calculations if numpy not available
    import math
    np = type('numpy_fallback', (), {
        'log10': math.log10
    })()