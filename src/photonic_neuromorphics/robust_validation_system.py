"""
Robust Validation System for Photonic Neuromorphic Computing

Comprehensive validation framework with advanced error handling, input validation,
output sanitization, and real-time monitoring for production-ready systems.
"""

import json
import time
import hashlib
import logging
import warnings
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import traceback
from functools import wraps


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"
    NORMAL = "normal"
    RELAXED = "relaxed"
    DEVELOPMENT = "development"


class ValidationCategory(Enum):
    """Categories of validation."""
    INPUT_VALIDATION = "input_validation"
    OUTPUT_SANITIZATION = "output_sanitization"
    PARAMETER_BOUNDS = "parameter_bounds"
    PHYSICAL_CONSTRAINTS = "physical_constraints"
    SECURITY_CHECKS = "security_checks"


@dataclass
class ValidationRule:
    """Individual validation rule definition."""
    name: str
    category: ValidationCategory
    validator: Callable[[Any], bool]
    error_message: str
    severity: str = "error"  # error, warning, info
    auto_fix: Optional[Callable[[Any], Any]] = None
    
    def validate(self, value: Any) -> Tuple[bool, str, Any]:
        """Execute validation rule."""
        try:
            is_valid = self.validator(value)
            if not is_valid and self.auto_fix:
                fixed_value = self.auto_fix(value)
                return True, f"Auto-fixed: {self.error_message}", fixed_value
            return is_valid, self.error_message if not is_valid else "", value
        except Exception as e:
            return False, f"Validation error in {self.name}: {str(e)}", value


@dataclass
class ValidationResult:
    """Result of validation process."""
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    auto_fixes: List[str] = field(default_factory=list)
    validation_time: float = 0.0
    validated_data: Any = None
    
    def add_result(self, rule_name: str, passed: bool, message: str, severity: str, fixed_data: Any = None):
        """Add validation result."""
        if fixed_data is not None:
            self.auto_fixes.append(f"{rule_name}: {message}")
            self.validated_data = fixed_data
        elif not passed:
            if severity == "error":
                self.errors.append(f"{rule_name}: {message}")
            elif severity == "warning":
                self.warnings.append(f"{rule_name}: {message}")
            else:
                self.info.append(f"{rule_name}: {message}")
        
        self.passed = self.passed and (passed or severity != "error")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            'passed': self.passed,
            'total_errors': len(self.errors),
            'total_warnings': len(self.warnings),
            'total_info': len(self.info),
            'auto_fixes_applied': len(self.auto_fixes),
            'validation_time': self.validation_time
        }


class PhotonicValidationFramework:
    """Comprehensive validation framework for photonic systems."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.NORMAL):
        self.validation_level = validation_level
        self.validation_rules = {}
        self.validation_history = []
        self.performance_metrics = {
            'total_validations': 0,
            'successful_validations': 0,
            'auto_fixes_applied': 0,
            'avg_validation_time': 0.0
        }
        
        self._initialize_default_rules()
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup validation logging."""
        self.logger = logging.getLogger('photonic_validation')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _initialize_default_rules(self):
        """Initialize default validation rules for photonic systems."""
        
        # Input validation rules
        self.add_validation_rule(ValidationRule(
            name="wavelength_range",
            category=ValidationCategory.INPUT_VALIDATION,
            validator=lambda x: isinstance(x, (int, float)) and 1e-6 <= x <= 10e-6,
            error_message="Wavelength must be between 1-10 micrometers",
            auto_fix=lambda x: max(1e-6, min(10e-6, float(x))) if isinstance(x, (int, float)) else 1550e-9
        ))
        
        self.add_validation_rule(ValidationRule(
            name="power_range",
            category=ValidationCategory.INPUT_VALIDATION,
            validator=lambda x: isinstance(x, (int, float)) and 0 < x <= 1.0,
            error_message="Power must be between 0 and 1 Watt",
            auto_fix=lambda x: max(1e-6, min(1.0, float(x))) if isinstance(x, (int, float)) else 1e-3
        ))
        
        self.add_validation_rule(ValidationRule(
            name="efficiency_bounds",
            category=ValidationCategory.PARAMETER_BOUNDS,
            validator=lambda x: isinstance(x, (int, float)) and 0 <= x <= 1.0,
            error_message="Efficiency must be between 0 and 1",
            auto_fix=lambda x: max(0.0, min(1.0, float(x))) if isinstance(x, (int, float)) else 0.8
        ))
        
        self.add_validation_rule(ValidationRule(
            name="temperature_range",
            category=ValidationCategory.PHYSICAL_CONSTRAINTS,
            validator=lambda x: isinstance(x, (int, float)) and 200 <= x <= 400,
            error_message="Temperature must be between 200-400 K",
            auto_fix=lambda x: max(200, min(400, float(x))) if isinstance(x, (int, float)) else 298.15
        ))
        
        self.add_validation_rule(ValidationRule(
            name="layer_sizes_positive",
            category=ValidationCategory.INPUT_VALIDATION,
            validator=lambda x: isinstance(x, list) and all(isinstance(size, int) and size > 0 for size in x),
            error_message="Layer sizes must be positive integers",
            auto_fix=lambda x: [max(1, int(size)) for size in x if isinstance(size, (int, float))] if isinstance(x, list) else [100, 50, 10]
        ))
        
        # Security validation rules
        self.add_validation_rule(ValidationRule(
            name="no_code_injection",
            category=ValidationCategory.SECURITY_CHECKS,
            validator=lambda x: not any(dangerous in str(x).lower() for dangerous in ['exec', 'eval', 'import', '__']),
            error_message="Potentially dangerous code patterns detected",
            severity="error"
        ))
        
        self.add_validation_rule(ValidationRule(
            name="reasonable_data_size",
            category=ValidationCategory.SECURITY_CHECKS,
            validator=lambda x: len(str(x)) < 10000,  # 10KB limit
            error_message="Data size exceeds reasonable limits",
            auto_fix=lambda x: str(x)[:9999] if len(str(x)) >= 10000 else x
        ))
    
    def add_validation_rule(self, rule: ValidationRule):
        """Add a custom validation rule."""
        if rule.category not in self.validation_rules:
            self.validation_rules[rule.category] = []
        self.validation_rules[rule.category].append(rule)
    
    def validate_data(self, data: Any, categories: Optional[List[ValidationCategory]] = None) -> ValidationResult:
        """Validate data against specified categories of rules."""
        start_time = time.time()
        result = ValidationResult(passed=True, validated_data=data)
        
        # Determine which categories to validate
        if categories is None:
            categories = list(self.validation_rules.keys())
        
        # Apply validation rules
        for category in categories:
            if category in self.validation_rules:
                for rule in self.validation_rules[category]:
                    try:
                        # Skip strict rules in relaxed mode
                        if (self.validation_level == ValidationLevel.RELAXED and 
                            rule.severity == "error" and 
                            category != ValidationCategory.SECURITY_CHECKS):
                            continue
                        
                        is_valid, message, fixed_data = rule.validate(data)
                        result.add_result(rule.name, is_valid, message, rule.severity, fixed_data)
                        
                        # Update data if auto-fixed
                        if fixed_data is not None and fixed_data != data:
                            data = fixed_data
                            result.validated_data = data
                            self.performance_metrics['auto_fixes_applied'] += 1
                    
                    except Exception as e:
                        error_msg = f"Exception in validation rule {rule.name}: {str(e)}"
                        result.add_result(rule.name, False, error_msg, "error")
                        self.logger.error(error_msg)
        
        # Finalize result
        result.validation_time = time.time() - start_time
        result.validated_data = data if result.validated_data is None else result.validated_data
        
        # Update metrics
        self.performance_metrics['total_validations'] += 1
        if result.passed:
            self.performance_metrics['successful_validations'] += 1
        
        # Update average validation time
        current_avg = self.performance_metrics['avg_validation_time']
        total_validations = self.performance_metrics['total_validations']
        self.performance_metrics['avg_validation_time'] = (
            (current_avg * (total_validations - 1) + result.validation_time) / total_validations
        )
        
        # Store validation history
        self.validation_history.append({
            'timestamp': time.time(),
            'result_summary': result.get_summary(),
            'data_hash': hashlib.md5(str(data).encode()).hexdigest()[:8]
        })
        
        # Log validation result
        if result.passed:
            self.logger.info(f"Validation passed in {result.validation_time:.4f}s")
        else:
            self.logger.warning(f"Validation failed: {len(result.errors)} errors, {len(result.warnings)} warnings")
        
        return result
    
    def validate_photonic_parameters(self, params: Dict[str, Any]) -> ValidationResult:
        """Validate photonic-specific parameters."""
        result = ValidationResult(passed=True, validated_data=params.copy())
        
        # Validate individual parameters
        parameter_validations = [
            ('wavelength', ValidationCategory.INPUT_VALIDATION),
            ('power', ValidationCategory.INPUT_VALIDATION),
            ('efficiency', ValidationCategory.PARAMETER_BOUNDS),
            ('temperature', ValidationCategory.PHYSICAL_CONSTRAINTS)
        ]
        
        for param_name, category in parameter_validations:
            if param_name in params:
                param_result = self.validate_data(params[param_name], [category])
                if not param_result.passed:
                    result.passed = False
                    result.errors.extend([f"{param_name}: {error}" for error in param_result.errors])
                    result.warnings.extend([f"{param_name}: {warning}" for warning in param_result.warnings])
                else:
                    # Update with validated/fixed value
                    result.validated_data[param_name] = param_result.validated_data
        
        # Cross-parameter validation
        validated_params = result.validated_data
        
        # Check power-efficiency relationship
        if 'power' in validated_params and 'efficiency' in validated_params:
            max_theoretical_efficiency = min(0.95, 1.0 - validated_params['power'] * 0.1)
            if validated_params['efficiency'] > max_theoretical_efficiency:
                result.warnings.append(
                    f"Efficiency {validated_params['efficiency']:.2f} may be too high for power level {validated_params['power']:.3f}W"
                )
        
        # Check wavelength-dependent losses
        if 'wavelength' in validated_params:
            wavelength = validated_params['wavelength']
            if wavelength < 1.2e-6 or wavelength > 1.8e-6:  # Outside telecom bands
                result.warnings.append(
                    f"Wavelength {wavelength*1e9:.1f}nm is outside optimal telecom bands (1200-1800nm)"
                )
        
        return result
    
    def validate_network_topology(self, topology: List[int]) -> ValidationResult:
        """Validate neural network topology."""
        result = self.validate_data(topology, [ValidationCategory.INPUT_VALIDATION])
        
        if result.passed:
            # Additional topology-specific validations
            validated_topology = result.validated_data
            
            # Check for reasonable topology
            if len(validated_topology) < 2:
                result.passed = False
                result.errors.append("Network must have at least input and output layers")
            
            # Check for reasonable layer size ratios
            for i in range(len(validated_topology) - 1):
                ratio = validated_topology[i] / validated_topology[i + 1]
                if ratio > 10:  # More than 10:1 ratio
                    result.warnings.append(
                        f"Large layer size ratio ({ratio:.1f}:1) between layers {i} and {i+1}"
                    )
            
            # Check total network size
            total_neurons = sum(validated_topology)
            if total_neurons > 100000:  # 100K neurons
                result.warnings.append(
                    f"Large network size ({total_neurons} neurons) may impact performance"
                )
        
        return result
    
    def create_validation_context(self, strict_mode: bool = False):
        """Create a validation context manager."""
        return ValidationContext(self, strict_mode)
    
    def get_validation_metrics(self) -> Dict[str, Any]:
        """Get validation performance metrics."""
        metrics = self.performance_metrics.copy()
        
        if metrics['total_validations'] > 0:
            metrics['success_rate'] = metrics['successful_validations'] / metrics['total_validations']
            metrics['auto_fix_rate'] = metrics['auto_fixes_applied'] / metrics['total_validations']
        else:
            metrics['success_rate'] = 0.0
            metrics['auto_fix_rate'] = 0.0
        
        # Recent validation trends
        recent_validations = self.validation_history[-100:]  # Last 100 validations
        if recent_validations:
            recent_success_count = sum(1 for v in recent_validations if v['result_summary']['passed'])
            metrics['recent_success_rate'] = recent_success_count / len(recent_validations)
            
            recent_times = [v['result_summary']['validation_time'] for v in recent_validations]
            metrics['recent_avg_time'] = sum(recent_times) / len(recent_times)
        
        return metrics
    
    def reset_metrics(self):
        """Reset validation metrics."""
        self.performance_metrics = {
            'total_validations': 0,
            'successful_validations': 0,
            'auto_fixes_applied': 0,
            'avg_validation_time': 0.0
        }
        self.validation_history.clear()


class ValidationContext:
    """Context manager for validation operations."""
    
    def __init__(self, validator: PhotonicValidationFramework, strict_mode: bool = False):
        self.validator = validator
        self.strict_mode = strict_mode
        self.original_level = validator.validation_level
        self.validation_results = []
    
    def __enter__(self):
        if self.strict_mode:
            self.validator.validation_level = ValidationLevel.STRICT
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.validator.validation_level = self.original_level
        
        if exc_type is not None:
            self.validator.logger.error(f"Exception in validation context: {exc_val}")
        
        return False  # Don't suppress exceptions
    
    def validate(self, data: Any, categories: Optional[List[ValidationCategory]] = None) -> ValidationResult:
        """Validate data within this context."""
        result = self.validator.validate_data(data, categories)
        self.validation_results.append(result)
        return result
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of all validations in this context."""
        if not self.validation_results:
            return {'total_validations': 0}
        
        total_passed = sum(1 for r in self.validation_results if r.passed)
        total_errors = sum(len(r.errors) for r in self.validation_results)
        total_warnings = sum(len(r.warnings) for r in self.validation_results)
        total_time = sum(r.validation_time for r in self.validation_results)
        
        return {
            'total_validations': len(self.validation_results),
            'passed_validations': total_passed,
            'success_rate': total_passed / len(self.validation_results),
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'total_validation_time': total_time,
            'avg_validation_time': total_time / len(self.validation_results)
        }


def validated_function(categories: Optional[List[ValidationCategory]] = None, 
                      auto_fix: bool = True,
                      validator: Optional[PhotonicValidationFramework] = None):
    """Decorator for automatic function parameter validation."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Use global validator if none provided
            if validator is None:
                global_validator = PhotonicValidationFramework()
            else:
                global_validator = validator
            
            # Validate function arguments
            validation_errors = []
            validated_kwargs = kwargs.copy()
            
            for arg_name, arg_value in kwargs.items():
                result = global_validator.validate_data(arg_value, categories)
                
                if not result.passed and not auto_fix:
                    validation_errors.extend(result.errors)
                elif auto_fix and result.validated_data != arg_value:
                    validated_kwargs[arg_name] = result.validated_data
            
            if validation_errors:
                raise ValueError(f"Validation failed in {func.__name__}: {'; '.join(validation_errors)}")
            
            # Call function with validated parameters
            return func(*args, **validated_kwargs)
        
        return wrapper
    return decorator


# Example usage and demonstration
class RobustPhotonicSystem:
    """Example of a robust photonic system with comprehensive validation."""
    
    def __init__(self):
        self.validator = PhotonicValidationFramework(ValidationLevel.NORMAL)
        self.system_state = {}
    
    @validated_function(categories=[ValidationCategory.INPUT_VALIDATION, ValidationCategory.PARAMETER_BOUNDS])
    def configure_system(self, wavelength: float = 1550e-9, power: float = 1e-3, efficiency: float = 0.8):
        """Configure photonic system with validation."""
        config = {
            'wavelength': wavelength,
            'power': power,
            'efficiency': efficiency,
            'timestamp': time.time()
        }
        
        # Additional cross-parameter validation
        validation_result = self.validator.validate_photonic_parameters(config)
        
        if validation_result.passed:
            self.system_state.update(validation_result.validated_data)
            return True, "System configured successfully"
        else:
            error_msg = f"Configuration failed: {'; '.join(validation_result.errors)}"
            return False, error_msg
    
    def create_network(self, topology: List[int]) -> Tuple[bool, str]:
        """Create neural network with topology validation."""
        validation_result = self.validator.validate_network_topology(topology)
        
        if validation_result.passed:
            self.system_state['network_topology'] = validation_result.validated_data
            return True, f"Network created with topology {validation_result.validated_data}"
        else:
            return False, f"Invalid topology: {'; '.join(validation_result.errors)}"
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health including validation metrics."""
        return {
            'system_state': self.system_state,
            'validation_metrics': self.validator.get_validation_metrics(),
            'validation_level': self.validator.validation_level.value,
            'health_status': 'healthy' if self.validator.get_validation_metrics().get('success_rate', 0) > 0.8 else 'degraded'
        }


def demonstrate_robust_validation():
    """Demonstrate robust validation system."""
    print("🛡️ Demonstrating Robust Validation System")
    print("=" * 50)
    
    # Create robust photonic system
    system = RobustPhotonicSystem()
    
    # Test valid configuration
    print("\n1. Testing valid configuration:")
    success, message = system.configure_system(wavelength=1550e-9, power=1e-3, efficiency=0.8)
    print(f"   Result: {success}, Message: {message}")
    
    # Test invalid configuration (auto-fix)
    print("\n2. Testing invalid configuration with auto-fix:")
    success, message = system.configure_system(wavelength=15e-6, power=2.0, efficiency=1.5)  # Out of bounds
    print(f"   Result: {success}, Message: {message}")
    
    # Test network topology
    print("\n3. Testing network topology validation:")
    success, message = system.create_network([784, 256, 128, 10])
    print(f"   Result: {success}, Message: {message}")
    
    # Test validation context
    print("\n4. Testing validation context:")
    with system.validator.create_validation_context(strict_mode=True) as ctx:
        result1 = ctx.validate(1550e-9, [ValidationCategory.INPUT_VALIDATION])
        result2 = ctx.validate([100, 50, 10], [ValidationCategory.INPUT_VALIDATION])
        
        context_summary = ctx.get_context_summary()
        print(f"   Context validations: {context_summary['total_validations']}")
        print(f"   Success rate: {context_summary['success_rate']:.2f}")
    
    # Get system health
    print("\n5. System health report:")
    health = system.get_system_health()
    print(f"   Health status: {health['health_status']}")
    print(f"   Total validations: {health['validation_metrics']['total_validations']}")
    print(f"   Success rate: {health['validation_metrics'].get('success_rate', 0):.2f}")
    print(f"   Auto-fixes applied: {health['validation_metrics']['auto_fixes_applied']}")
    
    return health


if __name__ == "__main__":
    demonstrate_robust_validation()