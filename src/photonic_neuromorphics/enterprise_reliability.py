"""
Enterprise Reliability Framework for Photonic Neuromorphic Systems

Production-grade reliability features including fault tolerance, self-healing,
graceful degradation, circuit breakers, and comprehensive health checking.
"""

import time
import threading
import logging
import json
import random
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from contextlib import contextmanager
import traceback


class SystemState(Enum):
    """Overall system health states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"
    RECOVERING = "recovering"
    MAINTENANCE = "maintenance"


class ComponentHealth(Enum):
    """Individual component health states."""
    OPERATIONAL = "operational"
    WARNING = "warning"
    ERROR = "error"
    OFFLINE = "offline"


class FailureMode(Enum):
    """Types of failures that can occur."""
    OPTICAL_LOSS = "optical_loss"
    THERMAL_RUNAWAY = "thermal_runaway"
    COUPLING_DEGRADATION = "coupling_degradation"
    QUANTUM_DECOHERENCE = "quantum_decoherence"
    PROCESSING_TIMEOUT = "processing_timeout"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    NETWORK_PARTITION = "network_partition"
    POWER_FLUCTUATION = "power_fluctuation"


@dataclass
class HealthCheck:
    """Health check definition."""
    name: str
    check_function: Callable[[], Tuple[bool, str, Dict[str, Any]]]
    interval_seconds: float = 30.0
    timeout_seconds: float = 5.0
    failure_threshold: int = 3
    recovery_threshold: int = 2
    current_failures: int = 0
    current_successes: int = 0
    last_check_time: float = 0.0
    last_result: Optional[Tuple[bool, str, Dict[str, Any]]] = None


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3
    success_threshold: int = 2


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking calls
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.half_open_calls = 0
        self.lock = threading.RLock()
        
        # Metrics
        self.total_calls = 0
        self.total_failures = 0
        self.state_changes = []
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        with self.lock:
            self.total_calls += 1
            
            # Check if circuit is open
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time < self.config.recovery_timeout:
                    raise Exception(f"Circuit breaker {self.name} is OPEN")
                else:
                    # Try to move to half-open
                    self._change_state(CircuitBreakerState.HALF_OPEN)
            
            # In half-open state, limit the number of calls
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    raise Exception(f"Circuit breaker {self.name} half-open limit exceeded")
                self.half_open_calls += 1
        
        # Execute the function
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            raise e
    
    def _record_success(self):
        """Record successful call."""
        with self.lock:
            self.success_count += 1
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.success_count >= self.config.success_threshold:
                    self._change_state(CircuitBreakerState.CLOSED)
                    self.failure_count = 0
                    self.success_count = 0
    
    def _record_failure(self):
        """Record failed call."""
        with self.lock:
            self.failure_count += 1
            self.total_failures += 1
            self.last_failure_time = time.time()
            
            if (self.state == CircuitBreakerState.CLOSED and 
                self.failure_count >= self.config.failure_threshold):
                self._change_state(CircuitBreakerState.OPEN)
            elif self.state == CircuitBreakerState.HALF_OPEN:
                self._change_state(CircuitBreakerState.OPEN)
    
    def _change_state(self, new_state: CircuitBreakerState):
        """Change circuit breaker state."""
        old_state = self.state
        self.state = new_state
        self.half_open_calls = 0
        
        # Record state change
        self.state_changes.append({
            'timestamp': time.time(),
            'from_state': old_state.value,
            'to_state': new_state.value,
            'failure_count': self.failure_count
        })
        
        logging.info(f"Circuit breaker {self.name}: {old_state.value} -> {new_state.value}")
    
    def force_open(self):
        """Force circuit breaker open (for maintenance)."""
        with self.lock:
            self._change_state(CircuitBreakerState.OPEN)
    
    def force_close(self):
        """Force circuit breaker closed (after maintenance)."""
        with self.lock:
            self._change_state(CircuitBreakerState.CLOSED)
            self.failure_count = 0
            self.success_count = 0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        with self.lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'total_calls': self.total_calls,
                'total_failures': self.total_failures,
                'failure_rate': self.total_failures / max(self.total_calls, 1),
                'last_failure_time': self.last_failure_time,
                'state_changes': self.state_changes[-10:]  # Last 10 state changes
            }


class HealthChecker:
    """Manages health checks for system components."""
    
    def __init__(self):
        self.health_checks = {}
        self.check_results = defaultdict(list)
        self.checking_active = False
        self.check_thread = None
        self.lock = threading.RLock()
        
        # Overall health state
        self.system_health = ComponentHealth.OPERATIONAL
        self.health_history = deque(maxlen=1000)
    
    def add_health_check(self, health_check: HealthCheck):
        """Add a health check."""
        with self.lock:
            self.health_checks[health_check.name] = health_check
    
    def remove_health_check(self, name: str):
        """Remove a health check."""
        with self.lock:
            if name in self.health_checks:
                del self.health_checks[name]
            if name in self.check_results:
                del self.check_results[name]
    
    def start_health_checking(self):
        """Start continuous health checking."""
        if self.checking_active:
            return
        
        self.checking_active = True
        self.check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.check_thread.start()
        logging.info("Health checking started")
    
    def stop_health_checking(self):
        """Stop health checking."""
        self.checking_active = False
        if self.check_thread:
            self.check_thread.join(timeout=5.0)
        logging.info("Health checking stopped")
    
    def _health_check_loop(self):
        """Main health checking loop."""
        while self.checking_active:
            current_time = time.time()
            
            # Run due health checks
            for check_name, health_check in list(self.health_checks.items()):
                if current_time - health_check.last_check_time >= health_check.interval_seconds:
                    self._run_health_check(health_check)
            
            # Update overall system health
            self._update_system_health()
            
            time.sleep(1.0)  # Check every second for due health checks
    
    def _run_health_check(self, health_check: HealthCheck):
        """Run an individual health check."""
        try:
            # Set timeout for health check
            start_time = time.time()
            
            # Execute health check
            success, message, details = health_check.check_function()
            
            # Check timeout
            if time.time() - start_time > health_check.timeout_seconds:
                success = False
                message = f"Health check timeout ({health_check.timeout_seconds}s)"
            
            # Update health check state
            health_check.last_check_time = time.time()
            health_check.last_result = (success, message, details)
            
            if success:
                health_check.current_failures = 0
                health_check.current_successes += 1
            else:
                health_check.current_failures += 1
                health_check.current_successes = 0
            
            # Store result
            result_record = {
                'timestamp': time.time(),
                'success': success,
                'message': message,
                'details': details,
                'check_duration': time.time() - start_time
            }
            
            with self.lock:
                self.check_results[health_check.name].append(result_record)
                # Keep only last 100 results per check
                if len(self.check_results[health_check.name]) > 100:
                    self.check_results[health_check.name].pop(0)
        
        except Exception as e:
            # Health check failed with exception
            health_check.last_check_time = time.time()
            health_check.current_failures += 1
            health_check.current_successes = 0
            
            error_message = f"Health check exception: {str(e)}"
            health_check.last_result = (False, error_message, {'exception': str(e)})
            
            logging.error(f"Health check {health_check.name} failed: {e}")
    
    def _update_system_health(self):
        """Update overall system health based on component health."""
        component_states = []
        
        for check_name, health_check in self.health_checks.items():
            if health_check.current_failures >= health_check.failure_threshold:
                component_states.append(ComponentHealth.ERROR)
            elif health_check.current_failures > 0:
                component_states.append(ComponentHealth.WARNING)
            else:
                component_states.append(ComponentHealth.OPERATIONAL)
        
        # Determine overall health
        if not component_states:
            new_health = ComponentHealth.OPERATIONAL
        elif ComponentHealth.ERROR in component_states:
            new_health = ComponentHealth.ERROR
        elif ComponentHealth.WARNING in component_states:
            new_health = ComponentHealth.WARNING
        else:
            new_health = ComponentHealth.OPERATIONAL
        
        # Update system health if changed
        if new_health != self.system_health:
            old_health = self.system_health
            self.system_health = new_health
            
            health_change = {
                'timestamp': time.time(),
                'from_health': old_health.value,
                'to_health': new_health.value,
                'component_states': [state.value for state in component_states]
            }
            
            self.health_history.append(health_change)
            logging.info(f"System health changed: {old_health.value} -> {new_health.value}")
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        with self.lock:
            component_health = {}
            
            for check_name, health_check in self.health_checks.items():
                last_result = health_check.last_result
                
                if health_check.current_failures >= health_check.failure_threshold:
                    status = ComponentHealth.ERROR.value
                elif health_check.current_failures > 0:
                    status = ComponentHealth.WARNING.value
                else:
                    status = ComponentHealth.OPERATIONAL.value
                
                component_health[check_name] = {
                    'status': status,
                    'current_failures': health_check.current_failures,
                    'current_successes': health_check.current_successes,
                    'last_check_time': health_check.last_check_time,
                    'last_result': {
                        'success': last_result[0] if last_result else None,
                        'message': last_result[1] if last_result else None
                    } if last_result else None
                }
            
            return {
                'system_health': self.system_health.value,
                'component_health': component_health,
                'total_health_checks': len(self.health_checks),
                'checking_active': self.checking_active,
                'health_changes': list(self.health_history)[-10:]  # Last 10 changes
            }


class SelfHealingSystem:
    """Self-healing capabilities for photonic neuromorphic systems."""
    
    def __init__(self):
        self.healing_strategies = {}
        self.healing_history = deque(maxlen=100)
        self.auto_healing_enabled = True
        self.lock = threading.RLock()
    
    def register_healing_strategy(self, failure_mode: FailureMode, 
                                strategy: Callable[[Dict[str, Any]], bool]):
        """Register a healing strategy for a failure mode."""
        with self.lock:
            self.healing_strategies[failure_mode] = strategy
    
    def attempt_healing(self, failure_mode: FailureMode, 
                       context: Dict[str, Any]) -> Tuple[bool, str]:
        """Attempt to heal a specific failure mode."""
        if not self.auto_healing_enabled:
            return False, "Auto-healing is disabled"
        
        with self.lock:
            if failure_mode not in self.healing_strategies:
                return False, f"No healing strategy for {failure_mode.value}"
            
            healing_start = time.time()
            
            try:
                strategy = self.healing_strategies[failure_mode]
                success = strategy(context)
                
                healing_duration = time.time() - healing_start
                
                # Record healing attempt
                healing_record = {
                    'timestamp': time.time(),
                    'failure_mode': failure_mode.value,
                    'success': success,
                    'duration': healing_duration,
                    'context': context
                }
                
                self.healing_history.append(healing_record)
                
                if success:
                    message = f"Successfully healed {failure_mode.value}"
                    logging.info(message)
                else:
                    message = f"Failed to heal {failure_mode.value}"
                    logging.warning(message)
                
                return success, message
            
            except Exception as e:
                error_message = f"Healing strategy exception for {failure_mode.value}: {str(e)}"
                logging.error(error_message)
                
                healing_record = {
                    'timestamp': time.time(),
                    'failure_mode': failure_mode.value,
                    'success': False,
                    'duration': time.time() - healing_start,
                    'context': context,
                    'error': str(e)
                }
                
                self.healing_history.append(healing_record)
                return False, error_message
    
    def get_healing_statistics(self) -> Dict[str, Any]:
        """Get self-healing statistics."""
        with self.lock:
            if not self.healing_history:
                return {'total_attempts': 0}
            
            total_attempts = len(self.healing_history)
            successful_attempts = sum(1 for record in self.healing_history if record['success'])
            
            failure_mode_stats = defaultdict(lambda: {'attempts': 0, 'successes': 0})
            
            for record in self.healing_history:
                mode = record['failure_mode']
                failure_mode_stats[mode]['attempts'] += 1
                if record['success']:
                    failure_mode_stats[mode]['successes'] += 1
            
            return {
                'total_attempts': total_attempts,
                'successful_attempts': successful_attempts,
                'success_rate': successful_attempts / total_attempts,
                'failure_mode_stats': dict(failure_mode_stats),
                'auto_healing_enabled': self.auto_healing_enabled,
                'recent_attempts': list(self.healing_history)[-10:]
            }


class EnterpriseReliabilityFramework:
    """Comprehensive enterprise reliability framework."""
    
    def __init__(self):
        self.health_checker = HealthChecker()
        self.circuit_breakers = {}
        self.self_healing = SelfHealingSystem()
        self.system_state = SystemState.HEALTHY
        self.reliability_metrics = {
            'uptime_start': time.time(),
            'total_operations': 0,
            'failed_operations': 0,
            'recovered_failures': 0
        }
        
        self._setup_default_health_checks()
        self._setup_default_healing_strategies()
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup reliability logging."""
        self.logger = logging.getLogger('enterprise_reliability')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _setup_default_health_checks(self):
        """Setup default health checks."""
        
        # Photonic system health check
        def photonic_health_check():
            # Simulate photonic system health
            temp = random.uniform(280, 370)  # Temperature check
            efficiency = random.uniform(0.6, 0.95)  # Efficiency check
            
            if temp > 350:
                return False, f"Temperature too high: {temp:.1f}K", {'temperature': temp}
            elif efficiency < 0.7:
                return False, f"Efficiency too low: {efficiency:.2f}", {'efficiency': efficiency}
            else:
                return True, "Photonic system healthy", {'temperature': temp, 'efficiency': efficiency}
        
        photonic_check = HealthCheck(
            name="photonic_system",
            check_function=photonic_health_check,
            interval_seconds=10.0,
            failure_threshold=2
        )
        self.health_checker.add_health_check(photonic_check)
        
        # Memory health check
        def memory_health_check():
            # Simulate memory usage check
            memory_usage = random.uniform(0.3, 0.9)
            
            if memory_usage > 0.85:
                return False, f"High memory usage: {memory_usage:.1%}", {'memory_usage': memory_usage}
            else:
                return True, "Memory usage normal", {'memory_usage': memory_usage}
        
        memory_check = HealthCheck(
            name="memory_usage",
            check_function=memory_health_check,
            interval_seconds=15.0,
            failure_threshold=3
        )
        self.health_checker.add_health_check(memory_check)
    
    def _setup_default_healing_strategies(self):
        """Setup default self-healing strategies."""
        
        def heal_thermal_runaway(context):
            """Heal thermal runaway by reducing power."""
            self.logger.info("Attempting thermal runaway healing: reducing power")
            # Simulate power reduction
            time.sleep(0.1)  # Simulate healing time
            return random.choice([True, False])  # Random success for demo
        
        def heal_optical_loss(context):
            """Heal optical loss by adjusting coupling."""
            self.logger.info("Attempting optical loss healing: adjusting coupling")
            time.sleep(0.05)
            return random.choice([True, False])
        
        def heal_processing_timeout(context):
            """Heal processing timeout by restarting component."""
            self.logger.info("Attempting timeout healing: restarting component")
            time.sleep(0.2)
            return True  # Restart usually works
        
        self.self_healing.register_healing_strategy(FailureMode.THERMAL_RUNAWAY, heal_thermal_runaway)
        self.self_healing.register_healing_strategy(FailureMode.OPTICAL_LOSS, heal_optical_loss)
        self.self_healing.register_healing_strategy(FailureMode.PROCESSING_TIMEOUT, heal_processing_timeout)
    
    def add_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Add a circuit breaker."""
        if config is None:
            config = CircuitBreakerConfig()
        
        circuit_breaker = CircuitBreaker(name, config)
        self.circuit_breakers[name] = circuit_breaker
        return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self.circuit_breakers.get(name)
    
    @contextmanager
    def reliable_operation(self, operation_name: str, 
                          circuit_breaker_name: Optional[str] = None,
                          auto_heal_on_failure: bool = True):
        """Context manager for reliable operations."""
        start_time = time.time()
        self.reliability_metrics['total_operations'] += 1
        
        try:
            # Use circuit breaker if specified
            if circuit_breaker_name and circuit_breaker_name in self.circuit_breakers:
                circuit_breaker = self.circuit_breakers[circuit_breaker_name]
                if circuit_breaker.state == CircuitBreakerState.OPEN:
                    raise Exception(f"Circuit breaker {circuit_breaker_name} is OPEN")
            
            yield
            
            # Record success
            duration = time.time() - start_time
            self.logger.debug(f"Operation {operation_name} succeeded in {duration:.3f}s")
        
        except Exception as e:
            # Record failure
            self.reliability_metrics['failed_operations'] += 1
            duration = time.time() - start_time
            
            self.logger.warning(f"Operation {operation_name} failed in {duration:.3f}s: {str(e)}")
            
            # Attempt self-healing if enabled
            if auto_heal_on_failure:
                # Try to identify failure mode and heal
                failure_mode = self._identify_failure_mode(str(e))
                if failure_mode:
                    success, heal_message = self.self_healing.attempt_healing(
                        failure_mode, 
                        {'operation': operation_name, 'error': str(e), 'duration': duration}
                    )
                    
                    if success:
                        self.reliability_metrics['recovered_failures'] += 1
                        self.logger.info(f"Self-healing successful for {operation_name}: {heal_message}")
            
            # Update circuit breaker
            if circuit_breaker_name and circuit_breaker_name in self.circuit_breakers:
                self.circuit_breakers[circuit_breaker_name]._record_failure()
            
            raise e
    
    def _identify_failure_mode(self, error_message: str) -> Optional[FailureMode]:
        """Identify failure mode from error message."""
        error_lower = error_message.lower()
        
        if 'timeout' in error_lower:
            return FailureMode.PROCESSING_TIMEOUT
        elif 'temperature' in error_lower or 'thermal' in error_lower:
            return FailureMode.THERMAL_RUNAWAY
        elif 'optical' in error_lower or 'loss' in error_lower:
            return FailureMode.OPTICAL_LOSS
        elif 'memory' in error_lower:
            return FailureMode.MEMORY_EXHAUSTION
        elif 'power' in error_lower:
            return FailureMode.POWER_FLUCTUATION
        
        return None
    
    def start_reliability_monitoring(self):
        """Start all reliability monitoring."""
        self.health_checker.start_health_checking()
        self.logger.info("Enterprise reliability monitoring started")
    
    def stop_reliability_monitoring(self):
        """Stop all reliability monitoring."""
        self.health_checker.stop_health_checking()
        self.logger.info("Enterprise reliability monitoring stopped")
    
    def get_reliability_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive reliability dashboard."""
        uptime = time.time() - self.reliability_metrics['uptime_start']
        total_ops = self.reliability_metrics['total_operations']
        failed_ops = self.reliability_metrics['failed_operations']
        
        # Calculate reliability metrics
        availability = 1.0 - (failed_ops / max(total_ops, 1))
        mtbf = uptime / max(failed_ops, 1) if failed_ops > 0 else uptime
        recovery_rate = self.reliability_metrics['recovered_failures'] / max(failed_ops, 1) if failed_ops > 0 else 0
        
        # Get component statuses
        health_summary = self.health_checker.get_health_summary()
        
        circuit_breaker_status = {}
        for name, cb in self.circuit_breakers.items():
            circuit_breaker_status[name] = cb.get_metrics()
        
        healing_stats = self.self_healing.get_healing_statistics()
        
        return {
            'system_state': self.system_state.value,
            'uptime_hours': uptime / 3600,
            'reliability_metrics': {
                'availability': availability,
                'mtbf_hours': mtbf / 3600,
                'total_operations': total_ops,
                'failed_operations': failed_ops,
                'recovery_rate': recovery_rate
            },
            'health_summary': health_summary,
            'circuit_breakers': circuit_breaker_status,
            'self_healing': healing_stats,
            'timestamp': time.time()
        }


def demonstrate_enterprise_reliability():
    """Demonstrate enterprise reliability framework."""
    print("🛡️ Demonstrating Enterprise Reliability Framework")
    print("=" * 65)
    
    # Create reliability framework
    reliability = EnterpriseReliabilityFramework()
    
    # Add circuit breakers
    processing_cb = reliability.add_circuit_breaker("processing", CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=10.0
    ))
    
    optical_cb = reliability.add_circuit_breaker("optical", CircuitBreakerConfig(
        failure_threshold=2,
        recovery_timeout=5.0
    ))
    
    # Start monitoring
    print("\n1. Starting reliability monitoring...")
    reliability.start_reliability_monitoring()
    
    # Simulate operations with various failure scenarios
    print("\n2. Simulating system operations...")
    
    def simulate_processing_operation():
        """Simulate a processing operation that might fail."""
        if random.random() < 0.2:  # 20% failure rate
            raise Exception("Processing timeout occurred")
        time.sleep(0.01)  # Simulate work
        return "Processing complete"
    
    def simulate_optical_operation():
        """Simulate an optical operation that might fail."""
        if random.random() < 0.15:  # 15% failure rate
            raise Exception("Optical loss detected")
        time.sleep(0.005)  # Simulate work
        return "Optical operation complete"
    
    # Run operations
    for i in range(50):
        # Processing operation with reliability
        try:
            with reliability.reliable_operation("processing", "processing", auto_heal_on_failure=True):
                result = processing_cb.call(simulate_processing_operation)
        except Exception as e:
            pass  # Expected failures for demonstration
        
        # Optical operation with reliability
        try:
            with reliability.reliable_operation("optical", "optical", auto_heal_on_failure=True):
                result = optical_cb.call(simulate_optical_operation)
        except Exception as e:
            pass  # Expected failures for demonstration
        
        time.sleep(0.02)  # Brief pause between operations
    
    # Wait for health checks to run
    time.sleep(2.0)
    
    # Get reliability dashboard
    print("\n3. Reliability Dashboard:")
    dashboard = reliability.get_reliability_dashboard()
    
    # System overview
    print(f"   System State: {dashboard['system_state']}")
    print(f"   Uptime: {dashboard['uptime_hours']:.2f} hours")
    
    # Reliability metrics
    metrics = dashboard['reliability_metrics']
    print(f"   Availability: {metrics['availability']:.3f} ({metrics['availability']*100:.1f}%)")
    print(f"   MTBF: {metrics['mtbf_hours']:.2f} hours")
    print(f"   Total Operations: {metrics['total_operations']}")
    print(f"   Failed Operations: {metrics['failed_operations']}")
    print(f"   Recovery Rate: {metrics['recovery_rate']:.2f}")
    
    # Circuit breaker status
    print("\n4. Circuit Breaker Status:")
    for cb_name, cb_metrics in dashboard['circuit_breakers'].items():
        print(f"   {cb_name}: {cb_metrics['state']} (failures: {cb_metrics['failure_count']})")
        print(f"      Total calls: {cb_metrics['total_calls']}, Failure rate: {cb_metrics['failure_rate']:.3f}")
    
    # Health check status
    print("\n5. Component Health:")
    health = dashboard['health_summary']
    print(f"   System Health: {health['system_health']}")
    print(f"   Active Health Checks: {health['total_health_checks']}")
    
    for component, component_health in health['component_health'].items():
        print(f"   {component}: {component_health['status']} (failures: {component_health['current_failures']})")
    
    # Self-healing statistics
    print("\n6. Self-Healing Statistics:")
    healing = dashboard['self_healing']
    if healing['total_attempts'] > 0:
        print(f"   Total Healing Attempts: {healing['total_attempts']}")
        print(f"   Successful Healings: {healing['successful_attempts']}")
        print(f"   Success Rate: {healing['success_rate']:.2f}")
        
        for mode, stats in healing['failure_mode_stats'].items():
            print(f"   {mode}: {stats['successes']}/{stats['attempts']} successful")
    else:
        print("   No healing attempts recorded")
    
    # Stop monitoring
    print("\n7. Stopping reliability monitoring...")
    reliability.stop_reliability_monitoring()
    
    return dashboard


if __name__ == "__main__":
    demonstrate_enterprise_reliability()