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
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    security_level: str = "standard"  # standard, high, critical
    risk_score: float = 0.0
    
    def is_expired(self, timeout: int = 3600) -> bool:
        """Check if session is expired."""
        return time.time() - self.last_activity > timeout
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = time.time()
    
    def calculate_risk_score(self) -> float:
        """Calculate session risk score based on various factors."""
        risk_factors = []
        
        # Session age factor
        session_age = time.time() - self.created_at
        age_risk = min(session_age / 86400, 1.0)  # Normalize to 24 hours
        risk_factors.append(age_risk * 0.2)
        
        # Activity pattern factor
        inactivity = time.time() - self.last_activity
        inactivity_risk = min(inactivity / 3600, 1.0)  # Normalize to 1 hour
        risk_factors.append(inactivity_risk * 0.3)
        
        # Permission scope factor
        permission_risk = len(self.permissions) / 10.0  # Assume max 10 permissions
        risk_factors.append(min(permission_risk, 1.0) * 0.2)
        
        # Security level factor
        security_level_risk = {"critical": 0.1, "high": 0.3, "standard": 0.5}
        risk_factors.append(security_level_risk.get(self.security_level, 0.5) * 0.3)
        
        self.risk_score = sum(risk_factors)
        return self.risk_score


class AdvancedThreatDetectionSystem:
    """
    Advanced threat detection system with behavioral analysis and ML-based anomaly detection.
    
    Implements zero-trust security principles with continuous monitoring,
    behavioral analysis, and adaptive threat response.
    """
    
    def __init__(self, enable_ml_detection: bool = True):
        self.enable_ml_detection = enable_ml_detection
        self.threat_patterns = {}
        self.behavioral_baselines = {}
        self.threat_history = []
        self.active_sessions = {}
        self.security_events = []
        
        # Machine learning components
        if enable_ml_detection:
            self._initialize_ml_detection()
        
        # Threat scoring
        self.threat_thresholds = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8,
            "critical": 0.95
        }
        
        # Rate limiting
        self.rate_limiters = {}
        
        self._logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
    
    def _initialize_ml_detection(self):
        """Initialize ML-based threat detection components."""
        try:
            import numpy as np
            from sklearn.ensemble import IsolationForest
            from sklearn.cluster import DBSCAN
            from collections import deque
            
            # Anomaly detection for behavioral analysis
            self.anomaly_detector = IsolationForest(
                contamination=0.05,  # 5% expected anomalies
                random_state=42
            )
            
            # Clustering for pattern recognition
            self.pattern_analyzer = DBSCAN(
                eps=0.3,
                min_samples=5
            )
            
            # Feature history for training
            self.behavior_features = deque(maxlen=10000)
            self.ml_initialized = True
            
        except ImportError:
            self._logger.warning("scikit-learn not available, disabling ML threat detection")
            self.ml_initialized = False
    
    def analyze_request(
        self,
        request_data: Dict[str, Any],
        session: SecureSimulationSession
    ) -> Dict[str, Any]:
        """
        Analyze incoming request for threats and anomalies.
        
        Args:
            request_data: Request data to analyze
            session: User session
            
        Returns:
            Threat analysis results
        """
        analysis_start = time.perf_counter()
        
        # Extract features for analysis
        features = self._extract_security_features(request_data, session)
        
        # Perform multi-layered threat analysis
        threat_score = 0.0
        threat_indicators = []
        
        # 1. Pattern-based detection
        pattern_threats = self._detect_pattern_threats(features)
        threat_score += pattern_threats["score"]
        threat_indicators.extend(pattern_threats["indicators"])
        
        # 2. Behavioral analysis
        behavioral_threats = self._analyze_behavioral_anomalies(features, session)
        threat_score += behavioral_threats["score"]
        threat_indicators.extend(behavioral_threats["indicators"])
        
        # 3. Rate limiting check
        rate_limit_threats = self._check_rate_limits(session)
        threat_score += rate_limit_threats["score"]
        threat_indicators.extend(rate_limit_threats["indicators"])
        
        # 4. ML-based anomaly detection
        if self.ml_initialized:
            ml_threats = self._ml_anomaly_detection(features)
            threat_score += ml_threats["score"]
            threat_indicators.extend(ml_threats["indicators"])
        
        # 5. Input validation threats
        validation_threats = self._detect_input_validation_threats(request_data)
        threat_score += validation_threats["score"]
        threat_indicators.extend(validation_threats["indicators"])
        
        # Normalize threat score
        threat_score = min(threat_score, 1.0)
        
        # Determine threat level
        threat_level = self._determine_threat_level(threat_score)
        
        analysis_time = time.perf_counter() - analysis_start
        
        # Record security event
        security_event = {
            "timestamp": time.time(),
            "session_id": session.session_id,
            "threat_score": threat_score,
            "threat_level": threat_level,
            "indicators": threat_indicators,
            "analysis_time": analysis_time,
            "request_features": features
        }
        
        with self._lock:
            self.security_events.append(security_event)
            # Keep events manageable
            if len(self.security_events) > 10000:
                self.security_events = self.security_events[-5000:]
        
        return {
            "threat_score": threat_score,
            "threat_level": threat_level,
            "indicators": threat_indicators,
            "analysis_time": analysis_time,
            "recommended_action": self._recommend_action(threat_level),
            "allow_request": threat_score < self.threat_thresholds["high"]
        }
    
    def _extract_security_features(
        self,
        request_data: Dict[str, Any],
        session: SecureSimulationSession
    ) -> Dict[str, Any]:
        """Extract security-relevant features from request."""
        features = {
            # Request characteristics
            "request_size": len(str(request_data)),
            "param_count": len(request_data),
            "has_file_upload": any("file" in str(k).lower() for k in request_data.keys()),
            "has_script_content": any("<script" in str(v) for v in request_data.values() if isinstance(v, str)),
            "has_sql_patterns": any(
                sql_keyword in str(v).lower() 
                for v in request_data.values() 
                if isinstance(v, str)
                for sql_keyword in ["select", "insert", "update", "delete", "union", "drop"]
            ),
            
            # Session characteristics
            "session_age": time.time() - session.created_at,
            "session_activity": time.time() - session.last_activity,
            "permission_count": len(session.permissions),
            "session_risk_score": session.risk_score,
            
            # Timing characteristics
            "hour_of_day": int((time.time() % 86400) / 3600),
            "day_of_week": int(time.time() / 86400) % 7,
            
            # Pattern characteristics
            "numeric_params": sum(1 for v in request_data.values() if isinstance(v, (int, float))),
            "string_params": sum(1 for v in request_data.values() if isinstance(v, str)),
            "list_params": sum(1 for v in request_data.values() if isinstance(v, list)),
        }
        
        return features
    
    def _detect_pattern_threats(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Detect threats based on known attack patterns."""
        threat_score = 0.0
        indicators = []
        
        # SQL injection patterns
        if features.get("has_sql_patterns", False):
            threat_score += 0.7
            indicators.append("SQL injection patterns detected")
        
        # XSS patterns
        if features.get("has_script_content", False):
            threat_score += 0.6
            indicators.append("Script content detected (potential XSS)")
        
        # Unusual request size
        if features.get("request_size", 0) > 1024 * 1024:  # 1MB
            threat_score += 0.4
            indicators.append("Unusually large request size")
        
        # Excessive parameters
        if features.get("param_count", 0) > 100:
            threat_score += 0.3
            indicators.append("Excessive parameter count")
        
        # Suspicious timing
        hour = features.get("hour_of_day", 12)
        if hour < 6 or hour > 22:  # Outside business hours
            threat_score += 0.1
            indicators.append("Request outside normal business hours")
        
        return {
            "score": min(threat_score, 1.0),
            "indicators": indicators
        }
    
    def _analyze_behavioral_anomalies(
        self,
        features: Dict[str, Any],
        session: SecureSimulationSession
    ) -> Dict[str, Any]:
        """Analyze behavioral anomalies compared to baseline."""
        threat_score = 0.0
        indicators = []
        
        user_id = session.user_id or "anonymous"
        
        # Check if we have behavioral baseline for this user
        if user_id not in self.behavioral_baselines:
            self.behavioral_baselines[user_id] = {
                "request_sizes": [],
                "param_counts": [],
                "activity_patterns": [],
                "typical_hours": set()
            }
        
        baseline = self.behavioral_baselines[user_id]
        
        # Analyze request size anomaly
        request_size = features.get("request_size", 0)
        if baseline["request_sizes"]:
            avg_size = sum(baseline["request_sizes"]) / len(baseline["request_sizes"])
            if request_size > avg_size * 5:  # 5x larger than average
                threat_score += 0.3
                indicators.append("Request size significantly above user baseline")
        
        # Analyze parameter count anomaly
        param_count = features.get("param_count", 0)
        if baseline["param_counts"]:
            avg_params = sum(baseline["param_counts"]) / len(baseline["param_counts"])
            if param_count > avg_params * 3:  # 3x more parameters
                threat_score += 0.2
                indicators.append("Parameter count above user baseline")
        
        # Analyze activity pattern anomaly
        current_hour = features.get("hour_of_day", 12)
        if baseline["typical_hours"] and current_hour not in baseline["typical_hours"]:
            threat_score += 0.2
            indicators.append("Activity outside typical hours for user")
        
        # Update baseline (with limits to prevent memory issues)
        baseline["request_sizes"].append(request_size)
        if len(baseline["request_sizes"]) > 1000:
            baseline["request_sizes"] = baseline["request_sizes"][-500:]
        
        baseline["param_counts"].append(param_count)
        if len(baseline["param_counts"]) > 1000:
            baseline["param_counts"] = baseline["param_counts"][-500:]
        
        baseline["typical_hours"].add(current_hour)
        
        return {
            "score": min(threat_score, 1.0),
            "indicators": indicators
        }
    
    def _check_rate_limits(self, session: SecureSimulationSession) -> Dict[str, Any]:
        """Check rate limiting violations."""
        threat_score = 0.0
        indicators = []
        
        user_id = session.user_id or session.session_id
        current_time = time.time()
        
        # Initialize rate limiter for user
        if user_id not in self.rate_limiters:
            self.rate_limiters[user_id] = {
                "requests": [],
                "last_reset": current_time
            }
        
        rate_limiter = self.rate_limiters[user_id]
        
        # Clean old requests (sliding window of 1 hour)
        rate_limiter["requests"] = [
            req_time for req_time in rate_limiter["requests"]
            if current_time - req_time < 3600
        ]
        
        # Add current request
        rate_limiter["requests"].append(current_time)
        
        # Check rate limits
        requests_per_hour = len(rate_limiter["requests"])
        
        if requests_per_hour > 1000:  # High rate
            threat_score += 0.8
            indicators.append(f"Excessive request rate: {requests_per_hour}/hour")
        elif requests_per_hour > 500:  # Medium rate
            threat_score += 0.4
            indicators.append(f"Elevated request rate: {requests_per_hour}/hour")
        
        # Check burst rate (requests per minute)
        recent_requests = [
            req_time for req_time in rate_limiter["requests"]
            if current_time - req_time < 60
        ]
        
        if len(recent_requests) > 100:  # High burst
            threat_score += 0.6
            indicators.append(f"High burst rate: {len(recent_requests)}/minute")
        
        return {
            "score": min(threat_score, 1.0),
            "indicators": indicators
        }
    
    def _ml_anomaly_detection(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """ML-based anomaly detection."""
        threat_score = 0.0
        indicators = []
        
        if not self.ml_initialized:
            return {"score": 0.0, "indicators": []}
        
        try:
            import numpy as np
            
            # Convert features to numerical array
            feature_vector = [
                features.get("request_size", 0),
                features.get("param_count", 0),
                features.get("session_age", 0),
                features.get("session_activity", 0),
                features.get("permission_count", 0),
                features.get("hour_of_day", 12),
                features.get("numeric_params", 0),
                features.get("string_params", 0),
                int(features.get("has_sql_patterns", False)),
                int(features.get("has_script_content", False))
            ]
            
            # Add to feature history
            self.behavior_features.append(feature_vector)
            
            # Retrain model periodically
            if len(self.behavior_features) >= 100 and len(self.behavior_features) % 100 == 0:
                self._retrain_anomaly_detector()
            
            # Predict anomaly if we have enough data
            if len(self.behavior_features) >= 50:
                features_array = np.array([feature_vector])
                prediction = self.anomaly_detector.predict(features_array)
                anomaly_score = self.anomaly_detector.decision_function(features_array)[0]
                
                if prediction[0] == -1:  # Anomaly detected
                    threat_score = min(abs(anomaly_score) / 2.0, 0.7)  # Scale to 0-0.7
                    indicators.append(f"ML anomaly detected (score: {anomaly_score:.3f})")
        
        except Exception as e:
            self._logger.warning(f"ML anomaly detection failed: {e}")
        
        return {
            "score": threat_score,
            "indicators": indicators
        }
    
    def _detect_input_validation_threats(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect input validation threats."""
        threat_score = 0.0
        indicators = []
        
        for key, value in request_data.items():
            if isinstance(value, str):
                # Check for common injection patterns
                suspicious_patterns = [
                    r"<script[^>]*>",  # Script tags
                    r"javascript:",     # JavaScript protocol
                    r"vbscript:",      # VBScript protocol
                    r"data:",          # Data URI
                    r"file://",        # File protocol
                    r"\.\./",          # Directory traversal
                    r"cmd\.exe",       # Command execution
                    r"powershell",     # PowerShell
                    r"/etc/passwd",    # System file access
                    r"SELECT.*FROM",   # SQL injection
                    r"UNION.*SELECT",  # SQL union injection
                ]
                
                for pattern in suspicious_patterns:
                    import re
                    if re.search(pattern, value, re.IGNORECASE):
                        threat_score += 0.5
                        indicators.append(f"Suspicious pattern in {key}: {pattern}")
                        break
                
                # Check for extremely long strings (potential buffer overflow)
                if len(value) > 10000:
                    threat_score += 0.3
                    indicators.append(f"Extremely long input in {key}")
                
                # Check for binary content in text fields
                if any(ord(c) < 32 and c not in '\t\n\r' for c in value):
                    threat_score += 0.4
                    indicators.append(f"Binary content in text field {key}")
        
        return {
            "score": min(threat_score, 1.0),
            "indicators": indicators
        }
    
    def _retrain_anomaly_detector(self):
        """Retrain the anomaly detection model."""
        if not self.ml_initialized or len(self.behavior_features) < 50:
            return
        
        try:
            import numpy as np
            
            features_array = np.array(list(self.behavior_features))
            self.anomaly_detector.fit(features_array)
            self._logger.debug("Retrained anomaly detection model")
            
        except Exception as e:
            self._logger.warning(f"Failed to retrain anomaly detector: {e}")
    
    def _determine_threat_level(self, threat_score: float) -> str:
        """Determine threat level based on score."""
        if threat_score >= self.threat_thresholds["critical"]:
            return "critical"
        elif threat_score >= self.threat_thresholds["high"]:
            return "high"
        elif threat_score >= self.threat_thresholds["medium"]:
            return "medium"
        elif threat_score >= self.threat_thresholds["low"]:
            return "low"
        else:
            return "none"
    
    def _recommend_action(self, threat_level: str) -> str:
        """Recommend action based on threat level."""
        recommendations = {
            "none": "allow",
            "low": "allow_with_monitoring",
            "medium": "require_additional_auth",
            "high": "block_with_review",
            "critical": "block_immediately"
        }
        return recommendations.get(threat_level, "block_immediately")
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get comprehensive security metrics."""
        with self._lock:
            total_events = len(self.security_events)
            
            if total_events == 0:
                return {
                    "total_events": 0,
                    "threat_distribution": {},
                    "average_threat_score": 0.0,
                    "blocked_requests": 0,
                    "active_sessions": len(self.active_sessions)
                }
            
            # Calculate threat distribution
            threat_levels = [event["threat_level"] for event in self.security_events]
            threat_distribution = {}
            for level in ["none", "low", "medium", "high", "critical"]:
                threat_distribution[level] = threat_levels.count(level)
            
            # Calculate average threat score
            avg_threat_score = sum(event["threat_score"] for event in self.security_events) / total_events
            
            # Count blocked requests
            blocked_requests = sum(
                1 for event in self.security_events
                if event["threat_score"] >= self.threat_thresholds["high"]
            )
            
            # Recent activity (last hour)
            recent_events = [
                event for event in self.security_events
                if time.time() - event["timestamp"] < 3600
            ]
            
            return {
                "total_events": total_events,
                "recent_events": len(recent_events),
                "threat_distribution": threat_distribution,
                "average_threat_score": avg_threat_score,
                "blocked_requests": blocked_requests,
                "block_rate": blocked_requests / total_events,
                "active_sessions": len(self.active_sessions),
                "ml_detection_enabled": self.ml_initialized,
                "behavioral_profiles": len(self.behavioral_baselines),
                "rate_limited_users": len(self.rate_limiters)
            }


class ZeroTrustSecurityManager:
    """
    Zero-trust security manager implementing comprehensive security controls.
    
    Enforces zero-trust principles: never trust, always verify.
    Implements continuous authentication, authorization, and monitoring.
    """
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.threat_detector = AdvancedThreatDetectionSystem()
        self.active_sessions = {}
        self.security_policies = {}
        self.audit_log = []
        
        # Initialize default security policies
        self._initialize_security_policies()
        
        self._logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
    
    def _initialize_security_policies(self):
        """Initialize default security policies."""
        self.security_policies = {
            "require_session_validation": True,
            "require_input_sanitization": True,
            "enable_rate_limiting": True,
            "enable_threat_detection": True,
            "max_session_duration": 3600,  # 1 hour
            "require_reauth_on_privilege_escalation": True,
            "enable_behavioral_monitoring": True,
            "audit_all_operations": True
        }
    
    def create_secure_session(
        self,
        user_id: Optional[str] = None,
        permissions: List[str] = None,
        security_level: str = "standard",
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> SecureSimulationSession:
        """Create a new secure session with comprehensive validation."""
        session = SecureSimulationSession(
            user_id=user_id,
            permissions=permissions or [],
            security_level=security_level,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        # Calculate initial risk score
        session.calculate_risk_score()
        
        # Store session
        with self._lock:
            self.active_sessions[session.session_id] = session
            self.threat_detector.active_sessions[session.session_id] = session
        
        # Audit log
        self._audit_log("session_created", {
            "session_id": session.session_id,
            "user_id": user_id,
            "security_level": security_level,
            "permissions": permissions
        })
        
        self._logger.info(f"Created secure session: {session.session_id}")
        return session
    
    def validate_request(
        self,
        session_id: str,
        request_data: Dict[str, Any],
        required_permissions: List[str] = None
    ) -> Dict[str, Any]:
        """
        Validate request with comprehensive security checks.
        
        Implements zero-trust validation: every request is untrusted
        until proven otherwise through multiple security layers.
        """
        validation_start = time.perf_counter()
        
        # 1. Session validation
        session_validation = self._validate_session(session_id)
        if not session_validation["valid"]:
            return {
                "valid": False,
                "reason": session_validation["reason"],
                "threat_level": "high",
                "recommended_action": "block_immediately"
            }
        
        session = session_validation["session"]
        
        # 2. Permission validation
        if required_permissions:
            permission_validation = self._validate_permissions(session, required_permissions)
            if not permission_validation["valid"]:
                return {
                    "valid": False,
                    "reason": permission_validation["reason"],
                    "threat_level": "medium",
                    "recommended_action": "require_additional_auth"
                }
        
        # 3. Input sanitization and validation
        input_validation = self._validate_and_sanitize_input(request_data)
        if not input_validation["valid"]:
            return {
                "valid": False,
                "reason": input_validation["reason"],
                "threat_level": "high",
                "recommended_action": "block_with_review",
                "sanitized_input": input_validation.get("sanitized_data")
            }
        
        # 4. Threat detection analysis
        threat_analysis = self.threat_detector.analyze_request(request_data, session)
        
        # 5. Update session activity
        session.update_activity()
        
        validation_time = time.perf_counter() - validation_start
        
        # 6. Make final decision
        is_valid = (
            session_validation["valid"] and
            input_validation["valid"] and
            threat_analysis["allow_request"]
        )
        
        # 7. Audit log
        self._audit_log("request_validated", {
            "session_id": session_id,
            "valid": is_valid,
            "threat_score": threat_analysis["threat_score"],
            "threat_level": threat_analysis["threat_level"],
            "validation_time": validation_time
        })
        
        return {
            "valid": is_valid,
            "session": session,
            "threat_analysis": threat_analysis,
            "validation_time": validation_time,
            "sanitized_input": input_validation.get("sanitized_data", request_data)
        }
    
    def _validate_session(self, session_id: str) -> Dict[str, Any]:
        """Validate session exists and is still valid."""
        with self._lock:
            if session_id not in self.active_sessions:
                return {
                    "valid": False,
                    "reason": "Session not found",
                    "session": None
                }
            
            session = self.active_sessions[session_id]
        
        # Check if session is expired
        if session.is_expired(self.config.session_timeout):
            with self._lock:
                del self.active_sessions[session_id]
            return {
                "valid": False,
                "reason": "Session expired",
                "session": None
            }
        
        # Check if session is active
        if not session.is_active:
            return {
                "valid": False,
                "reason": "Session deactivated",
                "session": None
            }
        
        return {
            "valid": True,
            "reason": "Session valid",
            "session": session
        }
    
    def _validate_permissions(
        self,
        session: SecureSimulationSession,
        required_permissions: List[str]
    ) -> Dict[str, Any]:
        """Validate session has required permissions."""
        missing_permissions = [
            perm for perm in required_permissions
            if perm not in session.permissions
        ]
        
        if missing_permissions:
            return {
                "valid": False,
                "reason": f"Missing permissions: {missing_permissions}",
                "missing_permissions": missing_permissions
            }
        
        return {
            "valid": True,
            "reason": "Permissions validated"
        }
    
    def _validate_and_sanitize_input(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize input data."""
        sanitized_data = {}
        validation_errors = []
        
        for key, value in request_data.items():
            try:
                # Size validation
                if isinstance(value, str) and len(value) > self.config.max_file_size:
                    validation_errors.append(f"Input {key} exceeds maximum size")
                    continue
                
                # Type validation and sanitization
                if isinstance(value, str):
                    sanitized_value = self._sanitize_string(value)
                    sanitized_data[key] = sanitized_value
                elif isinstance(value, (int, float)):
                    # Validate numeric ranges
                    if abs(value) > 1e12:  # Prevent numerical overflow
                        validation_errors.append(f"Numeric value {key} out of safe range")
                        continue
                    sanitized_data[key] = value
                elif isinstance(value, (list, dict)):
                    # Recursive validation for complex types
                    if isinstance(value, dict):
                        nested_validation = self._validate_and_sanitize_input(value)
                        if not nested_validation["valid"]:
                            validation_errors.extend(nested_validation["errors"])
                            continue
                        sanitized_data[key] = nested_validation["sanitized_data"]
                    else:
                        sanitized_data[key] = value  # Lists handled separately if needed
                else:
                    sanitized_data[key] = value
                    
            except Exception as e:
                validation_errors.append(f"Validation error for {key}: {str(e)}")
        
        return {
            "valid": len(validation_errors) == 0,
            "sanitized_data": sanitized_data,
            "errors": validation_errors,
            "reason": "; ".join(validation_errors) if validation_errors else "Input validated"
        }
    
    def _sanitize_string(self, value: str) -> str:
        """Sanitize string input to prevent injection attacks."""
        import html
        import re
        
        # HTML entity encoding
        sanitized = html.escape(value)
        
        # Remove or escape potentially dangerous patterns
        dangerous_patterns = [
            (r'<script[^>]*>.*?</script>', ''),  # Remove script tags
            (r'javascript:', 'removed:'),         # Remove javascript protocol
            (r'vbscript:', 'removed:'),          # Remove vbscript protocol
            (r'data:', 'removed:'),              # Remove data URI
            (r'file://', 'removed://'),          # Remove file protocol
        ]
        
        for pattern, replacement in dangerous_patterns:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def _audit_log(self, event_type: str, event_data: Dict[str, Any]):
        """Add entry to audit log."""
        if not self.config.enable_audit_logging:
            return
        
        audit_entry = {
            "timestamp": time.time(),
            "event_type": event_type,
            "event_data": event_data,
            "correlation_id": event_data.get("session_id")
        }
        
        with self._lock:
            self.audit_log.append(audit_entry)
            # Keep audit log manageable
            if len(self.audit_log) > 100000:
                self.audit_log = self.audit_log[-50000:]
        
        self._logger.info(f"Security audit: {event_type} - {event_data}")
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard data."""
        with self._lock:
            active_session_count = len(self.active_sessions)
            
            # Calculate session risk distribution
            risk_distribution = {"low": 0, "medium": 0, "high": 0}
            for session in self.active_sessions.values():
                risk_score = session.calculate_risk_score()
                if risk_score < 0.3:
                    risk_distribution["low"] += 1
                elif risk_score < 0.7:
                    risk_distribution["medium"] += 1
                else:
                    risk_distribution["high"] += 1
            
            # Recent audit events
            recent_audit_events = [
                event for event in self.audit_log
                if time.time() - event["timestamp"] < 3600  # Last hour
            ]
        
        # Get threat detection metrics
        threat_metrics = self.threat_detector.get_security_metrics()
        
        return {
            "active_sessions": active_session_count,
            "session_risk_distribution": risk_distribution,
            "recent_audit_events": len(recent_audit_events),
            "threat_detection_metrics": threat_metrics,
            "security_policies": self.security_policies.copy(),
            "system_health": {
                "threat_detection_enabled": True,
                "behavioral_monitoring_enabled": True,
                "rate_limiting_enabled": True,
                "audit_logging_enabled": self.config.enable_audit_logging
            }
        }
    
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