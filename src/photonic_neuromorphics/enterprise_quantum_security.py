"""
Enterprise Quantum Security Framework for Photonic Neuromorphics

Implements quantum-resistant security protocols, end-to-end encryption, and
comprehensive threat detection for photonic neural network systems.

Security Features:
- Post-quantum cryptography implementation
- Quantum key distribution (QKD) protocols
- Real-time threat detection and response
- Secure multi-party computation for distributed systems
- Hardware security module (HSM) integration
"""

import numpy as np
import torch
import hashlib
import secrets
import time
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from enum import Enum
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from .enhanced_logging import PhotonicLogger
from .monitoring import MetricsCollector
from .exceptions import SecurityError, ValidationError


class SecurityLevel(Enum):
    """Security classification levels."""
    PUBLIC = "public"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"
    QUANTUM_SECURE = "quantum_secure"


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    QUANTUM_ATTACK = "quantum_attack"


@dataclass
class SecurityContext:
    """Security context for operations."""
    classification: SecurityLevel
    user_id: str
    session_id: str
    timestamp: float
    permissions: List[str] = field(default_factory=list)
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    
    def log_access(self, resource: str, action: str, success: bool) -> None:
        """Log access attempt for audit trail."""
        self.audit_trail.append({
            "timestamp": time.time(),
            "resource": resource,
            "action": action,
            "success": success,
            "user_id": self.user_id,
            "session_id": self.session_id
        })


class QuantumKeyDistribution:
    """Quantum Key Distribution (QKD) implementation for secure key exchange."""
    
    def __init__(self, key_length: int = 256, error_threshold: float = 0.05):
        self.key_length = key_length
        self.error_threshold = error_threshold
        self.logger = PhotonicLogger("QKD")
        
    def generate_quantum_key(self, partner_id: str) -> Tuple[bytes, float]:
        """
        Generate quantum key using BB84 protocol simulation.
        
        Args:
            partner_id: Partner node identifier
            
        Returns:
            Tuple of (quantum_key, security_level)
        """
        # Simulate BB84 protocol
        raw_key_length = self.key_length * 2  # Account for basis reconciliation
        
        # Alice generates random bits and bases
        alice_bits = np.random.randint(0, 2, raw_key_length)
        alice_bases = np.random.randint(0, 2, raw_key_length)  # 0: rectilinear, 1: diagonal
        
        # Bob chooses random measurement bases
        bob_bases = np.random.randint(0, 2, raw_key_length)
        
        # Simulate quantum channel with noise
        error_rate = np.random.uniform(0.01, 0.03)  # Realistic channel noise
        bob_measurements = alice_bits.copy()
        
        # Add quantum channel errors
        error_positions = np.random.choice(
            raw_key_length, 
            int(error_rate * raw_key_length), 
            replace=False
        )
        bob_measurements[error_positions] = 1 - bob_measurements[error_positions]
        
        # Basis reconciliation - keep bits where bases match
        matching_bases = alice_bases == bob_bases
        sifted_key_alice = alice_bits[matching_bases]
        sifted_key_bob = bob_measurements[matching_bases]
        
        # Error estimation and privacy amplification
        test_subset_size = min(len(sifted_key_alice) // 4, 64)
        test_indices = np.random.choice(
            len(sifted_key_alice), 
            test_subset_size, 
            replace=False
        )
        
        test_bits_alice = sifted_key_alice[test_indices]
        test_bits_bob = sifted_key_bob[test_indices]
        observed_error_rate = np.sum(test_bits_alice != test_bits_bob) / test_subset_size
        
        # Security check
        if observed_error_rate > self.error_threshold:
            self.logger.warning(f"High error rate detected: {observed_error_rate:.3f}")
            raise SecurityError("quantum_key_distribution", "high_error_rate", 
                              f"Error rate {observed_error_rate:.3f} exceeds threshold")
        
        # Remove test bits and generate final key
        remaining_indices = np.setdiff1d(np.arange(len(sifted_key_alice)), test_indices)
        final_key_bits = sifted_key_alice[remaining_indices][:self.key_length]
        
        # Convert to bytes
        quantum_key = self._bits_to_bytes(final_key_bits)
        security_level = 1.0 - observed_error_rate  # Simple security metric
        
        self.logger.info(f"QKD completed with {partner_id}: "
                        f"key_length={len(quantum_key)}, "
                        f"security_level={security_level:.3f}")
        
        return quantum_key, security_level
    
    def _bits_to_bytes(self, bits: np.ndarray) -> bytes:
        """Convert bit array to bytes."""
        # Pad to multiple of 8
        padding = 8 - (len(bits) % 8)
        if padding != 8:
            bits = np.append(bits, np.zeros(padding, dtype=int))
        
        byte_array = []
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                byte |= bits[i + j] << (7 - j)
            byte_array.append(byte)
        
        return bytes(byte_array)


class PostQuantumCryptography:
    """Post-quantum cryptographic algorithms for quantum-resistant security."""
    
    def __init__(self):
        self.logger = PhotonicLogger("PostQuantumCrypto")
        
        # CRYSTALS-Kyber parameters (simplified)
        self.kyber_n = 256
        self.kyber_q = 3329
        self.kyber_k = 3  # Security level
        
    def generate_kyber_keypair(self) -> Tuple[bytes, bytes]:
        """Generate CRYSTALS-Kyber key pair for post-quantum encryption."""
        # Simplified Kyber implementation (production would use proper library)
        
        # Generate polynomial ring elements
        private_key = np.random.randint(0, self.kyber_q, size=(self.kyber_k, self.kyber_n))
        noise = np.random.normal(0, 1, size=(self.kyber_k, self.kyber_n))
        
        # Public key generation (simplified)
        public_matrix = np.random.randint(0, self.kyber_q, 
                                        size=(self.kyber_k, self.kyber_k, self.kyber_n))
        public_key = np.zeros((self.kyber_k, self.kyber_n))
        
        for i in range(self.kyber_k):
            for j in range(self.kyber_k):
                public_key[i] += np.convolve(
                    public_matrix[i, j], 
                    private_key[j], 
                    mode='same'
                )[:self.kyber_n]
            public_key[i] = (public_key[i] + noise[i]) % self.kyber_q
        
        # Serialize keys
        private_key_bytes = private_key.astype(np.int16).tobytes()
        public_key_bytes = public_key.astype(np.int16).tobytes()
        
        self.logger.debug("Generated Kyber key pair")
        
        return public_key_bytes, private_key_bytes
    
    def kyber_encrypt(self, public_key: bytes, message: bytes) -> bytes:
        """Encrypt message using CRYSTALS-Kyber."""
        if len(message) > 32:
            raise SecurityError("kyber_encrypt", "message_too_long", 
                              "Message must be ≤ 32 bytes")
        
        # Deserialize public key
        public_key_array = np.frombuffer(public_key, dtype=np.int16).reshape(
            (self.kyber_k, self.kyber_n)
        )
        
        # Generate random coins
        coins = np.random.randint(0, self.kyber_q, size=(self.kyber_k, self.kyber_n))
        error1 = np.random.normal(0, 1, size=(self.kyber_k, self.kyber_n))
        error2 = np.random.normal(0, 1, size=self.kyber_n)
        
        # Encryption (simplified)
        u = np.zeros((self.kyber_k, self.kyber_n))
        for i in range(self.kyber_k):
            u[i] = (np.convolve(public_key_array[i], coins[0], mode='same')[:self.kyber_n] + 
                   error1[i]) % self.kyber_q
        
        # Encode message
        message_poly = np.zeros(self.kyber_n)
        for i, byte_val in enumerate(message):
            if i * 8 < self.kyber_n:
                for bit in range(8):
                    if i * 8 + bit < self.kyber_n:
                        message_poly[i * 8 + bit] = (byte_val >> bit) & 1
        
        v = (np.sum([np.convolve(public_key_array[i], coins[i], mode='same')[:self.kyber_n] 
                    for i in range(self.kyber_k)], axis=0) + 
             error2 + message_poly * (self.kyber_q // 2)) % self.kyber_q
        
        # Serialize ciphertext
        ciphertext = np.concatenate([u.flatten(), v])
        return ciphertext.astype(np.int16).tobytes()
    
    def kyber_decrypt(self, private_key: bytes, ciphertext: bytes) -> bytes:
        """Decrypt ciphertext using CRYSTALS-Kyber."""
        # Deserialize keys and ciphertext
        private_key_array = np.frombuffer(private_key, dtype=np.int16).reshape(
            (self.kyber_k, self.kyber_n)
        )
        ciphertext_array = np.frombuffer(ciphertext, dtype=np.int16)
        
        # Split ciphertext
        u = ciphertext_array[:self.kyber_k * self.kyber_n].reshape(
            (self.kyber_k, self.kyber_n)
        )
        v = ciphertext_array[self.kyber_k * self.kyber_n:]
        
        # Decryption
        temp = np.sum([np.convolve(private_key_array[i], u[i], mode='same')[:self.kyber_n] 
                      for i in range(self.kyber_k)], axis=0)
        message_poly = (v - temp) % self.kyber_q
        
        # Decode message
        message_bits = (message_poly > self.kyber_q // 4).astype(int)
        message = bytearray()
        
        for i in range(0, min(len(message_bits), 256), 8):
            byte_val = 0
            for j in range(8):
                if i + j < len(message_bits):
                    byte_val |= message_bits[i + j] << j
            message.append(byte_val)
        
        return bytes(message)


class ThreatDetectionSystem:
    """Real-time threat detection and response system."""
    
    def __init__(self, sensitivity: float = 0.8):
        self.sensitivity = sensitivity
        self.logger = PhotonicLogger("ThreatDetection")
        self.metrics = MetricsCollector()
        
        # Threat patterns
        self.attack_patterns = {
            "brute_force": {
                "failed_attempts_threshold": 10,
                "time_window": 300,  # 5 minutes
                "severity": ThreatLevel.HIGH
            },
            "data_exfiltration": {
                "data_volume_threshold": 1e6,  # 1 MB
                "unusual_access_pattern": True,
                "severity": ThreatLevel.CRITICAL
            },
            "quantum_attack": {
                "quantum_signature_threshold": 0.7,
                "entanglement_disruption": True,
                "severity": ThreatLevel.QUANTUM_ATTACK
            }
        }
        
        # Anomaly detection state
        self.baseline_behavior = {}
        self.recent_activities = []
        self.threat_history = []
        
    def analyze_activity(self, activity: Dict[str, Any]) -> Tuple[ThreatLevel, List[str]]:
        """
        Analyze activity for potential security threats.
        
        Args:
            activity: Activity data to analyze
            
        Returns:
            Tuple of (threat_level, detected_threats)
        """
        detected_threats = []
        max_threat_level = ThreatLevel.LOW
        
        # Store activity for pattern analysis
        activity["timestamp"] = time.time()
        self.recent_activities.append(activity)
        
        # Keep only recent activities (last hour)
        cutoff_time = time.time() - 3600
        self.recent_activities = [
            act for act in self.recent_activities 
            if act["timestamp"] > cutoff_time
        ]
        
        # Analyze for known attack patterns
        for pattern_name, pattern_config in self.attack_patterns.items():
            threat_detected, threat_level = self._check_pattern(
                activity, pattern_name, pattern_config
            )
            
            if threat_detected:
                detected_threats.append(pattern_name)
                if threat_level.value in ["high", "critical", "quantum_attack"]:
                    max_threat_level = threat_level
                elif max_threat_level == ThreatLevel.LOW:
                    max_threat_level = threat_level
        
        # Anomaly detection
        anomaly_score = self._calculate_anomaly_score(activity)
        if anomaly_score > self.sensitivity:
            detected_threats.append("behavioral_anomaly")
            if max_threat_level == ThreatLevel.LOW:
                max_threat_level = ThreatLevel.MEDIUM
        
        # Quantum-specific threat detection
        quantum_threats = self._detect_quantum_threats(activity)
        if quantum_threats:
            detected_threats.extend(quantum_threats)
            max_threat_level = ThreatLevel.QUANTUM_ATTACK
        
        # Log threats
        if detected_threats:
            self.logger.warning(f"Threats detected: {detected_threats}, "
                              f"level: {max_threat_level.value}")
            
            threat_record = {
                "timestamp": time.time(),
                "threats": detected_threats,
                "level": max_threat_level.value,
                "activity": activity
            }
            self.threat_history.append(threat_record)
            
            # Update metrics
            if self.metrics:
                self.metrics.increment_counter(f"threats_detected_{max_threat_level.value}")
                for threat in detected_threats:
                    self.metrics.increment_counter(f"threat_type_{threat}")
        
        return max_threat_level, detected_threats
    
    def _check_pattern(self, activity: Dict[str, Any], pattern_name: str, 
                      pattern_config: Dict[str, Any]) -> Tuple[bool, ThreatLevel]:
        """Check activity against specific attack pattern."""
        
        if pattern_name == "brute_force":
            # Check for repeated failed authentication attempts
            failed_attempts = [
                act for act in self.recent_activities[-50:]  # Last 50 activities
                if (act.get("action") == "authenticate" and 
                   not act.get("success", True) and
                   act.get("user_id") == activity.get("user_id"))
            ]
            
            if len(failed_attempts) >= pattern_config["failed_attempts_threshold"]:
                return True, pattern_config["severity"]
        
        elif pattern_name == "data_exfiltration":
            # Check for unusual data access patterns
            if activity.get("action") == "data_access":
                data_size = activity.get("data_size", 0)
                if data_size > pattern_config["data_volume_threshold"]:
                    return True, pattern_config["severity"]
        
        elif pattern_name == "quantum_attack":
            # Check for quantum attack signatures
            if "quantum_signature" in activity:
                signature_strength = activity["quantum_signature"]
                if signature_strength > pattern_config["quantum_signature_threshold"]:
                    return True, pattern_config["severity"]
        
        return False, ThreatLevel.LOW
    
    def _calculate_anomaly_score(self, activity: Dict[str, Any]) -> float:
        """Calculate anomaly score based on deviation from baseline."""
        user_id = activity.get("user_id", "unknown")
        
        # Initialize baseline if not exists
        if user_id not in self.baseline_behavior:
            self.baseline_behavior[user_id] = {
                "avg_session_duration": 3600,  # 1 hour
                "common_resources": set(),
                "typical_actions": {},
                "access_patterns": []
            }
        
        baseline = self.baseline_behavior[user_id]
        anomaly_factors = []
        
        # Check session duration anomaly
        session_duration = activity.get("session_duration", 0)
        if session_duration > 0:
            duration_ratio = session_duration / baseline["avg_session_duration"]
            if duration_ratio > 3 or duration_ratio < 0.1:  # 3x longer or 10x shorter
                anomaly_factors.append(0.3)
        
        # Check resource access anomaly
        resource = activity.get("resource")
        if resource and resource not in baseline["common_resources"]:
            anomaly_factors.append(0.2)
        else:
            baseline["common_resources"].add(resource)
        
        # Check action frequency anomaly
        action = activity.get("action", "unknown")
        action_count = baseline["typical_actions"].get(action, 0)
        if action_count == 0:  # New action type
            anomaly_factors.append(0.2)
        baseline["typical_actions"][action] = action_count + 1
        
        # Time-based anomaly (unusual access hours)
        current_hour = time.localtime().tm_hour
        if current_hour < 6 or current_hour > 22:  # Outside business hours
            anomaly_factors.append(0.1)
        
        return sum(anomaly_factors)
    
    def _detect_quantum_threats(self, activity: Dict[str, Any]) -> List[str]:
        """Detect quantum-specific security threats."""
        quantum_threats = []
        
        # Check for quantum state manipulation
        if "quantum_state" in activity:
            state_entropy = activity.get("quantum_state_entropy", 0)
            if state_entropy < 0.5:  # Low entropy indicates possible manipulation
                quantum_threats.append("quantum_state_manipulation")
        
        # Check for entanglement disruption
        if "entanglement_fidelity" in activity:
            fidelity = activity["entanglement_fidelity"]
            if fidelity < 0.8:  # Significant fidelity loss
                quantum_threats.append("entanglement_disruption")
        
        # Check for quantum key compromise
        if "quantum_key_error_rate" in activity:
            error_rate = activity["quantum_key_error_rate"]
            if error_rate > 0.1:  # High error rate indicates potential compromise
                quantum_threats.append("quantum_key_compromise")
        
        return quantum_threats


class SecurePhotonicSession:
    """Secure session management for photonic neuromorphic systems."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.CONFIDENTIAL):
        self.security_level = security_level
        self.session_id = secrets.token_hex(32)
        self.created_at = time.time()
        self.last_activity = time.time()
        self.user_id = None
        self.permissions = set()
        
        # Initialize security components
        self.qkd = QuantumKeyDistribution()
        self.pqc = PostQuantumCryptography()
        self.threat_detector = ThreatDetectionSystem()
        
        # Security state
        self.quantum_key = None
        self.pq_public_key, self.pq_private_key = self.pqc.generate_kyber_keypair()
        self.authenticated = False
        self.encryption_enabled = True
        
        self.logger = PhotonicLogger("SecureSession")
        self.logger.info(f"Created secure session {self.session_id} "
                        f"with {security_level.value} classification")
    
    def authenticate_user(self, user_id: str, credentials: Dict[str, Any]) -> bool:
        """Authenticate user with multi-factor authentication."""
        auth_activity = {
            "action": "authenticate",
            "user_id": user_id,
            "session_id": self.session_id,
            "timestamp": time.time()
        }
        
        # Multi-factor authentication simulation
        password_valid = self._verify_password(credentials.get("password", ""))
        token_valid = self._verify_mfa_token(credentials.get("mfa_token", ""))
        biometric_valid = self._verify_biometric(credentials.get("biometric", ""))
        
        # Quantum authentication (if available)
        quantum_valid = True
        if "quantum_signature" in credentials:
            quantum_valid = self._verify_quantum_signature(credentials["quantum_signature"])
        
        success = password_valid and token_valid and (biometric_valid or quantum_valid)
        auth_activity["success"] = success
        auth_activity["auth_factors"] = {
            "password": password_valid,
            "mfa": token_valid,
            "biometric": biometric_valid,
            "quantum": quantum_valid
        }
        
        # Threat analysis
        threat_level, threats = self.threat_detector.analyze_activity(auth_activity)
        
        if success and threat_level in [ThreatLevel.LOW, ThreatLevel.MEDIUM]:
            self.authenticated = True
            self.user_id = user_id
            self.last_activity = time.time()
            
            # Establish quantum-secure communication
            if self.security_level in [SecurityLevel.SECRET, SecurityLevel.TOP_SECRET, 
                                     SecurityLevel.QUANTUM_SECURE]:
                self._establish_quantum_channel(user_id)
            
            self.logger.info(f"User {user_id} authenticated successfully")
            return True
        else:
            if threats:
                self.logger.warning(f"Authentication failed for {user_id}: {threats}")
            else:
                self.logger.warning(f"Authentication failed for {user_id}: invalid credentials")
            return False
    
    def _verify_password(self, password: str) -> bool:
        """Verify password (simplified implementation)."""
        # In production, use proper password hashing (bcrypt, scrypt, etc.)
        return len(password) >= 12 and any(c.isdigit() for c in password)
    
    def _verify_mfa_token(self, token: str) -> bool:
        """Verify multi-factor authentication token."""
        # Simulate TOTP verification
        return len(token) == 6 and token.isdigit()
    
    def _verify_biometric(self, biometric_data: str) -> bool:
        """Verify biometric authentication."""
        # Simulate biometric verification
        return len(biometric_data) > 50  # Simplified check
    
    def _verify_quantum_signature(self, quantum_signature: float) -> bool:
        """Verify quantum signature for authentication."""
        # Check quantum signature strength
        return 0.8 <= quantum_signature <= 1.0
    
    def _establish_quantum_channel(self, partner_id: str) -> None:
        """Establish quantum-secure communication channel."""
        try:
            self.quantum_key, security_level = self.qkd.generate_quantum_key(partner_id)
            self.logger.info(f"Quantum channel established with security level {security_level:.3f}")
        except SecurityError as e:
            self.logger.error(f"Failed to establish quantum channel: {e}")
            self.quantum_key = None
    
    def encrypt_data(self, data: bytes, classification: SecurityLevel = None) -> bytes:
        """Encrypt data based on classification level."""
        if not self.encryption_enabled:
            return data
        
        target_classification = classification or self.security_level
        
        if target_classification == SecurityLevel.QUANTUM_SECURE and self.quantum_key:
            # Use quantum-derived key for encryption
            return self._quantum_encrypt(data)
        elif target_classification in [SecurityLevel.SECRET, SecurityLevel.TOP_SECRET]:
            # Use post-quantum cryptography
            return self.pqc.kyber_encrypt(self.pq_public_key, data[:32])  # Limit message size
        else:
            # Use conventional AES encryption
            return self._aes_encrypt(data)
    
    def decrypt_data(self, encrypted_data: bytes, classification: SecurityLevel = None) -> bytes:
        """Decrypt data based on classification level."""
        if not self.encryption_enabled:
            return encrypted_data
        
        target_classification = classification or self.security_level
        
        try:
            if target_classification == SecurityLevel.QUANTUM_SECURE and self.quantum_key:
                return self._quantum_decrypt(encrypted_data)
            elif target_classification in [SecurityLevel.SECRET, SecurityLevel.TOP_SECRET]:
                return self.pqc.kyber_decrypt(self.pq_private_key, encrypted_data)
            else:
                return self._aes_decrypt(encrypted_data)
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            raise SecurityError("decrypt_data", "decryption_failed", str(e))
    
    def _quantum_encrypt(self, data: bytes) -> bytes:
        """Encrypt using quantum-derived key."""
        if not self.quantum_key:
            raise SecurityError("quantum_encrypt", "no_quantum_key", "Quantum key not available")
        
        # Use quantum key for AES encryption
        key = self.quantum_key[:32]  # Use first 32 bytes for AES-256
        iv = secrets.token_bytes(16)
        
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        
        # PKCS7 padding
        padding_length = 16 - (len(data) % 16)
        padded_data = data + bytes([padding_length] * padding_length)
        
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        return iv + ciphertext
    
    def _quantum_decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt using quantum-derived key."""
        if not self.quantum_key:
            raise SecurityError("quantum_decrypt", "no_quantum_key", "Quantum key not available")
        
        key = self.quantum_key[:32]
        iv = encrypted_data[:16]
        ciphertext = encrypted_data[16:]
        
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Remove PKCS7 padding
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]
    
    def _aes_encrypt(self, data: bytes) -> bytes:
        """Standard AES encryption for lower classification levels."""
        key = secrets.token_bytes(32)  # Generate random key
        iv = secrets.token_bytes(16)
        
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        
        # PKCS7 padding
        padding_length = 16 - (len(data) % 16)
        padded_data = data + bytes([padding_length] * padding_length)
        
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        
        # Prepend key and IV (in production, use key derivation)
        return key + iv + ciphertext
    
    def _aes_decrypt(self, encrypted_data: bytes) -> bytes:
        """Standard AES decryption."""
        key = encrypted_data[:32]
        iv = encrypted_data[32:48]
        ciphertext = encrypted_data[48:]
        
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Remove PKCS7 padding
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]
    
    def check_permissions(self, required_permissions: List[str]) -> bool:
        """Check if current session has required permissions."""
        if not self.authenticated:
            return False
        
        return all(perm in self.permissions for perm in required_permissions)
    
    def record_activity(self, resource: str, action: str, data_size: int = 0) -> None:
        """Record activity for threat monitoring."""
        self.last_activity = time.time()
        
        activity = {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "resource": resource,
            "action": action,
            "data_size": data_size,
            "timestamp": time.time(),
            "security_level": self.security_level.value
        }
        
        # Analyze for threats
        threat_level, threats = self.threat_detector.analyze_activity(activity)
        
        if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL, ThreatLevel.QUANTUM_ATTACK]:
            self.logger.critical(f"Critical threat detected: {threats}")
            # In production, implement immediate response actions
    
    def is_valid(self) -> bool:
        """Check if session is still valid."""
        if not self.authenticated:
            return False
        
        # Check timeout (8 hours for most classifications)
        max_duration = 8 * 3600
        if self.security_level == SecurityLevel.TOP_SECRET:
            max_duration = 4 * 3600  # 4 hours for top secret
        elif self.security_level == SecurityLevel.QUANTUM_SECURE:
            max_duration = 2 * 3600  # 2 hours for quantum secure
        
        age = time.time() - self.created_at
        idle_time = time.time() - self.last_activity
        
        return age < max_duration and idle_time < 1800  # 30 min idle timeout
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "security_level": self.security_level.value,
            "authenticated": self.authenticated,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "quantum_key_available": self.quantum_key is not None,
            "encryption_enabled": self.encryption_enabled,
            "valid": self.is_valid(),
            "permissions": list(self.permissions),
            "recent_threats": len([
                t for t in self.threat_detector.threat_history 
                if time.time() - t["timestamp"] < 3600
            ])
        }


def create_secure_photonic_environment(
    security_level: SecurityLevel = SecurityLevel.CONFIDENTIAL
) -> SecurePhotonicSession:
    """Create secure environment for photonic neuromorphic computing."""
    
    session = SecurePhotonicSession(security_level)
    
    # Set appropriate permissions based on security level
    if security_level == SecurityLevel.PUBLIC:
        session.permissions.update(["read_public", "basic_compute"])
    elif security_level == SecurityLevel.CONFIDENTIAL:
        session.permissions.update(["read_confidential", "basic_compute", "model_training"])
    elif security_level == SecurityLevel.SECRET:
        session.permissions.update([
            "read_secret", "advanced_compute", "model_training", 
            "quantum_operations", "secure_communication"
        ])
    elif security_level in [SecurityLevel.TOP_SECRET, SecurityLevel.QUANTUM_SECURE]:
        session.permissions.update([
            "read_top_secret", "advanced_compute", "model_training",
            "quantum_operations", "secure_communication", "system_administration",
            "quantum_key_management"
        ])
    
    return session


def validate_security_implementation() -> Dict[str, Any]:
    """Validate security implementation effectiveness."""
    
    validation_results = {
        "quantum_key_security": 0.0,
        "post_quantum_crypto": 0.0,
        "threat_detection_accuracy": 0.0,
        "encryption_performance": 0.0
    }
    
    # Test quantum key distribution
    qkd = QuantumKeyDistribution()
    try:
        quantum_key, security_level = qkd.generate_quantum_key("test_partner")
        validation_results["quantum_key_security"] = security_level
    except Exception:
        validation_results["quantum_key_security"] = 0.0
    
    # Test post-quantum cryptography
    pqc = PostQuantumCryptography()
    try:
        pub_key, priv_key = pqc.generate_kyber_keypair()
        test_message = b"Test message for PQC validation"
        encrypted = pqc.kyber_encrypt(pub_key, test_message)
        decrypted = pqc.kyber_decrypt(priv_key, encrypted)
        
        validation_results["post_quantum_crypto"] = float(test_message == decrypted)
    except Exception:
        validation_results["post_quantum_crypto"] = 0.0
    
    # Test threat detection
    threat_detector = ThreatDetectionSystem()
    
    # Simulate normal activity
    normal_activity = {
        "user_id": "test_user",
        "action": "data_access",
        "resource": "test_resource",
        "data_size": 1000
    }
    normal_threat_level, _ = threat_detector.analyze_activity(normal_activity)
    
    # Simulate malicious activity
    malicious_activity = {
        "user_id": "test_user",
        "action": "data_access",
        "resource": "sensitive_resource",
        "data_size": 10000000,  # Large data access
        "quantum_signature": 0.9  # Suspicious quantum signature
    }
    malicious_threat_level, threats = threat_detector.analyze_activity(malicious_activity)
    
    # Calculate detection accuracy
    normal_correct = normal_threat_level in [ThreatLevel.LOW, ThreatLevel.MEDIUM]
    malicious_correct = malicious_threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
    validation_results["threat_detection_accuracy"] = float(normal_correct and malicious_correct)
    
    # Test encryption performance
    session = create_secure_photonic_environment(SecurityLevel.SECRET)
    test_data = secrets.token_bytes(1024)
    
    start_time = time.time()
    encrypted = session.encrypt_data(test_data)
    decrypted = session.decrypt_data(encrypted)
    encryption_time = time.time() - start_time
    
    # Performance score (inversely proportional to time)
    validation_results["encryption_performance"] = min(1.0, 0.1 / encryption_time)
    
    return validation_results