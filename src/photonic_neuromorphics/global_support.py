"""
Global-First Implementation Support for Photonic Neuromorphic Systems.

This module provides comprehensive internationalization (i18n), localization (l10n),
compliance, and cross-platform support for worldwide deployment of photonic
neural networks with GDPR, CCPA, and other regulatory compliance.
"""

import os
import sys
import locale
import json
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import threading
from enum import Enum
import hashlib
import base64

# Import for timezone support
try:
    import pytz
except ImportError:
    pytz = None

# Import for encryption (data protection)
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


class SupportedLanguage(Enum):
    """Supported languages for i18n."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    KOREAN = "ko"
    RUSSIAN = "ru"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    DUTCH = "nl"


class ComplianceRegion(Enum):
    """Compliance regions with different regulatory requirements."""
    EU = "eu"          # GDPR
    US = "us"          # CCPA, COPPA
    CANADA = "ca"      # PIPEDA
    UK = "uk"          # UK GDPR
    SINGAPORE = "sg"   # PDPA
    JAPAN = "jp"       # APPI
    AUSTRALIA = "au"   # Privacy Act
    BRAZIL = "br"      # LGPD
    INDIA = "in"       # Personal Data Protection Bill
    GLOBAL = "global"  # Universal compliance


class DataCategory(Enum):
    """Categories of data for compliance tracking."""
    PERSONAL_IDENTIFIABLE = "pii"
    SENSITIVE_PERSONAL = "sensitive"
    BIOMETRIC = "biometric"
    HEALTH = "health"
    FINANCIAL = "financial"
    BEHAVIORAL = "behavioral"
    TECHNICAL = "technical"
    ANONYMOUS = "anonymous"


@dataclass
class LocalizationConfig:
    """Configuration for localization settings."""
    language: SupportedLanguage = SupportedLanguage.ENGLISH
    region: str = "US"
    timezone: str = "UTC"
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    number_format: str = "en_US"
    currency: str = "USD"
    decimal_separator: str = "."
    thousands_separator: str = ","
    rtl_support: bool = False  # Right-to-left languages


@dataclass
class ComplianceConfig:
    """Configuration for regulatory compliance."""
    primary_region: ComplianceRegion = ComplianceRegion.GLOBAL
    additional_regions: List[ComplianceRegion] = field(default_factory=list)
    data_retention_days: int = 365
    anonymization_enabled: bool = True
    encryption_enabled: bool = True
    audit_logging: bool = True
    consent_management: bool = True
    data_minimization: bool = True
    purpose_limitation: bool = True
    storage_limitation: bool = True


@dataclass  
class DataProcessingRecord:
    """Record of data processing for compliance."""
    record_id: str
    data_subject_id: Optional[str]
    data_categories: List[DataCategory]
    processing_purpose: str
    legal_basis: str
    processor: str
    timestamp: datetime
    retention_period: int  # days
    anonymized: bool = False
    encrypted: bool = False
    consent_given: bool = False
    consent_withdrawn: bool = False


class InternationalizationManager:
    """Comprehensive i18n and l10n manager."""
    
    def __init__(
        self,
        config: LocalizationConfig,
        translations_dir: Optional[str] = None
    ):
        self.config = config
        self.translations_dir = Path(translations_dir) if translations_dir else Path(__file__).parent / "translations"
        
        # Language translations
        self.translations: Dict[str, Dict[str, str]] = {}
        self.current_language = config.language
        
        # Locale settings
        self.locale_lock = threading.Lock()
        
        # Initialize
        self._load_translations()
        self._setup_locale()
        
        self.logger = logging.getLogger(__name__)
    
    def _load_translations(self) -> None:
        """Load translation files."""
        try:
            # Create translations directory if it doesn't exist
            self.translations_dir.mkdir(parents=True, exist_ok=True)
            
            # Load translation files
            for lang in SupportedLanguage:
                lang_file = self.translations_dir / f"{lang.value}.json"
                
                if lang_file.exists():
                    with open(lang_file, 'r', encoding='utf-8') as f:
                        self.translations[lang.value] = json.load(f)
                else:
                    # Create empty translation file
                    self.translations[lang.value] = {}
                    self._create_default_translations(lang.value)
            
        except Exception as e:
            self.logger.error(f"Failed to load translations: {e}")
            # Fallback to English
            self.translations = {"en": self._get_default_translations()}
    
    def _create_default_translations(self, language: str) -> None:
        """Create default translation file for a language."""
        try:
            default_translations = self._get_default_translations()
            
            # Save to file
            lang_file = self.translations_dir / f"{language}.json"
            with open(lang_file, 'w', encoding='utf-8') as f:
                json.dump(default_translations, f, indent=2, ensure_ascii=False)
            
            self.translations[language] = default_translations
            
        except Exception as e:
            self.logger.error(f"Failed to create translations for {language}: {e}")
    
    def _get_default_translations(self) -> Dict[str, str]:
        """Get default English translations."""
        return {
            # Common terms
            "error": "Error",
            "warning": "Warning",
            "info": "Information",
            "success": "Success",
            "loading": "Loading",
            "processing": "Processing",
            "complete": "Complete",
            "failed": "Failed",
            
            # Photonic neuromorphic specific
            "photonic_network": "Photonic Network",
            "neural_simulation": "Neural Simulation",
            "optical_power": "Optical Power",
            "wavelength": "Wavelength",
            "simulation_mode": "Simulation Mode",
            "spike_train": "Spike Train",
            "neuron_threshold": "Neuron Threshold",
            "synaptic_weight": "Synaptic Weight",
            "plasticity_rule": "Plasticity Rule",
            "rtl_generation": "RTL Generation",
            
            # System messages
            "initialization_complete": "System initialization complete",
            "simulation_started": "Simulation started",
            "simulation_complete": "Simulation completed successfully",
            "configuration_loaded": "Configuration loaded",
            "optimization_enabled": "Performance optimization enabled",
            "scaling_active": "Auto-scaling is active",
            
            # Error messages
            "invalid_input": "Invalid input provided",
            "simulation_failed": "Simulation failed",
            "configuration_error": "Configuration error",
            "resource_exhausted": "System resources exhausted",
            "optical_model_error": "Optical model error",
            "convergence_failed": "Simulation convergence failed",
            
            # Compliance and privacy
            "data_processing_notice": "Data processing notice",
            "consent_required": "Consent required for data processing",
            "privacy_policy": "Privacy Policy",
            "data_retention": "Data retention period",
            "anonymization_applied": "Data anonymization applied",
            "encryption_enabled": "Data encryption enabled",
            "audit_log_entry": "Audit log entry created",
            "consent_withdrawn": "Data processing consent withdrawn",
            "data_deleted": "Personal data deleted as requested",
            
            # Units and formatting
            "nanometer": "nm",
            "micrometer": "μm", 
            "milliwatt": "mW",
            "microwatt": "μW",
            "nanosecond": "ns",
            "picosecond": "ps",
            "hertz": "Hz",
            "kilohertz": "kHz",
            "megahertz": "MHz",
            "gigahertz": "GHz"
        }
    
    def _setup_locale(self) -> None:
        """Setup system locale based on configuration."""
        try:
            with self.locale_lock:
                # Set locale for number formatting
                locale_name = f"{self.config.language.value}_{self.config.region}"
                
                # Try to set the locale
                try:
                    locale.setlocale(locale.LC_ALL, locale_name)
                except locale.Error:
                    # Fallback to C locale
                    locale.setlocale(locale.LC_ALL, 'C')
                    self.logger.warning(f"Could not set locale {locale_name}, using C locale")
                
        except Exception as e:
            self.logger.error(f"Locale setup failed: {e}")
    
    def translate(self, key: str, **kwargs) -> str:
        """Translate a key to the current language."""
        lang_code = self.current_language.value
        
        # Get translation
        if lang_code in self.translations and key in self.translations[lang_code]:
            translation = self.translations[lang_code][key]
        elif key in self.translations.get("en", {}):
            # Fallback to English
            translation = self.translations["en"][key]
        else:
            # Fallback to key itself
            translation = key.replace("_", " ").title()
        
        # Format with kwargs
        try:
            if kwargs:
                translation = translation.format(**kwargs)
        except (KeyError, ValueError) as e:
            self.logger.warning(f"Translation formatting failed for '{key}': {e}")
        
        return translation
    
    def set_language(self, language: SupportedLanguage) -> None:
        """Change current language."""
        self.current_language = language
        self.logger.info(f"Language changed to: {language.value}")
    
    def format_number(self, number: Union[int, float], decimals: int = 2) -> str:
        """Format number according to locale."""
        try:
            if isinstance(number, int):
                return locale.format_string("%d", number, grouping=True)
            else:
                format_str = f"%.{decimals}f"
                return locale.format_string(format_str, number, grouping=True)
        except:
            # Fallback formatting
            if isinstance(number, int):
                return f"{number:,}"
            else:
                return f"{number:,.{decimals}f}"
    
    def format_datetime(self, dt: datetime) -> str:
        """Format datetime according to locale."""
        try:
            if pytz and self.config.timezone != "UTC":
                tz = pytz.timezone(self.config.timezone)
                dt = dt.replace(tzinfo=timezone.utc).astimezone(tz)
            
            date_str = dt.strftime(self.config.date_format)
            time_str = dt.strftime(self.config.time_format)
            
            return f"{date_str} {time_str}"
            
        except Exception as e:
            self.logger.error(f"DateTime formatting failed: {e}")
            return dt.isoformat()
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages."""
        return [
            {
                "code": lang.value,
                "name": self._get_language_name(lang),
                "native_name": self._get_native_language_name(lang)
            }
            for lang in SupportedLanguage
        ]
    
    def _get_language_name(self, language: SupportedLanguage) -> str:
        """Get English name of language."""
        names = {
            SupportedLanguage.ENGLISH: "English",
            SupportedLanguage.SPANISH: "Spanish",
            SupportedLanguage.FRENCH: "French",
            SupportedLanguage.GERMAN: "German",
            SupportedLanguage.JAPANESE: "Japanese",
            SupportedLanguage.CHINESE_SIMPLIFIED: "Chinese (Simplified)",
            SupportedLanguage.CHINESE_TRADITIONAL: "Chinese (Traditional)",
            SupportedLanguage.KOREAN: "Korean",
            SupportedLanguage.RUSSIAN: "Russian",
            SupportedLanguage.PORTUGUESE: "Portuguese",
            SupportedLanguage.ITALIAN: "Italian",
            SupportedLanguage.DUTCH: "Dutch"
        }
        return names.get(language, language.value)
    
    def _get_native_language_name(self, language: SupportedLanguage) -> str:
        """Get native name of language."""
        names = {
            SupportedLanguage.ENGLISH: "English",
            SupportedLanguage.SPANISH: "Español",
            SupportedLanguage.FRENCH: "Français",
            SupportedLanguage.GERMAN: "Deutsch",
            SupportedLanguage.JAPANESE: "日本語",
            SupportedLanguage.CHINESE_SIMPLIFIED: "简体中文",
            SupportedLanguage.CHINESE_TRADITIONAL: "繁體中文",
            SupportedLanguage.KOREAN: "한국어",
            SupportedLanguage.RUSSIAN: "Русский",
            SupportedLanguage.PORTUGUESE: "Português",
            SupportedLanguage.ITALIAN: "Italiano",
            SupportedLanguage.DUTCH: "Nederlands"
        }
        return names.get(language, language.value)


class ComplianceManager:
    """Comprehensive compliance management system."""
    
    def __init__(
        self,
        config: ComplianceConfig,
        i18n_manager: Optional[InternationalizationManager] = None
    ):
        self.config = config
        self.i18n = i18n_manager
        
        # Data processing records
        self.processing_records: Dict[str, DataProcessingRecord] = {}
        self.processing_lock = threading.Lock()
        
        # Encryption setup
        self.encryption_key: Optional[bytes] = None
        self.cipher_suite = None
        
        if CRYPTO_AVAILABLE and config.encryption_enabled:
            self._setup_encryption()
        
        # Audit logging
        self.audit_logger = None
        if config.audit_logging:
            self._setup_audit_logging()
        
        self.logger = logging.getLogger(__name__)
    
    def _setup_encryption(self) -> None:
        """Setup encryption for data protection."""
        try:
            # In production, this should be loaded from secure key management
            password = b"photonic_neuromorphic_secure_key"  # Should be from environment
            salt = b"stable_salt_for_consistency"  # Should be random per deployment
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            
            self.encryption_key = base64.urlsafe_b64encode(kdf.derive(password))
            self.cipher_suite = Fernet(self.encryption_key)
            
            self.logger.info("Encryption initialized for data protection")
            
        except Exception as e:
            self.logger.error(f"Encryption setup failed: {e}")
            self.config.encryption_enabled = False
    
    def _setup_audit_logging(self) -> None:
        """Setup audit logging."""
        try:
            # Create audit logger
            self.audit_logger = logging.getLogger("photonic_audit")
            self.audit_logger.setLevel(logging.INFO)
            
            # Create file handler for audit logs
            audit_file = Path("logs") / "audit.log"
            audit_file.parent.mkdir(exist_ok=True)
            
            handler = logging.FileHandler(audit_file, encoding='utf-8')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            
            self.audit_logger.addHandler(handler)
            
            self.logger.info("Audit logging initialized")
            
        except Exception as e:
            self.logger.error(f"Audit logging setup failed: {e}")
            self.config.audit_logging = False
    
    def encrypt_data(self, data: Union[str, bytes]) -> str:
        """Encrypt sensitive data."""
        if not self.config.encryption_enabled or not self.cipher_suite:
            return data if isinstance(data, str) else data.decode()
        
        try:
            if isinstance(data, str):
                data = data.encode()
            
            encrypted_data = self.cipher_suite.encrypt(data)
            return base64.urlsafe_b64encode(encrypted_data).decode()
            
        except Exception as e:
            self.logger.error(f"Data encryption failed: {e}")
            return data if isinstance(data, str) else data.decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        if not self.config.encryption_enabled or not self.cipher_suite:
            return encrypted_data
        
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.cipher_suite.decrypt(encrypted_bytes)
            return decrypted_data.decode()
            
        except Exception as e:
            self.logger.error(f"Data decryption failed: {e}")
            return encrypted_data
    
    def record_data_processing(
        self,
        data_subject_id: Optional[str],
        data_categories: List[DataCategory],
        processing_purpose: str,
        legal_basis: str,
        processor: str = "photonic_neuromorphic_system",
        consent_given: bool = False,
        retention_days: Optional[int] = None
    ) -> str:
        """Record data processing activity for compliance."""
        
        # Generate record ID
        record_id = self._generate_record_id(data_subject_id, processing_purpose)
        
        # Create processing record
        record = DataProcessingRecord(
            record_id=record_id,
            data_subject_id=self.encrypt_data(data_subject_id) if data_subject_id else None,
            data_categories=data_categories,
            processing_purpose=processing_purpose,
            legal_basis=legal_basis,
            processor=processor,
            timestamp=datetime.now(timezone.utc),
            retention_period=retention_days or self.config.data_retention_days,
            anonymized=self.config.anonymization_enabled and not data_subject_id,
            encrypted=self.config.encryption_enabled and data_subject_id is not None,
            consent_given=consent_given
        )
        
        # Store record
        with self.processing_lock:
            self.processing_records[record_id] = record
        
        # Audit log entry
        if self.audit_logger:
            self.audit_logger.info(
                f"Data processing recorded - ID: {record_id}, "
                f"Purpose: {processing_purpose}, "
                f"Categories: {[cat.value for cat in data_categories]}, "
                f"Consent: {consent_given}"
            )
        
        self.logger.debug(f"Data processing recorded: {record_id}")
        return record_id
    
    def _generate_record_id(self, data_subject_id: Optional[str], purpose: str) -> str:
        """Generate unique record ID."""
        content = f"{data_subject_id or 'anonymous'}:{purpose}:{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def withdraw_consent(self, data_subject_id: str) -> List[str]:
        """Handle consent withdrawal - mark records and schedule deletion."""
        affected_records = []
        encrypted_subject_id = self.encrypt_data(data_subject_id)
        
        with self.processing_lock:
            for record_id, record in self.processing_records.items():
                if record.data_subject_id == encrypted_subject_id:
                    record.consent_withdrawn = True
                    affected_records.append(record_id)
        
        # Audit log
        if self.audit_logger:
            self.audit_logger.info(
                f"Consent withdrawn for data subject, "
                f"affected records: {len(affected_records)}"
            )
        
        # Schedule data deletion
        self._schedule_data_deletion(affected_records)
        
        return affected_records
    
    def _schedule_data_deletion(self, record_ids: List[str]) -> None:
        """Schedule data deletion (placeholder for actual implementation)."""
        # In a real system, this would schedule actual data deletion
        for record_id in record_ids:
            self.logger.info(f"Scheduled deletion for record: {record_id}")
    
    def anonymize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize personal data."""
        if not self.config.anonymization_enabled:
            return data
        
        anonymized = data.copy()
        
        # Remove direct identifiers
        identifiers = ['id', 'email', 'phone', 'name', 'address', 'ip_address']
        for identifier in identifiers:
            if identifier in anonymized:
                del anonymized[identifier]
        
        # Hash quasi-identifiers
        quasi_identifiers = ['user_agent', 'session_id', 'device_id']
        for qi in quasi_identifiers:
            if qi in anonymized:
                anonymized[qi] = hashlib.sha256(str(anonymized[qi]).encode()).hexdigest()[:8]
        
        return anonymized
    
    def check_retention_compliance(self) -> List[str]:
        """Check data retention compliance and identify expired records."""
        expired_records = []
        current_time = datetime.now(timezone.utc)
        
        with self.processing_lock:
            for record_id, record in self.processing_records.items():
                retention_deadline = record.timestamp + timedelta(days=record.retention_period)
                
                if current_time > retention_deadline:
                    expired_records.append(record_id)
        
        return expired_records
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report."""
        with self.processing_lock:
            total_records = len(self.processing_records)
            encrypted_records = sum(1 for r in self.processing_records.values() if r.encrypted)
            anonymized_records = sum(1 for r in self.processing_records.values() if r.anonymized)
            consent_given_records = sum(1 for r in self.processing_records.values() if r.consent_given)
            withdrawn_consent_records = sum(1 for r in self.processing_records.values() if r.consent_withdrawn)
        
        expired_records = self.check_retention_compliance()
        
        return {
            "compliance_regions": [r.value for r in [self.config.primary_region] + self.config.additional_regions],
            "total_processing_records": total_records,
            "encrypted_records": encrypted_records,
            "anonymized_records": anonymized_records,
            "consent_given_records": consent_given_records,
            "withdrawn_consent_records": withdrawn_consent_records,
            "expired_records": len(expired_records),
            "encryption_enabled": self.config.encryption_enabled,
            "anonymization_enabled": self.config.anonymization_enabled,
            "audit_logging_enabled": self.config.audit_logging,
            "data_retention_days": self.config.data_retention_days
        }


class CrossPlatformManager:
    """Cross-platform compatibility manager."""
    
    def __init__(self):
        self.platform = sys.platform
        self.python_version = sys.version_info
        
        # Platform-specific configurations
        self.platform_configs = {
            "win32": self._get_windows_config(),
            "linux": self._get_linux_config(), 
            "darwin": self._get_macos_config()
        }
        
        self.logger = logging.getLogger(__name__)
    
    def _get_windows_config(self) -> Dict[str, Any]:
        """Windows-specific configuration."""
        return {
            "path_separator": "\\",
            "line_ending": "\r\n",
            "case_sensitive": False,
            "max_path_length": 260,
            "supports_symlinks": False,  # Requires admin privileges
            "default_encoding": "utf-8",
            "temp_dir": os.environ.get("TEMP", "C:\\temp"),
            "config_dir": os.path.expanduser("~\\AppData\\Local\\PhotonicNeuromorphic"),
            "log_dir": os.path.expanduser("~\\AppData\\Local\\PhotonicNeuromorphic\\logs")
        }
    
    def _get_linux_config(self) -> Dict[str, Any]:
        """Linux-specific configuration.""" 
        return {
            "path_separator": "/",
            "line_ending": "\n", 
            "case_sensitive": True,
            "max_path_length": 4096,
            "supports_symlinks": True,
            "default_encoding": "utf-8",
            "temp_dir": "/tmp",
            "config_dir": os.path.expanduser("~/.config/photonic-neuromorphic"),
            "log_dir": os.path.expanduser("~/.local/share/photonic-neuromorphic/logs")
        }
    
    def _get_macos_config(self) -> Dict[str, Any]:
        """macOS-specific configuration."""
        return {
            "path_separator": "/",
            "line_ending": "\n",
            "case_sensitive": False,  # By default, can be configured
            "max_path_length": 1024,
            "supports_symlinks": True,
            "default_encoding": "utf-8",
            "temp_dir": "/tmp",
            "config_dir": os.path.expanduser("~/Library/Application Support/PhotonicNeuromorphic"),
            "log_dir": os.path.expanduser("~/Library/Logs/PhotonicNeuromorphic")
        }
    
    def get_platform_config(self) -> Dict[str, Any]:
        """Get configuration for current platform."""
        return self.platform_configs.get(self.platform, self.platform_configs["linux"])
    
    def ensure_directories(self) -> None:
        """Ensure required directories exist."""
        config = self.get_platform_config()
        
        for dir_type in ["config_dir", "log_dir"]:
            directory = Path(config[dir_type])
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Ensured {dir_type}: {directory}")
    
    def normalize_path(self, path: Union[str, Path]) -> Path:
        """Normalize path for current platform."""
        return Path(path).resolve()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        return {
            "platform": self.platform,
            "python_version": f"{self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}",
            "architecture": os.uname().machine if hasattr(os, 'uname') else 'unknown',
            "cpu_count": os.cpu_count(),
            "encoding": sys.getdefaultencoding(),
            "file_system_encoding": sys.getfilesystemencoding(),
            "max_path_length": self.get_platform_config()["max_path_length"],
            "supports_symlinks": self.get_platform_config()["supports_symlinks"]
        }


class GlobalSupportManager:
    """Comprehensive global support management."""
    
    def __init__(
        self,
        localization_config: Optional[LocalizationConfig] = None,
        compliance_config: Optional[ComplianceConfig] = None,
        translations_dir: Optional[str] = None
    ):
        # Initialize configurations
        self.localization_config = localization_config or LocalizationConfig()
        self.compliance_config = compliance_config or ComplianceConfig()
        
        # Initialize managers
        self.i18n_manager = InternationalizationManager(
            self.localization_config, 
            translations_dir
        )
        self.compliance_manager = ComplianceManager(
            self.compliance_config,
            self.i18n_manager
        )
        self.platform_manager = CrossPlatformManager()
        
        # Setup directories
        self.platform_manager.ensure_directories()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Global support manager initialized")
    
    def translate(self, key: str, **kwargs) -> str:
        """Translate text to current language."""
        return self.i18n_manager.translate(key, **kwargs)
    
    def format_number(self, number: Union[int, float], decimals: int = 2) -> str:
        """Format number according to locale."""
        return self.i18n_manager.format_number(number, decimals)
    
    def format_datetime(self, dt: datetime) -> str:
        """Format datetime according to locale."""
        return self.i18n_manager.format_datetime(dt)
    
    def record_data_processing(
        self,
        data_subject_id: Optional[str],
        data_categories: List[DataCategory], 
        processing_purpose: str,
        legal_basis: str,
        consent_given: bool = False
    ) -> str:
        """Record data processing for compliance."""
        return self.compliance_manager.record_data_processing(
            data_subject_id, data_categories, processing_purpose, 
            legal_basis, consent_given=consent_given
        )
    
    def encrypt_sensitive_data(self, data: Union[str, bytes]) -> str:
        """Encrypt sensitive data."""
        return self.compliance_manager.encrypt_data(data)
    
    def withdraw_consent(self, data_subject_id: str) -> List[str]:
        """Handle consent withdrawal."""
        return self.compliance_manager.withdraw_consent(data_subject_id)
    
    def get_global_status(self) -> Dict[str, Any]:
        """Get comprehensive global support status."""
        return {
            "localization": {
                "current_language": self.i18n_manager.current_language.value,
                "supported_languages": len(SupportedLanguage),
                "timezone": self.localization_config.timezone,
                "date_format": self.localization_config.date_format
            },
            "compliance": self.compliance_manager.get_compliance_report(),
            "platform": self.platform_manager.get_system_info(),
            "features": {
                "i18n_enabled": True,
                "compliance_enabled": True,
                "encryption_available": CRYPTO_AVAILABLE,
                "encryption_enabled": self.compliance_config.encryption_enabled,
                "audit_logging": self.compliance_config.audit_logging
            }
        }


# Convenience functions for common operations
def create_global_support(
    language: SupportedLanguage = SupportedLanguage.ENGLISH,
    region: ComplianceRegion = ComplianceRegion.GLOBAL,
    timezone: str = "UTC"
) -> GlobalSupportManager:
    """Create global support manager with common configuration."""
    
    localization_config = LocalizationConfig(
        language=language,
        timezone=timezone,
        region=region.value.upper()
    )
    
    compliance_config = ComplianceConfig(
        primary_region=region,
        encryption_enabled=CRYPTO_AVAILABLE,
        audit_logging=True,
        anonymization_enabled=True
    )
    
    return GlobalSupportManager(localization_config, compliance_config)


def create_eu_compliant_support() -> GlobalSupportManager:
    """Create EU GDPR compliant support manager."""
    
    localization_config = LocalizationConfig(
        language=SupportedLanguage.ENGLISH,
        region="EU",
        timezone="Europe/Brussels",
        date_format="%d/%m/%Y"  # European date format
    )
    
    compliance_config = ComplianceConfig(
        primary_region=ComplianceRegion.EU,
        data_retention_days=365,
        encryption_enabled=True,
        anonymization_enabled=True,
        audit_logging=True,
        consent_management=True,
        data_minimization=True,
        purpose_limitation=True,
        storage_limitation=True
    )
    
    return GlobalSupportManager(localization_config, compliance_config)


def create_us_compliant_support() -> GlobalSupportManager:
    """Create US CCPA compliant support manager."""
    
    localization_config = LocalizationConfig(
        language=SupportedLanguage.ENGLISH,
        region="US", 
        timezone="America/New_York",
        date_format="%m/%d/%Y"  # US date format
    )
    
    compliance_config = ComplianceConfig(
        primary_region=ComplianceRegion.US,
        data_retention_days=365,
        encryption_enabled=True,
        anonymization_enabled=True,
        audit_logging=True
    )
    
    return GlobalSupportManager(localization_config, compliance_config)


# Global instance for convenience
_global_support_manager: Optional[GlobalSupportManager] = None


def get_global_support() -> GlobalSupportManager:
    """Get or create global support manager instance."""
    global _global_support_manager
    
    if _global_support_manager is None:
        _global_support_manager = create_global_support()
    
    return _global_support_manager


def set_global_support(manager: GlobalSupportManager) -> None:
    """Set global support manager instance."""
    global _global_support_manager
    _global_support_manager = manager


# Convenience translation function
def _(key: str, **kwargs) -> str:
    """Shorthand translation function."""
    return get_global_support().translate(key, **kwargs)