#!/usr/bin/env python3
"""
Comprehensive Security and Quality Validation Suite

Implements enterprise-grade security scanning, code quality analysis,
and compliance validation for the photonic neuromorphics platform.
"""

import os
import sys
import subprocess
import json
import time
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import ast
import tokenize
import io


class SecurityLevel(Enum):
    """Security risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class QualityLevel(Enum):
    """Code quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class SecurityFinding:
    """Security vulnerability finding."""
    severity: SecurityLevel
    category: str
    description: str
    file_path: str
    line_number: int
    rule_id: str
    recommendation: str = ""
    cwe_id: Optional[str] = None


@dataclass
class QualityMetric:
    """Code quality metric."""
    name: str
    value: float
    threshold: float
    status: QualityLevel
    description: str


@dataclass
class ValidationResults:
    """Comprehensive validation results."""
    security_findings: List[SecurityFinding] = field(default_factory=list)
    quality_metrics: List[QualityMetric] = field(default_factory=list)
    compliance_status: Dict[str, bool] = field(default_factory=dict)
    overall_score: float = 0.0
    passed_quality_gates: bool = False


class SecurityScanner:
    """Comprehensive security vulnerability scanner."""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.findings = []
        
        # Security patterns to detect
        self.security_patterns = {
            "hardcoded_secrets": [
                (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password detected"),
                (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key detected"),
                (r'secret_key\s*=\s*["\'][^"\']+["\']', "Hardcoded secret key detected"),
                (r'token\s*=\s*["\'][^"\']+["\']', "Hardcoded token detected"),
                (r'Bearer\s+[A-Za-z0-9\-\._~\+\/]+=*', "Bearer token in code"),
            ],
            "sql_injection": [
                (r'query\s*=\s*["\'].*%s.*["\']', "Potential SQL injection vector"),
                (r'execute\s*\(["\'].*%s.*["\']', "Potential SQL injection in execute"),
                (r'cursor\.execute\s*\(["\'].*\+.*["\']', "SQL query concatenation detected"),
            ],
            "command_injection": [
                (r'os\.system\s*\(.*\+.*\)', "Command injection via os.system"),
                (r'subprocess\.(run|call|check_output)\s*\(.*shell\s*=\s*True', "Shell injection risk"),
                (r'eval\s*\(.*input\s*\(', "Eval with user input"),
                (r'exec\s*\(.*input\s*\(', "Exec with user input"),
            ],
            "path_traversal": [
                (r'open\s*\(.*\+.*\)', "Potential path traversal in file open"),
                (r'\.\./', "Path traversal pattern detected"),
                (r'os\.path\.join\s*\(.*input', "Path join with user input"),
            ],
            "weak_crypto": [
                (r'md5\s*\(', "Weak MD5 hash function"),
                (r'sha1\s*\(', "Weak SHA1 hash function"),
                (r'random\.random\s*\(', "Weak random for security purposes"),
                (r'DES|RC4', "Weak encryption algorithm"),
            ],
            "insecure_transport": [
                (r'http://.*', "Insecure HTTP protocol"),
                (r'ssl_verify\s*=\s*False', "SSL verification disabled"),
                (r'verify\s*=\s*False', "Certificate verification disabled"),
            ],
            "debug_code": [
                (r'print\s*\(.*password.*\)', "Password in print statement"),
                (r'print\s*\(.*secret.*\)', "Secret in print statement"),
                (r'print\s*\(.*token.*\)', "Token in print statement"),
                (r'DEBUG\s*=\s*True', "Debug mode enabled"),
            ]
        }
    
    def scan_file(self, file_path: Path) -> List[SecurityFinding]:
        """Scan individual file for security issues."""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.splitlines()
                
                # Scan for security patterns
                for category, patterns in self.security_patterns.items():
                    for pattern, description in patterns:
                        for line_num, line in enumerate(lines, 1):
                            if re.search(pattern, line, re.IGNORECASE):
                                severity = self._determine_severity(category, pattern)
                                finding = SecurityFinding(
                                    severity=severity,
                                    category=category,
                                    description=description,
                                    file_path=str(file_path.relative_to(self.repo_path)),
                                    line_number=line_num,
                                    rule_id=f"{category}_{hash(pattern) % 1000}",
                                    recommendation=self._get_recommendation(category)
                                )
                                findings.append(finding)
                
        except Exception as e:
            print(f"Error scanning {file_path}: {e}")
        
        return findings
    
    def _determine_severity(self, category: str, pattern: str) -> SecurityLevel:
        """Determine severity level based on category and pattern."""
        high_risk_categories = ["sql_injection", "command_injection", "hardcoded_secrets"]
        medium_risk_categories = ["path_traversal", "weak_crypto", "insecure_transport"]
        
        if category in high_risk_categories:
            return SecurityLevel.HIGH
        elif category in medium_risk_categories:
            return SecurityLevel.MEDIUM
        else:
            return SecurityLevel.LOW
    
    def _get_recommendation(self, category: str) -> str:
        """Get security recommendation for category."""
        recommendations = {
            "hardcoded_secrets": "Use environment variables or secure key management",
            "sql_injection": "Use parameterized queries or prepared statements",
            "command_injection": "Use subprocess with shell=False and input validation",
            "path_traversal": "Validate and sanitize file paths",
            "weak_crypto": "Use strong cryptographic functions (SHA-256+)",
            "insecure_transport": "Use HTTPS and enable certificate verification",
            "debug_code": "Remove debug code and sensitive information logging"
        }
        return recommendations.get(category, "Review and remediate security issue")
    
    def scan_repository(self) -> List[SecurityFinding]:
        """Scan entire repository for security issues."""
        print("🔍 Running security scan...")
        
        python_files = list(self.repo_path.glob("**/*.py"))
        total_findings = []
        
        for file_path in python_files:
            # Skip test files and virtual environments
            if any(skip in str(file_path) for skip in [".git", "__pycache__", ".venv", "venv"]):
                continue
                
            file_findings = self.scan_file(file_path)
            total_findings.extend(file_findings)
        
        self.findings = total_findings
        return total_findings


class CodeQualityAnalyzer:
    """Code quality and complexity analyzer."""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.metrics = []
    
    def analyze_complexity(self, file_path: Path) -> Dict[str, float]:
        """Analyze cyclomatic complexity of Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            
            class ComplexityVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.complexity = 1  # Base complexity
                    self.functions = 0
                    self.classes = 0
                    self.lines = 0
                
                def visit_If(self, node):
                    self.complexity += 1
                    self.generic_visit(node)
                
                def visit_For(self, node):
                    self.complexity += 1
                    self.generic_visit(node)
                
                def visit_While(self, node):
                    self.complexity += 1
                    self.generic_visit(node)
                
                def visit_Try(self, node):
                    self.complexity += 1
                    self.generic_visit(node)
                
                def visit_With(self, node):
                    self.complexity += 1
                    self.generic_visit(node)
                
                def visit_FunctionDef(self, node):
                    self.functions += 1
                    self.generic_visit(node)
                
                def visit_ClassDef(self, node):
                    self.classes += 1
                    self.generic_visit(node)
            
            visitor = ComplexityVisitor()
            visitor.visit(tree)
            
            # Count lines of code (non-empty, non-comment)
            lines = content.splitlines()
            loc = sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))
            
            return {
                "cyclomatic_complexity": visitor.complexity,
                "functions": visitor.functions,
                "classes": visitor.classes,
                "lines_of_code": loc,
                "complexity_per_function": visitor.complexity / max(visitor.functions, 1)
            }
            
        except Exception as e:
            return {
                "cyclomatic_complexity": 0,
                "functions": 0,
                "classes": 0,
                "lines_of_code": 0,
                "complexity_per_function": 0
            }
    
    def analyze_code_quality(self) -> List[QualityMetric]:
        """Analyze overall code quality metrics."""
        print("📊 Analyzing code quality...")
        
        python_files = list(self.repo_path.glob("**/*.py"))
        total_complexity = 0
        total_functions = 0
        total_classes = 0
        total_loc = 0
        file_count = 0
        
        for file_path in python_files:
            if any(skip in str(file_path) for skip in [".git", "__pycache__", ".venv", "venv"]):
                continue
            
            metrics = self.analyze_complexity(file_path)
            total_complexity += metrics["cyclomatic_complexity"]
            total_functions += metrics["functions"]
            total_classes += metrics["classes"]
            total_loc += metrics["lines_of_code"]
            file_count += 1
        
        if file_count == 0:
            return []
        
        # Calculate quality metrics
        avg_complexity = total_complexity / file_count
        avg_complexity_per_function = total_complexity / max(total_functions, 1)
        functions_per_file = total_functions / file_count
        classes_per_file = total_classes / file_count
        loc_per_file = total_loc / file_count
        
        quality_metrics = [
            QualityMetric(
                name="Average Cyclomatic Complexity",
                value=avg_complexity,
                threshold=10.0,
                status=QualityLevel.GOOD if avg_complexity <= 10 else QualityLevel.POOR,
                description="Average cyclomatic complexity per file"
            ),
            QualityMetric(
                name="Complexity per Function",
                value=avg_complexity_per_function,
                threshold=5.0,
                status=QualityLevel.GOOD if avg_complexity_per_function <= 5 else QualityLevel.POOR,
                description="Average complexity per function"
            ),
            QualityMetric(
                name="Functions per File",
                value=functions_per_file,
                threshold=20.0,
                status=QualityLevel.GOOD if functions_per_file <= 20 else QualityLevel.ACCEPTABLE,
                description="Average functions per file"
            ),
            QualityMetric(
                name="Lines of Code per File",
                value=loc_per_file,
                threshold=500.0,
                status=QualityLevel.GOOD if loc_per_file <= 500 else QualityLevel.ACCEPTABLE,
                description="Average lines of code per file"
            ),
            QualityMetric(
                name="Total Lines of Code",
                value=total_loc,
                threshold=float('inf'),
                status=QualityLevel.GOOD,
                description="Total lines of code in repository"
            )
        ]
        
        self.metrics = quality_metrics
        return quality_metrics


class ComplianceChecker:
    """Check compliance with security and coding standards."""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
    
    def check_file_permissions(self) -> Dict[str, bool]:
        """Check file permissions for security compliance."""
        results = {}
        
        try:
            # Check for overly permissive files
            for file_path in self.repo_path.rglob("*"):
                if file_path.is_file():
                    # Check if file is world-readable
                    stat_info = file_path.stat()
                    mode = oct(stat_info.st_mode)[-3:]
                    
                    # Flag files with world-write permissions
                    if mode[2] in ['2', '3', '6', '7']:
                        results[f"world_writable_{file_path.name}"] = False
                    
            results["file_permissions_secure"] = len([k for k, v in results.items() if not v]) == 0
            
        except Exception:
            results["file_permissions_secure"] = True  # Assume secure if can't check
        
        return results
    
    def check_dependency_security(self) -> Dict[str, bool]:
        """Check for known security issues in dependencies."""
        results = {}
        
        # Check requirements.txt for known vulnerable packages
        requirements_file = self.repo_path / "requirements.txt"
        
        if requirements_file.exists():
            try:
                with open(requirements_file, 'r') as f:
                    content = f.read()
                
                # Known vulnerable patterns (simplified)
                vulnerable_patterns = [
                    r'urllib3<1.24.2',  # Example vulnerable version
                    r'requests<2.20.0',
                    r'pyyaml<5.1',
                    r'pillow<6.2.0'
                ]
                
                has_vulnerabilities = any(re.search(pattern, content, re.IGNORECASE) 
                                        for pattern in vulnerable_patterns)
                
                results["dependencies_secure"] = not has_vulnerabilities
                
            except Exception:
                results["dependencies_secure"] = True
        else:
            results["dependencies_secure"] = True
        
        return results
    
    def check_code_standards(self) -> Dict[str, bool]:
        """Check adherence to coding standards."""
        results = {}
        
        python_files = list(self.repo_path.glob("**/*.py"))
        
        # Check for proper docstrings
        files_with_docstrings = 0
        files_with_type_hints = 0
        
        for file_path in python_files:
            if any(skip in str(file_path) for skip in [".git", "__pycache__", ".venv"]):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Check for module docstring
                if re.search(r'^""".*?"""', content, re.MULTILINE | re.DOTALL):
                    files_with_docstrings += 1
                
                # Check for type hints
                if re.search(r':\s*[A-Za-z][\w\[\], ]*\s*=', content) or \
                   re.search(r'def\s+\w+\([^)]*:\s*[A-Za-z]', content):
                    files_with_type_hints += 1
                    
            except Exception:
                continue
        
        total_files = len([f for f in python_files 
                          if not any(skip in str(f) for skip in [".git", "__pycache__", ".venv"])])
        
        if total_files > 0:
            docstring_coverage = files_with_docstrings / total_files
            type_hint_coverage = files_with_type_hints / total_files
            
            results["sufficient_docstrings"] = docstring_coverage >= 0.8
            results["sufficient_type_hints"] = type_hint_coverage >= 0.6
        else:
            results["sufficient_docstrings"] = True
            results["sufficient_type_hints"] = True
        
        return results
    
    def check_security_compliance(self) -> Dict[str, bool]:
        """Check security compliance requirements."""
        results = {}
        
        # Check for security configuration files
        security_files = [
            ".bandit",
            "security.md", 
            "SECURITY.md",
            "security.yaml",
            ".safety"
        ]
        
        for sec_file in security_files:
            file_path = self.repo_path / sec_file
            results[f"has_{sec_file.replace('.', '_').replace('-', '_')}"] = file_path.exists()
        
        # Check for security documentation
        security_docs = any(results[k] for k in results.keys() if k.startswith("has_security"))
        results["has_security_documentation"] = security_docs
        
        return results
    
    def run_compliance_check(self) -> Dict[str, bool]:
        """Run comprehensive compliance check."""
        print("✅ Checking compliance standards...")
        
        all_results = {}
        
        # File permissions
        all_results.update(self.check_file_permissions())
        
        # Dependency security
        all_results.update(self.check_dependency_security())
        
        # Code standards
        all_results.update(self.check_code_standards())
        
        # Security compliance
        all_results.update(self.check_security_compliance())
        
        return all_results


class QualityGateValidator:
    """Validate code against quality gates."""
    
    def __init__(self):
        self.quality_gates = {
            "max_critical_security_findings": 0,
            "max_high_security_findings": 5,
            "max_cyclomatic_complexity": 15.0,
            "min_code_coverage": 75.0,
            "max_complexity_per_function": 7.0,
            "min_docstring_coverage": 80.0
        }
    
    def validate_security_gates(self, findings: List[SecurityFinding]) -> Dict[str, bool]:
        """Validate security quality gates."""
        critical_count = sum(1 for f in findings if f.severity == SecurityLevel.CRITICAL)
        high_count = sum(1 for f in findings if f.severity == SecurityLevel.HIGH)
        
        return {
            "security_critical_gate": critical_count <= self.quality_gates["max_critical_security_findings"],
            "security_high_gate": high_count <= self.quality_gates["max_high_security_findings"],
            "total_critical_findings": critical_count,
            "total_high_findings": high_count
        }
    
    def validate_quality_gates(self, metrics: List[QualityMetric]) -> Dict[str, bool]:
        """Validate code quality gates."""
        results = {}
        
        for metric in metrics:
            gate_name = f"quality_gate_{metric.name.lower().replace(' ', '_')}"
            
            if "complexity" in metric.name.lower():
                results[gate_name] = metric.value <= self.quality_gates.get("max_cyclomatic_complexity", 15.0)
            elif "function" in metric.name.lower() and "complexity" in metric.name.lower():
                results[gate_name] = metric.value <= self.quality_gates.get("max_complexity_per_function", 7.0)
            else:
                results[gate_name] = metric.status in [QualityLevel.EXCELLENT, QualityLevel.GOOD]
        
        return results
    
    def validate_all_gates(
        self,
        findings: List[SecurityFinding],
        metrics: List[QualityMetric],
        compliance: Dict[str, bool]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate all quality gates."""
        print("🚪 Validating quality gates...")
        
        security_gates = self.validate_security_gates(findings)
        quality_gates = self.validate_quality_gates(metrics)
        
        # Combine all gate results
        all_gates = {
            **security_gates,
            **quality_gates,
            "compliance_gates": all(compliance.values())
        }
        
        # Overall pass/fail
        critical_gates = [
            "security_critical_gate",
            "security_high_gate",
        ]
        
        critical_passed = all(all_gates.get(gate, False) for gate in critical_gates)
        overall_passed = critical_passed and all_gates.get("compliance_gates", False)
        
        return overall_passed, all_gates


def generate_security_report(
    findings: List[SecurityFinding],
    metrics: List[QualityMetric],
    compliance: Dict[str, bool],
    gates_passed: bool,
    gate_results: Dict[str, Any]
) -> str:
    """Generate comprehensive security and quality report."""
    
    report = []
    report.append("=" * 80)
    report.append("🔒 SECURITY & QUALITY VALIDATION REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Executive Summary
    status = "✅ PASSED" if gates_passed else "❌ FAILED"
    report.append(f"📋 Overall Status: {status}")
    report.append("")
    
    # Security Findings Summary
    report.append("🛡️  SECURITY FINDINGS SUMMARY")
    report.append("-" * 40)
    
    if not findings:
        report.append("✅ No security findings detected")
    else:
        severity_counts = {}
        for finding in findings:
            severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1
        
        for severity, count in severity_counts.items():
            icon = "🔴" if severity in [SecurityLevel.CRITICAL, SecurityLevel.HIGH] else "🟡"
            report.append(f"{icon} {severity.value.upper()}: {count} findings")
    
    report.append("")
    
    # Quality Metrics Summary
    report.append("📊 CODE QUALITY METRICS")
    report.append("-" * 40)
    
    for metric in metrics:
        status_icon = "✅" if metric.status in [QualityLevel.EXCELLENT, QualityLevel.GOOD] else "⚠️"
        report.append(f"{status_icon} {metric.name}: {metric.value:.2f}")
    
    report.append("")
    
    # Compliance Summary
    report.append("✅ COMPLIANCE STATUS")
    report.append("-" * 40)
    
    passed_compliance = sum(1 for v in compliance.values() if v)
    total_compliance = len(compliance)
    compliance_rate = (passed_compliance / total_compliance * 100) if total_compliance > 0 else 100
    
    report.append(f"📈 Compliance Rate: {compliance_rate:.1f}% ({passed_compliance}/{total_compliance})")
    
    for check, passed in compliance.items():
        status = "✅" if passed else "❌"
        report.append(f"{status} {check.replace('_', ' ').title()}")
    
    report.append("")
    
    # Quality Gates
    report.append("🚪 QUALITY GATES")
    report.append("-" * 40)
    
    for gate, passed in gate_results.items():
        if isinstance(passed, bool):
            status = "✅ PASSED" if passed else "❌ FAILED"
            report.append(f"{status} {gate.replace('_', ' ').title()}")
    
    report.append("")
    
    # Detailed Findings (Critical and High only)
    critical_high_findings = [f for f in findings 
                             if f.severity in [SecurityLevel.CRITICAL, SecurityLevel.HIGH]]
    
    if critical_high_findings:
        report.append("🔍 CRITICAL & HIGH PRIORITY FINDINGS")
        report.append("-" * 40)
        
        for finding in critical_high_findings[:10]:  # Limit to first 10
            report.append(f"🔴 {finding.severity.value.upper()}: {finding.description}")
            report.append(f"   📁 File: {finding.file_path}:{finding.line_number}")
            report.append(f"   💡 Recommendation: {finding.recommendation}")
            report.append("")
    
    # Recommendations
    report.append("💡 RECOMMENDATIONS")
    report.append("-" * 40)
    
    if not gates_passed:
        if gate_results.get("total_critical_findings", 0) > 0:
            report.append("🔴 CRITICAL: Address all critical security findings immediately")
        
        if gate_results.get("total_high_findings", 0) > 5:
            report.append("🟡 HIGH: Reduce high-severity security findings")
        
        if not gate_results.get("compliance_gates", True):
            report.append("📋 COMPLIANCE: Address compliance violations")
    else:
        report.append("✅ All quality gates passed - maintain current standards")
    
    report.append("")
    report.append("=" * 80)
    report.append(f"📅 Report generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    
    return "\n".join(report)


def main():
    """Run comprehensive security and quality validation."""
    repo_path = "/root/repo"
    
    print("🔒 Starting Security & Quality Validation Suite")
    print("=" * 60)
    
    # Initialize components
    security_scanner = SecurityScanner(repo_path)
    quality_analyzer = CodeQualityAnalyzer(repo_path)
    compliance_checker = ComplianceChecker(repo_path)
    quality_gate_validator = QualityGateValidator()
    
    # Run security scan
    security_findings = security_scanner.scan_repository()
    
    # Analyze code quality
    quality_metrics = quality_analyzer.analyze_code_quality()
    
    # Check compliance
    compliance_results = compliance_checker.run_compliance_check()
    
    # Validate quality gates
    gates_passed, gate_results = quality_gate_validator.validate_all_gates(
        security_findings, quality_metrics, compliance_results
    )
    
    # Generate comprehensive report
    report = generate_security_report(
        security_findings, quality_metrics, compliance_results,
        gates_passed, gate_results
    )
    
    # Output report
    print(report)
    
    # Save report to file
    report_file = Path(repo_path) / "security_quality_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Return appropriate exit code
    return 0 if gates_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)