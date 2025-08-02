#!/usr/bin/env python3
"""
Repository health check script for photonic neuromorphics simulation platform.
Performs comprehensive health assessment of the repository.
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple

import requests


class RepositoryHealthChecker:
    """Comprehensive repository health assessment."""
    
    def __init__(self, repo_path: str = "."):
        """Initialize health checker."""
        self.repo_path = Path(repo_path)
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo_name = os.getenv("GITHUB_REPOSITORY", "danieleschmidt/photonic-neuromorphics-sim")
        self.health_score = 0
        self.max_score = 0
        self.issues = []
        self.recommendations = []
        
    def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check."""
        print("Starting repository health check...")
        
        health_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "repository": self.repo_name,
            "checks": {},
            "summary": {}
        }
        
        # Run all health checks
        health_report["checks"]["code_quality"] = self.check_code_quality()
        health_report["checks"]["security"] = self.check_security()
        health_report["checks"]["documentation"] = self.check_documentation()
        health_report["checks"]["testing"] = self.check_testing()
        health_report["checks"]["ci_cd"] = self.check_ci_cd()
        health_report["checks"]["dependencies"] = self.check_dependencies()
        health_report["checks"]["repository_structure"] = self.check_repository_structure()
        health_report["checks"]["performance"] = self.check_performance()
        health_report["checks"]["compliance"] = self.check_compliance()
        
        # Calculate overall health score
        health_report["summary"] = self._calculate_summary()
        
        return health_report
    
    def check_code_quality(self) -> Dict[str, Any]:
        """Check code quality metrics."""
        print("Checking code quality...")
        
        checks = {}
        category_score = 0
        category_max = 0
        
        # Test coverage
        coverage_score, coverage_max = self._check_test_coverage()
        checks["test_coverage"] = {
            "score": coverage_score,
            "max_score": coverage_max,
            "status": "good" if coverage_score >= coverage_max * 0.8 else "warning" if coverage_score >= coverage_max * 0.6 else "poor"
        }
        category_score += coverage_score
        category_max += coverage_max
        
        # Code complexity
        complexity_score, complexity_max = self._check_code_complexity()
        checks["code_complexity"] = {
            "score": complexity_score,
            "max_score": complexity_max,
            "status": "good" if complexity_score >= complexity_max * 0.8 else "warning" if complexity_score >= complexity_max * 0.6 else "poor"
        }
        category_score += complexity_score
        category_max += complexity_max
        
        # Code style
        style_score, style_max = self._check_code_style()
        checks["code_style"] = {
            "score": style_score,
            "max_score": style_max,
            "status": "good" if style_score >= style_max * 0.8 else "warning" if style_score >= style_max * 0.6 else "poor"
        }
        category_score += style_score
        category_max += style_max
        
        # Documentation strings
        docstring_score, docstring_max = self._check_docstrings()
        checks["docstrings"] = {
            "score": docstring_score,
            "max_score": docstring_max,
            "status": "good" if docstring_score >= docstring_max * 0.8 else "warning" if docstring_score >= docstring_max * 0.6 else "poor"
        }
        category_score += docstring_score
        category_max += docstring_max
        
        self.health_score += category_score
        self.max_score += category_max
        
        return {
            "checks": checks,
            "overall_score": category_score,
            "max_score": category_max,
            "percentage": round((category_score / category_max) * 100, 1) if category_max > 0 else 0
        }
    
    def check_security(self) -> Dict[str, Any]:
        """Check security posture."""
        print("Checking security...")
        
        checks = {}
        category_score = 0
        category_max = 0
        
        # Vulnerability scan
        vuln_score, vuln_max = self._check_vulnerabilities()
        checks["vulnerabilities"] = {
            "score": vuln_score,
            "max_score": vuln_max,
            "status": "good" if vuln_score >= vuln_max * 0.9 else "warning" if vuln_score >= vuln_max * 0.7 else "poor"
        }
        category_score += vuln_score
        category_max += vuln_max
        
        # Secrets detection
        secrets_score, secrets_max = self._check_secrets()
        checks["secrets"] = {
            "score": secrets_score,
            "max_score": secrets_max,
            "status": "good" if secrets_score == secrets_max else "poor"
        }
        category_score += secrets_score
        category_max += secrets_max
        
        # Security configuration
        config_score, config_max = self._check_security_config()
        checks["security_config"] = {
            "score": config_score,
            "max_score": config_max,
            "status": "good" if config_score >= config_max * 0.8 else "warning" if config_score >= config_max * 0.6 else "poor"
        }
        category_score += config_score
        category_max += config_max
        
        self.health_score += category_score
        self.max_score += category_max
        
        return {
            "checks": checks,
            "overall_score": category_score,
            "max_score": category_max,
            "percentage": round((category_score / category_max) * 100, 1) if category_max > 0 else 0
        }
    
    def check_documentation(self) -> Dict[str, Any]:
        """Check documentation quality and completeness."""
        print("Checking documentation...")
        
        checks = {}
        category_score = 0
        category_max = 0
        
        # README quality
        readme_score, readme_max = self._check_readme()
        checks["readme"] = {
            "score": readme_score,
            "max_score": readme_max,
            "status": "good" if readme_score >= readme_max * 0.8 else "warning" if readme_score >= readme_max * 0.6 else "poor"
        }
        category_score += readme_score
        category_max += readme_max
        
        # Documentation completeness
        docs_score, docs_max = self._check_documentation_completeness()
        checks["completeness"] = {
            "score": docs_score,
            "max_score": docs_max,
            "status": "good" if docs_score >= docs_max * 0.8 else "warning" if docs_score >= docs_max * 0.6 else "poor"
        }
        category_score += docs_score
        category_max += docs_max
        
        # API documentation
        api_docs_score, api_docs_max = self._check_api_documentation()
        checks["api_docs"] = {
            "score": api_docs_score,
            "max_score": api_docs_max,
            "status": "good" if api_docs_score >= api_docs_max * 0.8 else "warning" if api_docs_score >= api_docs_max * 0.6 else "poor"
        }
        category_score += api_docs_score
        category_max += api_docs_max
        
        self.health_score += category_score
        self.max_score += category_max
        
        return {
            "checks": checks,
            "overall_score": category_score,
            "max_score": category_max,
            "percentage": round((category_score / category_max) * 100, 1) if category_max > 0 else 0
        }
    
    def check_testing(self) -> Dict[str, Any]:
        """Check testing infrastructure and quality."""
        print("Checking testing...")
        
        checks = {}
        category_score = 0
        category_max = 0
        
        # Test structure
        structure_score, structure_max = self._check_test_structure()
        checks["test_structure"] = {
            "score": structure_score,
            "max_score": structure_max,
            "status": "good" if structure_score >= structure_max * 0.8 else "warning" if structure_score >= structure_max * 0.6 else "poor"
        }
        category_score += structure_score
        category_max += structure_max
        
        # Test quality
        quality_score, quality_max = self._check_test_quality()
        checks["test_quality"] = {
            "score": quality_score,
            "max_score": quality_max,
            "status": "good" if quality_score >= quality_max * 0.8 else "warning" if quality_score >= quality_max * 0.6 else "poor"
        }
        category_score += quality_score
        category_max += quality_max
        
        self.health_score += category_score
        self.max_score += category_max
        
        return {
            "checks": checks,
            "overall_score": category_score,
            "max_score": category_max,
            "percentage": round((category_score / category_max) * 100, 1) if category_max > 0 else 0
        }
    
    def check_ci_cd(self) -> Dict[str, Any]:
        """Check CI/CD pipeline health."""
        print("Checking CI/CD...")
        
        checks = {}
        category_score = 0
        category_max = 0
        
        # Workflow presence
        workflow_score, workflow_max = self._check_workflows()
        checks["workflows"] = {
            "score": workflow_score,
            "max_score": workflow_max,
            "status": "good" if workflow_score >= workflow_max * 0.8 else "warning" if workflow_score >= workflow_max * 0.6 else "poor"
        }
        category_score += workflow_score
        category_max += workflow_max
        
        # Build status
        if self.github_token:
            build_score, build_max = self._check_build_status()
            checks["build_status"] = {
                "score": build_score,
                "max_score": build_max,
                "status": "good" if build_score >= build_max * 0.8 else "warning" if build_score >= build_max * 0.6 else "poor"
            }
            category_score += build_score
            category_max += build_max
        
        self.health_score += category_score
        self.max_score += category_max
        
        return {
            "checks": checks,
            "overall_score": category_score,
            "max_score": category_max,
            "percentage": round((category_score / category_max) * 100, 1) if category_max > 0 else 0
        }
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check dependency health."""
        print("Checking dependencies...")
        
        checks = {}
        category_score = 0
        category_max = 0
        
        # Dependency freshness
        freshness_score, freshness_max = self._check_dependency_freshness()
        checks["freshness"] = {
            "score": freshness_score,
            "max_score": freshness_max,
            "status": "good" if freshness_score >= freshness_max * 0.8 else "warning" if freshness_score >= freshness_max * 0.6 else "poor"
        }
        category_score += freshness_score
        category_max += freshness_max
        
        # License compatibility
        license_score, license_max = self._check_license_compatibility()
        checks["licenses"] = {
            "score": license_score,
            "max_score": license_max,
            "status": "good" if license_score >= license_max * 0.8 else "warning" if license_score >= license_max * 0.6 else "poor"
        }
        category_score += license_score
        category_max += license_max
        
        self.health_score += category_score
        self.max_score += category_max
        
        return {
            "checks": checks,
            "overall_score": category_score,
            "max_score": category_max,
            "percentage": round((category_score / category_max) * 100, 1) if category_max > 0 else 0
        }
    
    def check_repository_structure(self) -> Dict[str, Any]:
        """Check repository structure and organization."""
        print("Checking repository structure...")
        
        checks = {}
        category_score = 0
        category_max = 0
        
        # Essential files
        files_score, files_max = self._check_essential_files()
        checks["essential_files"] = {
            "score": files_score,
            "max_score": files_max,
            "status": "good" if files_score >= files_max * 0.8 else "warning" if files_score >= files_max * 0.6 else "poor"
        }
        category_score += files_score
        category_max += files_max
        
        # Directory structure
        structure_score, structure_max = self._check_directory_structure()
        checks["directory_structure"] = {
            "score": structure_score,
            "max_score": structure_max,
            "status": "good" if structure_score >= structure_max * 0.8 else "warning" if structure_score >= structure_max * 0.6 else "poor"
        }
        category_score += structure_score
        category_max += structure_max
        
        self.health_score += category_score
        self.max_score += category_max
        
        return {
            "checks": checks,
            "overall_score": category_score,
            "max_score": category_max,
            "percentage": round((category_score / category_max) * 100, 1) if category_max > 0 else 0
        }
    
    def check_performance(self) -> Dict[str, Any]:
        """Check performance-related aspects."""
        print("Checking performance...")
        
        checks = {}
        category_score = 0
        category_max = 0
        
        # Repository size
        size_score, size_max = self._check_repository_size()
        checks["repository_size"] = {
            "score": size_score,
            "max_score": size_max,
            "status": "good" if size_score >= size_max * 0.8 else "warning" if size_score >= size_max * 0.6 else "poor"
        }
        category_score += size_score
        category_max += size_max
        
        # Large files
        large_files_score, large_files_max = self._check_large_files()
        checks["large_files"] = {
            "score": large_files_score,
            "max_score": large_files_max,
            "status": "good" if large_files_score >= large_files_max * 0.8 else "warning" if large_files_score >= large_files_max * 0.6 else "poor"
        }
        category_score += large_files_score
        category_max += large_files_max
        
        self.health_score += category_score
        self.max_score += category_max
        
        return {
            "checks": checks,
            "overall_score": category_score,
            "max_score": category_max,
            "percentage": round((category_score / category_max) * 100, 1) if category_max > 0 else 0
        }
    
    def check_compliance(self) -> Dict[str, Any]:
        """Check compliance with best practices."""
        print("Checking compliance...")
        
        checks = {}
        category_score = 0
        category_max = 0
        
        # Git practices
        git_score, git_max = self._check_git_practices()
        checks["git_practices"] = {
            "score": git_score,
            "max_score": git_max,
            "status": "good" if git_score >= git_max * 0.8 else "warning" if git_score >= git_max * 0.6 else "poor"
        }
        category_score += git_score
        category_max += git_max
        
        # Branch protection
        if self.github_token:
            protection_score, protection_max = self._check_branch_protection()
            checks["branch_protection"] = {
                "score": protection_score,
                "max_score": protection_max,
                "status": "good" if protection_score >= protection_max * 0.8 else "warning" if protection_score >= protection_max * 0.6 else "poor"
            }
            category_score += protection_score
            category_max += protection_max
        
        self.health_score += category_score
        self.max_score += category_max
        
        return {
            "checks": checks,
            "overall_score": category_score,
            "max_score": category_max,
            "percentage": round((category_score / category_max) * 100, 1) if category_max > 0 else 0
        }
    
    def _check_test_coverage(self) -> Tuple[int, int]:
        """Check test coverage."""
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "--cov=src", "--cov-report=json", "--tb=no"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            if result.returncode == 0 and (self.repo_path / "coverage.json").exists():
                with open(self.repo_path / "coverage.json", 'r') as f:
                    coverage_data = json.load(f)
                    coverage = coverage_data["totals"]["percent_covered"]
                    
                if coverage >= 90:
                    return 10, 10
                elif coverage >= 80:
                    return 8, 10
                elif coverage >= 70:
                    return 6, 10
                elif coverage >= 50:
                    return 4, 10
                else:
                    return 2, 10
            else:
                self.issues.append("Test coverage could not be measured")
                return 0, 10
        except Exception:
            self.issues.append("Test coverage check failed")
            return 0, 10
    
    def _check_code_complexity(self) -> Tuple[int, int]:
        """Check code complexity."""
        try:
            result = subprocess.run(
                ["radon", "cc", "src/", "--average"],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                # Parse average complexity
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines:
                    if "Average complexity" in line:
                        complexity_match = re.search(r'(\d+\.\d+)', line)
                        if complexity_match:
                            complexity = float(complexity_match.group(1))
                            if complexity <= 5:
                                return 10, 10
                            elif complexity <= 10:
                                return 7, 10
                            elif complexity <= 15:
                                return 4, 10
                            else:
                                return 1, 10
            
            return 5, 10  # Default if parsing fails
        except Exception:
            return 5, 10
    
    def _check_code_style(self) -> Tuple[int, int]:
        """Check code style compliance."""
        try:
            result = subprocess.run(
                ["flake8", "src/", "--count", "--statistics"],
                capture_output=True, text=True
            )
            
            error_count = 0
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip().isdigit():
                        error_count += int(line.strip())
            
            if error_count == 0:
                return 10, 10
            elif error_count <= 10:
                return 7, 10
            elif error_count <= 50:
                return 4, 10
            else:
                return 1, 10
                
        except Exception:
            return 5, 10
    
    def _check_docstrings(self) -> Tuple[int, int]:
        """Check docstring coverage."""
        try:
            python_files = list(self.repo_path.glob("src/**/*.py"))
            if not python_files:
                return 0, 10
            
            total_functions = 0
            documented_functions = 0
            
            for py_file in python_files:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Simple regex to find function definitions and docstrings
                functions = re.findall(r'def\s+\w+\s*\([^)]*\):', content)
                docstrings = re.findall(r'def\s+\w+\s*\([^)]*\):\s*"""', content)
                
                total_functions += len(functions)
                documented_functions += len(docstrings)
            
            if total_functions == 0:
                return 10, 10
            
            coverage = documented_functions / total_functions
            if coverage >= 0.9:
                return 10, 10
            elif coverage >= 0.7:
                return 7, 10
            elif coverage >= 0.5:
                return 4, 10
            else:
                return 1, 10
                
        except Exception:
            return 5, 10
    
    def _check_vulnerabilities(self) -> Tuple[int, int]:
        """Check for security vulnerabilities."""
        try:
            # Check with safety
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True, text=True
            )
            
            vulnerabilities = 0
            if result.stdout and result.stdout.strip():
                try:
                    safety_data = json.loads(result.stdout)
                    vulnerabilities += len(safety_data)
                except json.JSONDecodeError:
                    pass
            
            # Check with bandit
            result = subprocess.run(
                ["bandit", "-r", "src/", "-f", "json"],
                capture_output=True, text=True
            )
            
            if result.stdout:
                try:
                    bandit_data = json.loads(result.stdout)
                    high_severity = len([issue for issue in bandit_data.get("results", []) 
                                       if issue.get("issue_severity") in ["MEDIUM", "HIGH"]])
                    vulnerabilities += high_severity
                except json.JSONDecodeError:
                    pass
            
            if vulnerabilities == 0:
                return 10, 10
            elif vulnerabilities <= 2:
                return 6, 10
            elif vulnerabilities <= 5:
                return 3, 10
            else:
                return 0, 10
                
        except Exception:
            return 5, 10
    
    def _check_secrets(self) -> Tuple[int, int]:
        """Check for exposed secrets."""
        try:
            result = subprocess.run(
                ["detect-secrets", "scan", "--all-files"],
                capture_output=True, text=True
            )
            
            if result.stdout:
                try:
                    secrets_data = json.loads(result.stdout)
                    secrets_count = len(secrets_data.get("results", {}))
                    
                    if secrets_count == 0:
                        return 10, 10
                    else:
                        self.issues.append(f"Found {secrets_count} potential secrets")
                        return 0, 10
                except json.JSONDecodeError:
                    return 5, 10
            else:
                return 10, 10
                
        except Exception:
            return 5, 10
    
    def _check_security_config(self) -> Tuple[int, int]:
        """Check security configuration files."""
        score = 0
        max_score = 10
        
        security_files = [
            "SECURITY.md",
            ".github/dependabot.yml",
            ".gitignore"
        ]
        
        for file_path in security_files:
            if (self.repo_path / file_path).exists():
                score += 3
        
        # Check if gitignore is comprehensive
        gitignore_path = self.repo_path / ".gitignore"
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                gitignore_content = f.read()
                
            important_patterns = ["*.log", "*.env", "__pycache__", "node_modules"]
            for pattern in important_patterns:
                if pattern in gitignore_content:
                    score += 0.25
        
        return min(score, max_score), max_score
    
    def _check_readme(self) -> Tuple[int, int]:
        """Check README quality."""
        readme_path = self.repo_path / "README.md"
        if not readme_path.exists():
            self.issues.append("README.md is missing")
            return 0, 10
        
        with open(readme_path, 'r', encoding='utf-8') as f:
            readme_content = f.read()
        
        score = 0
        
        # Check for essential sections
        essential_sections = [
            r"#.*[Oo]verview",
            r"#.*[Ii]nstall",
            r"#.*[Uu]sage",
            r"#.*[Cc]ontribut",
            r"#.*[Ll]icense"
        ]
        
        for pattern in essential_sections:
            if re.search(pattern, readme_content, re.MULTILINE):
                score += 2
        
        return score, 10
    
    def _check_documentation_completeness(self) -> Tuple[int, int]:
        """Check documentation completeness."""
        docs_dir = self.repo_path / "docs"
        score = 0
        
        if docs_dir.exists():
            score += 5
            
            # Check for specific documentation types
            doc_types = ["api", "guides", "tutorials", "examples"]
            for doc_type in doc_types:
                if any(docs_dir.glob(f"**/*{doc_type}*")):
                    score += 1
        
        return min(score, 10), 10
    
    def _check_api_documentation(self) -> Tuple[int, int]:
        """Check API documentation."""
        # This would check for generated API docs, Sphinx docs, etc.
        # For now, simple check for docstrings
        return self._check_docstrings()
    
    def _check_test_structure(self) -> Tuple[int, int]:
        """Check test directory structure."""
        tests_dir = self.repo_path / "tests"
        if not tests_dir.exists():
            self.issues.append("Tests directory is missing")
            return 0, 10
        
        score = 5  # Base score for having tests directory
        
        # Check for different test types
        test_types = ["unit", "integration", "e2e", "performance"]
        for test_type in test_types:
            if (tests_dir / test_type).exists():
                score += 1
        
        # Check for test configuration
        test_configs = ["pytest.ini", "pyproject.toml", "tox.ini"]
        for config in test_configs:
            if (self.repo_path / config).exists():
                score += 1
                break
        
        return min(score, 10), 10
    
    def _check_test_quality(self) -> Tuple[int, int]:
        """Check test quality metrics."""
        try:
            # Count test files
            test_files = list(self.repo_path.glob("tests/**/test_*.py"))
            test_files.extend(list(self.repo_path.glob("tests/**/*_test.py")))
            
            if not test_files:
                return 0, 10
            
            # Basic quality check - at least 1 test per 100 lines of source code
            src_files = list(self.repo_path.glob("src/**/*.py"))
            total_lines = 0
            
            for src_file in src_files:
                with open(src_file, 'r', encoding='utf-8') as f:
                    total_lines += len(f.readlines())
            
            expected_tests = max(1, total_lines // 100)
            test_ratio = len(test_files) / expected_tests
            
            if test_ratio >= 1:
                return 10, 10
            elif test_ratio >= 0.7:
                return 7, 10
            elif test_ratio >= 0.4:
                return 4, 10
            else:
                return 2, 10
                
        except Exception:
            return 5, 10
    
    def _check_workflows(self) -> Tuple[int, int]:
        """Check CI/CD workflows."""
        workflows_dir = self.repo_path / ".github" / "workflows"
        if not workflows_dir.exists():
            self.issues.append("GitHub workflows directory is missing")
            return 0, 10
        
        workflow_files = list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml"))
        if not workflow_files:
            self.issues.append("No GitHub workflow files found")
            return 0, 10
        
        score = 5  # Base score for having workflows
        
        # Check for essential workflows
        essential_workflows = ["ci", "test", "build", "deploy", "security"]
        for workflow_file in workflow_files:
            workflow_name = workflow_file.name.lower()
            for essential in essential_workflows:
                if essential in workflow_name:
                    score += 1
                    break
        
        return min(score, 10), 10
    
    def _check_build_status(self) -> Tuple[int, int]:
        """Check recent build status via GitHub API."""
        try:
            if not self.github_token:
                return 5, 10
            
            headers = {"Authorization": f"token {self.github_token}"}
            url = f"https://api.github.com/repos/{self.repo_name}/actions/runs"
            
            response = requests.get(url, headers=headers, params={"per_page": 10})
            if response.status_code == 200:
                runs = response.json()["workflow_runs"]
                
                if not runs:
                    return 5, 10
                
                success_count = sum(1 for run in runs if run["conclusion"] == "success")
                success_rate = success_count / len(runs)
                
                if success_rate >= 0.9:
                    return 10, 10
                elif success_rate >= 0.7:
                    return 7, 10
                elif success_rate >= 0.5:
                    return 4, 10
                else:
                    return 1, 10
            else:
                return 5, 10
                
        except Exception:
            return 5, 10
    
    def _check_dependency_freshness(self) -> Tuple[int, int]:
        """Check how up-to-date dependencies are."""
        try:
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True, text=True
            )
            
            if result.returncode == 0 and result.stdout:
                outdated = json.loads(result.stdout)
                outdated_count = len(outdated)
                
                if outdated_count == 0:
                    return 10, 10
                elif outdated_count <= 5:
                    return 7, 10
                elif outdated_count <= 15:
                    return 4, 10
                else:
                    return 1, 10
            else:
                return 10, 10
                
        except Exception:
            return 5, 10
    
    def _check_license_compatibility(self) -> Tuple[int, int]:
        """Check license file and compatibility."""
        license_file = self.repo_path / "LICENSE"
        if license_file.exists():
            return 10, 10
        else:
            self.issues.append("LICENSE file is missing")
            return 0, 10
    
    def _check_essential_files(self) -> Tuple[int, int]:
        """Check for essential repository files."""
        essential_files = [
            "README.md",
            "LICENSE",
            ".gitignore",
            "requirements.txt",
            "setup.py"
        ]
        
        score = 0
        for file_path in essential_files:
            if (self.repo_path / file_path).exists():
                score += 2
        
        return score, 10
    
    def _check_directory_structure(self) -> Tuple[int, int]:
        """Check directory structure organization."""
        expected_dirs = ["src", "tests", "docs", ".github"]
        score = 0
        
        for directory in expected_dirs:
            if (self.repo_path / directory).exists():
                score += 2.5
        
        return min(score, 10), 10
    
    def _check_repository_size(self) -> Tuple[int, int]:
        """Check repository size."""
        try:
            result = subprocess.run(
                ["du", "-sh", "."],
                capture_output=True, text=True,
                cwd=self.repo_path
            )
            
            if result.returncode == 0:
                size_str = result.stdout.split()[0]
                
                # Parse size (simple heuristic)
                if size_str.endswith("K"):
                    score = 10
                elif size_str.endswith("M"):
                    size_mb = float(size_str[:-1])
                    if size_mb < 50:
                        score = 10
                    elif size_mb < 200:
                        score = 7
                    else:
                        score = 4
                elif size_str.endswith("G"):
                    score = 1
                else:
                    score = 10
                
                return score, 10
            else:
                return 5, 10
                
        except Exception:
            return 5, 10
    
    def _check_large_files(self) -> Tuple[int, int]:
        """Check for large files in repository."""
        try:
            result = subprocess.run(
                ["find", ".", "-type", "f", "-size", "+10M"],
                capture_output=True, text=True,
                cwd=self.repo_path
            )
            
            large_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
            large_files = [f for f in large_files if not f.startswith('./.git')]
            
            if not large_files:
                return 10, 10
            elif len(large_files) <= 2:
                return 6, 10
            else:
                self.issues.append(f"Found {len(large_files)} large files (>10MB)")
                return 2, 10
                
        except Exception:
            return 5, 10
    
    def _check_git_practices(self) -> Tuple[int, int]:
        """Check Git best practices."""
        score = 0
        
        try:
            # Check for signed commits
            result = subprocess.run(
                ["git", "log", "--show-signature", "-1"],
                capture_output=True, text=True,
                cwd=self.repo_path
            )
            
            if "gpg:" in result.stdout.lower():
                score += 3
            
            # Check commit message format
            result = subprocess.run(
                ["git", "log", "--oneline", "-10"],
                capture_output=True, text=True,
                cwd=self.repo_path
            )
            
            if result.returncode == 0:
                commit_messages = result.stdout.strip().split('\n')
                conventional_commits = 0
                
                for message in commit_messages:
                    # Check for conventional commit format
                    if re.match(r'^[a-f0-9]+ (feat|fix|docs|style|refactor|test|chore):', message):
                        conventional_commits += 1
                
                if conventional_commits >= len(commit_messages) * 0.8:
                    score += 4
                elif conventional_commits >= len(commit_messages) * 0.5:
                    score += 2
            
            # Check for .gitattributes
            if (self.repo_path / ".gitattributes").exists():
                score += 3
                
        except Exception:
            pass
        
        return min(score, 10), 10
    
    def _check_branch_protection(self) -> Tuple[int, int]:
        """Check branch protection settings via GitHub API."""
        try:
            if not self.github_token:
                return 5, 10
            
            headers = {"Authorization": f"token {self.github_token}"}
            url = f"https://api.github.com/repos/{self.repo_name}/branches/main/protection"
            
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                protection = response.json()
                score = 0
                
                if protection.get("required_status_checks", {}).get("strict"):
                    score += 3
                
                if protection.get("enforce_admins", {}).get("enabled"):
                    score += 2
                
                if protection.get("required_pull_request_reviews"):
                    score += 3
                
                if protection.get("restrictions"):
                    score += 2
                
                return min(score, 10), 10
            else:
                self.issues.append("Branch protection not configured")
                return 0, 10
                
        except Exception:
            return 5, 10
    
    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate overall health summary."""
        overall_percentage = round((self.health_score / self.max_score) * 100, 1) if self.max_score > 0 else 0
        
        if overall_percentage >= 90:
            grade = "A"
            status = "excellent"
        elif overall_percentage >= 80:
            grade = "B"
            status = "good"
        elif overall_percentage >= 70:
            grade = "C"
            status = "fair"
        elif overall_percentage >= 60:
            grade = "D"
            status = "poor"
        else:
            grade = "F"
            status = "critical"
        
        return {
            "overall_score": self.health_score,
            "max_score": self.max_score,
            "percentage": overall_percentage,
            "grade": grade,
            "status": status,
            "issues_count": len(self.issues),
            "recommendations_count": len(self.recommendations),
            "issues": self.issues,
            "recommendations": self.recommendations
        }


def main():
    """Main function to run repository health check."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run repository health check")
    parser.add_argument("--repo-path", default=".",
                       help="Path to repository")
    parser.add_argument("--output", default="health_report.json",
                       help="Output file for health report")
    parser.add_argument("--format", choices=["json", "markdown"],
                       default="json", help="Output format")
    
    args = parser.parse_args()
    
    # Initialize health checker
    checker = RepositoryHealthChecker(args.repo_path)
    
    # Run health check
    report = checker.run_health_check()
    
    # Save report
    if args.format == "json":
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Health report saved to {args.output}")
    elif args.format == "markdown":
        markdown_report = _generate_markdown_report(report)
        markdown_file = args.output.replace(".json", ".md")
        with open(markdown_file, 'w') as f:
            f.write(markdown_report)
        print(f"Markdown report saved to {markdown_file}")
    
    # Print summary
    summary = report["summary"]
    print(f"\nRepository Health Score: {summary['percentage']}% ({summary['grade']})")
    print(f"Status: {summary['status'].title()}")
    print(f"Issues found: {summary['issues_count']}")
    
    if summary['issues']:
        print("\nTop Issues:")
        for issue in summary['issues'][:5]:
            print(f"- {issue}")


def _generate_markdown_report(report: Dict[str, Any]) -> str:
    """Generate markdown report from health check results."""
    md = f"# Repository Health Report\n\n"
    md += f"**Generated:** {report['timestamp']}\n"
    md += f"**Repository:** {report['repository']}\n\n"
    
    summary = report['summary']
    md += f"## Overall Score: {summary['percentage']}% ({summary['grade']})\n\n"
    md += f"**Status:** {summary['status'].title()}\n"
    md += f"**Issues:** {summary['issues_count']}\n\n"
    
    for category, results in report['checks'].items():
        md += f"## {category.replace('_', ' ').title()}\n\n"
        md += f"**Score:** {results['percentage']}%\n\n"
        
        for check_name, check_data in results['checks'].items():
            status_emoji = "✅" if check_data['status'] == "good" else "⚠️" if check_data['status'] == "warning" else "❌"
            md += f"- {status_emoji} **{check_name.replace('_', ' ').title()}:** {check_data['score']}/{check_data['max_score']}\n"
        md += "\n"
    
    if summary['issues']:
        md += "## Issues\n\n"
        for issue in summary['issues']:
            md += f"- {issue}\n"
        md += "\n"
    
    return md


if __name__ == "__main__":
    main()