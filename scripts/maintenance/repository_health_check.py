#!/usr/bin/env python3
"""
Repository Health Check Script

This script performs comprehensive health checks on the repository
including security, dependencies, documentation, and best practices.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests


class RepositoryHealthChecker:
    """Performs comprehensive repository health assessments."""
    
    def __init__(self, repo_path: Path = None):
        self.repo_path = repo_path or Path.cwd()
        self.health_score = 0
        self.max_score = 0
        self.issues = []
        self.recommendations = []
        
    def run_command(self, cmd: List[str], cwd: Path = None) -> Dict[str, any]:
        """Run a command and return structured result."""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.repo_path,
                capture_output=True,
                text=True,
                timeout=300
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Command timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def add_check(self, name: str, passed: bool, score: int, 
                  message: str = None, recommendation: str = None):
        """Add a health check result."""
        self.max_score += score
        if passed:
            self.health_score += score
            status = "✅"
        else:
            status = "❌"
            if message:
                self.issues.append(f"{name}: {message}")
            if recommendation:
                self.recommendations.append(recommendation)
        
        print(f"{status} {name}" + (f" - {message}" if message else ""))
    
    def check_git_repository(self) -> Dict[str, any]:
        """Check Git repository health."""
        print("\n🔍 Checking Git Repository Health...")
        
        checks = {}
        
        # Check if it's a git repository
        is_git_repo = (self.repo_path / ".git").exists()
        self.add_check(
            "Git Repository", 
            is_git_repo, 
            5,
            None if is_git_repo else "Not a Git repository",
            "Initialize Git repository with 'git init'" if not is_git_repo else None
        )
        checks["is_git_repo"] = is_git_repo
        
        if not is_git_repo:
            return checks
        
        # Check for commits
        result = self.run_command(["git", "rev-list", "--count", "HEAD"])
        commit_count = int(result["stdout"].strip()) if result["success"] else 0
        has_commits = commit_count > 0
        self.add_check(
            "Has Commits",
            has_commits,
            3,
            f"{commit_count} commits" if has_commits else "No commits found",
            "Make initial commit" if not has_commits else None
        )
        checks["commit_count"] = commit_count
        
        # Check for remote
        result = self.run_command(["git", "remote", "-v"])
        has_remote = result["success"] and result["stdout"].strip()
        self.add_check(
            "Remote Repository",
            has_remote,
            3,
            "Remote configured" if has_remote else "No remote repository",
            "Add remote with 'git remote add origin <url>'" if not has_remote else None
        )
        checks["has_remote"] = has_remote
        
        # Check working directory status
        result = self.run_command(["git", "status", "--porcelain"])
        is_clean = result["success"] and not result["stdout"].strip()
        self.add_check(
            "Clean Working Directory",
            is_clean,
            2,
            "Working directory clean" if is_clean else "Uncommitted changes",
            "Commit or stash changes" if not is_clean else None
        )
        checks["is_clean"] = is_clean
        
        return checks
    
    def check_project_structure(self) -> Dict[str, any]:
        """Check project structure and organization."""
        print("\n📁 Checking Project Structure...")
        
        checks = {}
        
        # Essential files
        essential_files = [
            ("README.md", "Project documentation"),
            ("LICENSE", "License file"),
            (".gitignore", "Git ignore patterns"),
        ]
        
        for file_name, description in essential_files:
            exists = (self.repo_path / file_name).exists()
            self.add_check(
                f"{description}",
                exists,
                3,
                "Present" if exists else "Missing",
                f"Create {file_name}" if not exists else None
            )
            checks[f"has_{file_name.lower().replace('.', '_')}"] = exists
        
        # Source code organization
        src_dirs = ["src/", "lib/", "app/"]
        has_src_structure = any((self.repo_path / d).exists() for d in src_dirs)
        self.add_check(
            "Source Code Organization",
            has_src_structure,
            2,
            "Organized source structure" if has_src_structure else "No clear source structure",
            "Organize source code in src/ directory" if not has_src_structure else None
        )
        checks["has_src_structure"] = has_src_structure
        
        # Tests directory
        test_dirs = ["tests/", "test/", "spec/"]
        has_tests_dir = any((self.repo_path / d).exists() for d in test_dirs)
        self.add_check(
            "Tests Directory",
            has_tests_dir,
            3,
            "Tests directory present" if has_tests_dir else "No tests directory",
            "Create tests/ directory" if not has_tests_dir else None
        )
        checks["has_tests_dir"] = has_tests_dir
        
        # Documentation directory
        docs_dirs = ["docs/", "doc/", "documentation/"]
        has_docs_dir = any((self.repo_path / d).exists() for d in docs_dirs)
        self.add_check(
            "Documentation Directory",
            has_docs_dir,
            2,
            "Documentation directory present" if has_docs_dir else "No documentation directory",
            "Create docs/ directory" if not has_docs_dir else None
        )
        checks["has_docs_dir"] = has_docs_dir
        
        return checks
    
    def check_dependencies(self) -> Dict[str, any]:
        """Check dependency management."""
        print("\n📦 Checking Dependencies...")
        
        checks = {}
        
        # Python dependencies
        python_dep_files = ["requirements.txt", "pyproject.toml", "setup.py", "Pipfile"]
        has_python_deps = any((self.repo_path / f).exists() for f in python_dep_files)
        self.add_check(
            "Python Dependencies",
            has_python_deps,
            3,
            "Dependency file present" if has_python_deps else "No dependency management",
            "Create requirements.txt or pyproject.toml" if not has_python_deps else None
        )
        checks["has_python_deps"] = has_python_deps
        
        # Check for outdated packages (if pip available)
        if has_python_deps:
            result = self.run_command(["pip", "list", "--outdated", "--format=json"])
            if result["success"]:
                try:
                    outdated = json.loads(result["stdout"])
                    outdated_count = len(outdated)
                    is_current = outdated_count == 0
                    self.add_check(
                        "Dependencies Up-to-date",
                        is_current,
                        2,
                        "All dependencies current" if is_current else f"{outdated_count} outdated packages",
                        "Update outdated dependencies" if not is_current else None
                    )
                    checks["outdated_packages"] = outdated_count
                except json.JSONDecodeError:
                    pass
        
        # Security vulnerabilities
        result = self.run_command(["safety", "check", "--json"])
        if result["returncode"] is not None:  # Safety was found
            has_vulns = result["returncode"] != 0
            if has_vulns and result["stdout"]:
                try:
                    vulns = json.loads(result["stdout"])
                    vuln_count = len(vulns)
                except json.JSONDecodeError:
                    vuln_count = 1  # At least one vulnerability
            else:
                vuln_count = 0
            
            is_secure = vuln_count == 0
            self.add_check(
                "Security Vulnerabilities",
                is_secure,
                5,
                "No vulnerabilities found" if is_secure else f"{vuln_count} vulnerabilities",
                "Fix security vulnerabilities" if not is_secure else None
            )
            checks["vulnerability_count"] = vuln_count
        
        return checks
    
    def check_code_quality(self) -> Dict[str, any]:
        """Check code quality and standards."""
        print("\n🔍 Checking Code Quality...")
        
        checks = {}
        
        # Code formatting (Python)
        if (self.repo_path / "pyproject.toml").exists() or any(Path(self.repo_path).glob("*.py")):
            # Check Black formatting
            result = self.run_command(["black", "--check", ".", "--quiet"])
            is_formatted = result["success"]
            self.add_check(
                "Code Formatting (Black)",
                is_formatted,
                3,
                "Code properly formatted" if is_formatted else "Code needs formatting",
                "Run 'black .' to format code" if not is_formatted else None
            )
            checks["black_formatted"] = is_formatted
            
            # Check linting
            result = self.run_command(["flake8", ".", "--count"])
            if result["success"]:
                violation_count = int(result["stdout"].strip()) if result["stdout"].strip().isdigit() else 0
                is_clean = violation_count == 0
                self.add_check(
                    "Linting (Flake8)",
                    is_clean,
                    3,
                    "No linting violations" if is_clean else f"{violation_count} violations",
                    "Fix linting violations" if not is_clean else None
                )
                checks["flake8_violations"] = violation_count
            
            # Type checking
            result = self.run_command(["mypy", ".", "--no-error-summary"])
            is_typed = result["success"]
            self.add_check(
                "Type Checking (MyPy)",
                is_typed,
                2,
                "Type checking passes" if is_typed else "Type errors found",
                "Fix type annotations" if not is_typed else None
            )
            checks["mypy_clean"] = is_typed
        
        return checks
    
    def check_testing(self) -> Dict[str, any]:
        """Check testing setup and coverage."""
        print("\n🧪 Checking Testing...")
        
        checks = {}
        
        # Test configuration
        test_configs = ["pytest.ini", "tox.ini", ".coverage", "setup.cfg"]
        has_test_config = any((self.repo_path / f).exists() for f in test_configs)
        self.add_check(
            "Test Configuration",
            has_test_config,
            2,
            "Test configuration present" if has_test_config else "No test configuration",
            "Create pytest.ini or similar test config" if not has_test_config else None
        )
        checks["has_test_config"] = has_test_config
        
        # Test files
        test_files = list(Path(self.repo_path).rglob("test_*.py")) + list(Path(self.repo_path).rglob("*_test.py"))
        test_count = len(test_files)
        has_tests = test_count > 0
        self.add_check(
            "Test Files",
            has_tests,
            4,
            f"{test_count} test files found" if has_tests else "No test files",
            "Create test files" if not has_tests else None
        )
        checks["test_count"] = test_count
        
        # Run tests if available
        if has_tests:
            result = self.run_command(["pytest", "--tb=no", "-q"])
            tests_pass = result["success"]
            self.add_check(
                "Test Execution",
                tests_pass,
                5,
                "All tests pass" if tests_pass else "Some tests fail",
                "Fix failing tests" if not tests_pass else None
            )
            checks["tests_pass"] = tests_pass
            
            # Coverage check
            result = self.run_command(["pytest", "--cov=src", "--cov-report=term-missing", "--tb=no", "-q"])
            if result["success"]:
                # Extract coverage percentage
                lines = result["stdout"].split('\n')
                coverage = 0
                for line in lines:
                    if "TOTAL" in line and "%" in line:
                        parts = line.split()
                        for part in parts:
                            if part.endswith('%'):
                                try:
                                    coverage = float(part[:-1])
                                    break
                                except ValueError:
                                    pass
                
                good_coverage = coverage >= 80
                self.add_check(
                    "Test Coverage",
                    good_coverage,
                    3,
                    f"{coverage:.1f}% coverage" if coverage > 0 else "No coverage data",
                    "Increase test coverage to 80%+" if not good_coverage else None
                )
                checks["coverage"] = coverage
        
        return checks
    
    def check_security(self) -> Dict[str, any]:
        """Check security practices."""
        print("\n🔒 Checking Security...")
        
        checks = {}
        
        # Security policy
        has_security_md = (self.repo_path / "SECURITY.md").exists()
        self.add_check(
            "Security Policy",
            has_security_md,
            2,
            "Security policy present" if has_security_md else "No security policy",
            "Create SECURITY.md" if not has_security_md else None
        )
        checks["has_security_policy"] = has_security_md
        
        # Secrets scanning
        result = self.run_command(["git", "log", "--all", "--grep=password", "--grep=secret", "--grep=key"])
        has_secret_commits = result["success"] and result["stdout"].strip()
        self.add_check(
            "No Secrets in History",
            not has_secret_commits,
            4,
            "No obvious secrets in commit history" if not has_secret_commits else "Potential secrets in commits",
            "Review and remove secrets from git history" if has_secret_commits else None
        )
        checks["clean_commit_history"] = not has_secret_commits
        
        # Bandit security scan (Python)
        result = self.run_command(["bandit", "-r", ".", "-f", "json"])
        if result["returncode"] is not None:  # Bandit was found
            if result["success"]:
                try:
                    bandit_data = json.loads(result["stdout"])
                    issue_count = len(bandit_data.get("results", []))
                except json.JSONDecodeError:
                    issue_count = 0
            else:
                issue_count = 1  # At least some issues
            
            is_secure = issue_count == 0
            self.add_check(
                "Security Scan (Bandit)",
                is_secure,
                3,
                "No security issues" if is_secure else f"{issue_count} security issues",
                "Fix security issues found by Bandit" if not is_secure else None
            )
            checks["bandit_issues"] = issue_count
        
        return checks
    
    def check_documentation(self) -> Dict[str, any]:
        """Check documentation quality."""
        print("\n📚 Checking Documentation...")
        
        checks = {}
        
        # README quality
        readme_path = self.repo_path / "README.md"
        if readme_path.exists():
            with open(readme_path, 'r') as f:
                readme_content = f.read()
            
            readme_sections = [
                "installation", "usage", "example", "contributing",
                "license", "getting started", "quick start"
            ]
            
            section_count = sum(1 for section in readme_sections 
                              if section.lower() in readme_content.lower())
            
            good_readme = section_count >= 3
            self.add_check(
                "README Quality",
                good_readme,
                3,
                f"README has {section_count}/7 key sections" if good_readme else "README needs improvement",
                "Improve README with installation, usage, and examples" if not good_readme else None
            )
            checks["readme_quality"] = section_count
        
        # API documentation
        has_api_docs = any(Path(self.repo_path).rglob("*.rst")) or \
                      (self.repo_path / "docs").exists()
        self.add_check(
            "API Documentation",
            has_api_docs,
            2,
            "API documentation present" if has_api_docs else "No API documentation",
            "Add API documentation" if not has_api_docs else None
        )
        checks["has_api_docs"] = has_api_docs
        
        # Contributing guidelines
        contributing_files = ["CONTRIBUTING.md", "CONTRIBUTING.rst", "docs/CONTRIBUTING.md"]
        has_contributing = any((self.repo_path / f).exists() for f in contributing_files)
        self.add_check(
            "Contributing Guidelines",
            has_contributing,
            2,
            "Contributing guidelines present" if has_contributing else "No contributing guidelines",
            "Create CONTRIBUTING.md" if not has_contributing else None
        )
        checks["has_contributing"] = has_contributing
        
        return checks
    
    def check_ci_cd(self) -> Dict[str, any]:
        """Check CI/CD setup."""
        print("\n🔄 Checking CI/CD...")
        
        checks = {}
        
        # GitHub Actions
        github_workflows = list((self.repo_path / ".github" / "workflows").glob("*.yml")) if \
                          (self.repo_path / ".github" / "workflows").exists() else []
        has_github_actions = len(github_workflows) > 0
        self.add_check(
            "GitHub Actions",
            has_github_actions,
            3,
            f"{len(github_workflows)} workflows configured" if has_github_actions else "No GitHub Actions",
            "Set up GitHub Actions workflows" if not has_github_actions else None
        )
        checks["github_actions_count"] = len(github_workflows)
        
        # Other CI configurations
        ci_files = [".travis.yml", "circle.yml", ".gitlab-ci.yml", "azure-pipelines.yml"]
        has_other_ci = any((self.repo_path / f).exists() for f in ci_files)
        self.add_check(
            "CI Configuration",
            has_github_actions or has_other_ci,
            2,
            "CI configured" if (has_github_actions or has_other_ci) else "No CI configuration",
            "Set up continuous integration" if not (has_github_actions or has_other_ci) else None
        )
        checks["has_ci"] = has_github_actions or has_other_ci
        
        # Dependabot
        dependabot_config = (self.repo_path / ".github" / "dependabot.yml").exists()
        self.add_check(
            "Dependency Updates",
            dependabot_config,
            2,
            "Dependabot configured" if dependabot_config else "No automated dependency updates",
            "Configure Dependabot for dependency updates" if not dependabot_config else None
        )
        checks["has_dependabot"] = dependabot_config
        
        return checks
    
    def generate_health_report(self, format_type: str = "markdown") -> str:
        """Generate comprehensive health report."""
        health_percentage = (self.health_score / self.max_score * 100) if self.max_score > 0 else 0
        
        if format_type == "markdown":
            lines = [
                "# Repository Health Report",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Repository: {self.repo_path}",
                "",
                f"## Overall Health Score: {health_percentage:.1f}% ({self.health_score}/{self.max_score})",
                "",
                self._get_health_grade(health_percentage),
                ""
            ]
            
            if self.issues:
                lines.extend([
                    "## Issues Found",
                    ""
                ])
                for issue in self.issues:
                    lines.append(f"- ❌ {issue}")
                lines.append("")
            
            if self.recommendations:
                lines.extend([
                    "## Recommendations",
                    ""
                ])
                for rec in self.recommendations:
                    lines.append(f"- 💡 {rec}")
                lines.append("")
            
            lines.extend([
                "## Health Categories",
                f"- **Repository Structure**: Well-organized project layout",
                f"- **Code Quality**: Formatting, linting, and type checking",
                f"- **Testing**: Test coverage and execution",
                f"- **Security**: Vulnerability scanning and best practices",
                f"- **Documentation**: README, API docs, and guidelines",
                f"- **CI/CD**: Automated workflows and dependency management",
                "",
                "---",
                "*Generated by Repository Health Checker*"
            ])
            
            return '\n'.join(lines)
        
        elif format_type == "json":
            return json.dumps({
                "timestamp": datetime.now().isoformat(),
                "repository": str(self.repo_path),
                "health_score": self.health_score,
                "max_score": self.max_score,
                "health_percentage": health_percentage,
                "grade": self._get_health_grade(health_percentage),
                "issues": self.issues,
                "recommendations": self.recommendations
            }, indent=2)
        
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _get_health_grade(self, percentage: float) -> str:
        """Get health grade based on percentage."""
        if percentage >= 90:
            return "🏆 Excellent (A)"
        elif percentage >= 80:
            return "✅ Good (B)"
        elif percentage >= 70:
            return "⚠️ Fair (C)"
        elif percentage >= 60:
            return "🔶 Poor (D)"
        else:
            return "🔴 Critical (F)"
    
    def run_complete_health_check(self):
        """Run all health checks."""
        print("🏥 Starting Repository Health Check...")
        
        # Run all health checks
        self.check_git_repository()
        self.check_project_structure()
        self.check_dependencies()
        self.check_code_quality()
        self.check_testing()
        self.check_security()
        self.check_documentation()
        self.check_ci_cd()
        
        print(f"\n📊 Health Check Complete!")
        print(f"Overall Score: {self.health_score}/{self.max_score} ({self.health_score/self.max_score*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Check repository health")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown",
                        help="Output format")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--repo-path", help="Repository path (default: current directory)")
    
    args = parser.parse_args()
    
    repo_path = Path(args.repo_path) if args.repo_path else Path.cwd()
    checker = RepositoryHealthChecker(repo_path)
    
    try:
        checker.run_complete_health_check()
        
        report = checker.generate_health_report(args.format)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"\n📄 Report saved to: {args.output}")
        else:
            print("\n" + report)
    
    except Exception as e:
        print(f"❌ Error in health check: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()