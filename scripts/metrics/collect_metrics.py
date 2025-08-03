#!/usr/bin/env python3
"""
Metrics Collection Script for Photonic Neuromorphics Project

This script collects comprehensive metrics about the repository,
code quality, security, and operational health.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import tempfile
import os


class MetricsCollector:
    """Collects various project metrics."""
    
    def __init__(self, repo_path: Path = None):
        self.repo_path = repo_path or Path.cwd()
        self.metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "repository": {
                "path": str(self.repo_path),
                "git_info": {}
            },
            "code_quality": {},
            "testing": {},
            "security": {},
            "dependencies": {},
            "build": {},
            "documentation": {}
        }
    
    def run_command(self, cmd: List[str], cwd: Path = None) -> Dict[str, Any]:
        """Run a command and return result with metrics."""
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
    
    def collect_git_metrics(self):
        """Collect Git repository metrics."""
        print("📊 Collecting Git metrics...")
        
        # Get current branch
        result = self.run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        if result["success"]:
            self.metrics["repository"]["git_info"]["current_branch"] = result["stdout"].strip()
        
        # Get commit count
        result = self.run_command(["git", "rev-list", "--count", "HEAD"])
        if result["success"]:
            self.metrics["repository"]["git_info"]["total_commits"] = int(result["stdout"].strip())
        
        # Get recent activity
        result = self.run_command(["git", "log", "--since='30 days ago'", "--oneline"])
        if result["success"]:
            self.metrics["repository"]["git_info"]["commits_last_30_days"] = len(result["stdout"].strip().split('\n')) if result["stdout"].strip() else 0
        
        # Get contributors
        result = self.run_command(["git", "log", "--format='%ae'", "--since='90 days ago'"])
        if result["success"]:
            contributors = set(result["stdout"].strip().split('\n')) if result["stdout"].strip() else set()
            self.metrics["repository"]["git_info"]["active_contributors_90_days"] = len(contributors)
    
    def collect_code_metrics(self):
        """Collect code quality metrics."""
        print("🔍 Collecting code quality metrics...")
        
        # Count lines of code
        extensions = ["*.py", "*.js", "*.ts", "*.go", "*.java", "*.cpp", "*.c"]
        total_loc = 0
        
        for ext in extensions:
            result = self.run_command(["find", "src", "-name", ext, "-exec", "wc", "-l", "{}", "+"])
            if result["success"]:
                lines = result["stdout"].strip().split('\n')
                for line in lines:
                    if line.strip() and not line.endswith("total"):
                        try:
                            total_loc += int(line.strip().split()[0])
                        except (ValueError, IndexError):
                            continue
        
        self.metrics["code_quality"]["lines_of_code"] = total_loc
        
        # Python-specific metrics
        if (self.repo_path / "pyproject.toml").exists() or (self.repo_path / "setup.py").exists():
            self._collect_python_metrics()
    
    def _collect_python_metrics(self):
        """Collect Python-specific code metrics."""
        # Check if Black is available
        result = self.run_command(["black", "--check", "--diff", "src", "tests"])
        self.metrics["code_quality"]["black_compliant"] = result["success"]
        
        # Check if isort is available
        result = self.run_command(["isort", "--check-only", "--diff", "src", "tests"])
        self.metrics["code_quality"]["isort_compliant"] = result["success"]
        
        # Run flake8 if available
        result = self.run_command(["flake8", "src", "tests", "--count"])
        if result["success"]:
            self.metrics["code_quality"]["flake8_violations"] = int(result["stdout"].strip()) if result["stdout"].strip().isdigit() else 0
        
        # Run mypy if available
        result = self.run_command(["mypy", "src", "--no-error-summary"])
        self.metrics["code_quality"]["mypy_compliant"] = result["success"]
    
    def collect_test_metrics(self):
        """Collect testing metrics."""
        print("🧪 Collecting test metrics...")
        
        # Count test files
        result = self.run_command(["find", "tests", "-name", "test_*.py", "-o", "-name", "*_test.py"])
        if result["success"]:
            test_files = len([line for line in result["stdout"].strip().split('\n') if line.strip()])
            self.metrics["testing"]["test_files"] = test_files
        
        # Run pytest with coverage if available
        if (self.repo_path / "pytest.ini").exists():
            result = self.run_command(["pytest", "--co", "-q"])
            if result["success"]:
                test_count = len([line for line in result["stdout"].split('\n') if 'test' in line.lower()])
                self.metrics["testing"]["test_count"] = test_count
            
            # Try to get coverage info
            result = self.run_command(["pytest", "--cov=src", "--cov-report=term-missing", "--quiet"])
            if result["success"] and "TOTAL" in result["stdout"]:
                lines = result["stdout"].split('\n')
                for line in lines:
                    if "TOTAL" in line:
                        parts = line.split()
                        if len(parts) >= 4 and parts[-1].endswith('%'):
                            try:
                                coverage = int(parts[-1][:-1])
                                self.metrics["testing"]["coverage_percentage"] = coverage
                            except ValueError:
                                pass
    
    def collect_security_metrics(self):
        """Collect security metrics."""
        print("🔒 Collecting security metrics...")
        
        # Run bandit if available
        result = self.run_command(["bandit", "-r", "src", "-f", "json"])
        if result["success"]:
            try:
                bandit_data = json.loads(result["stdout"])
                self.metrics["security"]["bandit_issues"] = len(bandit_data.get("results", []))
                severity_counts = {}
                for issue in bandit_data.get("results", []):
                    severity = issue.get("issue_severity", "UNKNOWN")
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
                self.metrics["security"]["bandit_severity_counts"] = severity_counts
            except json.JSONDecodeError:
                pass
        
        # Run safety if available
        result = self.run_command(["safety", "check", "--json"])
        if result["success"]:
            try:
                safety_data = json.loads(result["stdout"])
                self.metrics["security"]["safety_vulnerabilities"] = len(safety_data)
            except json.JSONDecodeError:
                pass
    
    def collect_dependency_metrics(self):
        """Collect dependency metrics."""
        print("📦 Collecting dependency metrics...")
        
        # Python dependencies
        if (self.repo_path / "requirements.txt").exists():
            with open(self.repo_path / "requirements.txt") as f:
                deps = [line.strip() for line in f if line.strip() and not line.startswith("#")]
                self.metrics["dependencies"]["python_dependencies"] = len(deps)
        
        # Check for outdated packages
        result = self.run_command(["pip", "list", "--outdated", "--format=json"])
        if result["success"]:
            try:
                outdated = json.loads(result["stdout"])
                self.metrics["dependencies"]["outdated_packages"] = len(outdated)
            except json.JSONDecodeError:
                pass
    
    def collect_build_metrics(self):
        """Collect build and deployment metrics."""
        print("🏗️ Collecting build metrics...")
        
        # Check Docker
        if (self.repo_path / "Dockerfile").exists():
            self.metrics["build"]["has_dockerfile"] = True
            
            # Try to build image to test
            result = self.run_command(["docker", "build", "-t", "test-build", "."])
            self.metrics["build"]["docker_build_success"] = result["success"]
        
        # Check if package can be built
        if (self.repo_path / "pyproject.toml").exists():
            result = self.run_command(["python", "-m", "build", "--outdir", "/tmp/build-test"])
            self.metrics["build"]["python_build_success"] = result["success"]
    
    def collect_documentation_metrics(self):
        """Collect documentation metrics."""
        print("📚 Collecting documentation metrics...")
        
        doc_files = [
            "README.md", "CONTRIBUTING.md", "LICENSE", "SECURITY.md",
            "CODE_OF_CONDUCT.md", "CHANGELOG.md"
        ]
        
        existing_docs = []
        for doc in doc_files:
            if (self.repo_path / doc).exists():
                existing_docs.append(doc)
        
        self.metrics["documentation"]["standard_docs_present"] = existing_docs
        self.metrics["documentation"]["documentation_coverage"] = len(existing_docs) / len(doc_files) * 100
        
        # Count documentation files in docs/
        if (self.repo_path / "docs").exists():
            result = self.run_command(["find", "docs", "-name", "*.md", "-o", "-name", "*.rst"])
            if result["success"]:
                doc_count = len([line for line in result["stdout"].strip().split('\n') if line.strip()])
                self.metrics["documentation"]["docs_directory_files"] = doc_count
    
    def generate_report(self, format_type: str = "json") -> str:
        """Generate metrics report in specified format."""
        if format_type == "json":
            return json.dumps(self.metrics, indent=2)
        elif format_type == "summary":
            return self._generate_summary_report()
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _generate_summary_report(self) -> str:
        """Generate a human-readable summary report."""
        lines = [
            "# Project Metrics Summary",
            f"Generated: {self.metrics['timestamp']}",
            "",
            "## Repository Information",
            f"- Current Branch: {self.metrics['repository']['git_info'].get('current_branch', 'unknown')}",
            f"- Total Commits: {self.metrics['repository']['git_info'].get('total_commits', 'unknown')}",
            f"- Recent Activity (30 days): {self.metrics['repository']['git_info'].get('commits_last_30_days', 'unknown')} commits",
            f"- Active Contributors (90 days): {self.metrics['repository']['git_info'].get('active_contributors_90_days', 'unknown')}",
            "",
            "## Code Quality",
            f"- Lines of Code: {self.metrics['code_quality'].get('lines_of_code', 'unknown')}",
            f"- Black Compliant: {'✅' if self.metrics['code_quality'].get('black_compliant') else '❌'}",
            f"- isort Compliant: {'✅' if self.metrics['code_quality'].get('isort_compliant') else '❌'}",
            f"- MyPy Compliant: {'✅' if self.metrics['code_quality'].get('mypy_compliant') else '❌'}",
            f"- Flake8 Violations: {self.metrics['code_quality'].get('flake8_violations', 'unknown')}",
            "",
            "## Testing",
            f"- Test Files: {self.metrics['testing'].get('test_files', 'unknown')}",
            f"- Test Count: {self.metrics['testing'].get('test_count', 'unknown')}",
            f"- Coverage: {self.metrics['testing'].get('coverage_percentage', 'unknown')}%",
            "",
            "## Security",
            f"- Bandit Issues: {self.metrics['security'].get('bandit_issues', 'unknown')}",
            f"- Safety Vulnerabilities: {self.metrics['security'].get('safety_vulnerabilities', 'unknown')}",
            "",
            "## Dependencies",
            f"- Python Dependencies: {self.metrics['dependencies'].get('python_dependencies', 'unknown')}",
            f"- Outdated Packages: {self.metrics['dependencies'].get('outdated_packages', 'unknown')}",
            "",
            "## Build Status",
            f"- Docker Build: {'✅' if self.metrics['build'].get('docker_build_success') else '❌'}",
            f"- Python Build: {'✅' if self.metrics['build'].get('python_build_success') else '❌'}",
            "",
            "## Documentation",
            f"- Documentation Coverage: {self.metrics['documentation'].get('documentation_coverage', 0):.1f}%",
            f"- Docs Directory Files: {self.metrics['documentation'].get('docs_directory_files', 'unknown')}",
        ]
        
        return '\n'.join(lines)
    
    def collect_all_metrics(self):
        """Collect all available metrics."""
        print("🚀 Starting comprehensive metrics collection...")
        
        self.collect_git_metrics()
        self.collect_code_metrics()
        self.collect_test_metrics()
        self.collect_security_metrics()
        self.collect_dependency_metrics()
        self.collect_build_metrics()
        self.collect_documentation_metrics()
        
        print("✅ Metrics collection completed!")


def main():
    parser = argparse.ArgumentParser(description="Collect project metrics")
    parser.add_argument("--format", choices=["json", "summary"], default="summary",
                        help="Output format")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--report", action="store_true",
                        help="Generate and display summary report")
    parser.add_argument("--repo-path", help="Repository path (default: current directory)")
    
    args = parser.parse_args()
    
    repo_path = Path(args.repo_path) if args.repo_path else Path.cwd()
    collector = MetricsCollector(repo_path)
    
    try:
        collector.collect_all_metrics()
        
        if args.report or args.format == "summary":
            report = collector.generate_report("summary")
            print("\n" + report)
        
        if args.format == "json" or args.output:
            output = collector.generate_report(args.format)
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(output)
                print(f"\n📊 Metrics saved to: {args.output}")
            else:
                print(output)
    
    except Exception as e:
        print(f"❌ Error collecting metrics: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()