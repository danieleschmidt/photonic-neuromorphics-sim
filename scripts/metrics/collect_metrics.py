#!/usr/bin/env python3
"""
Automated metrics collection script for photonic neuromorphics simulation platform.
Collects code quality, security, performance, and business metrics.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

import requests
from prometheus_client.parser import text_string_to_metric_families


class MetricsCollector:
    """Centralized metrics collection system."""
    
    def __init__(self, config_path: str = ".github/project-metrics.json"):
        """Initialize metrics collector with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.metrics_data = {}
        self.timestamp = datetime.utcnow().isoformat()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load metrics configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Configuration file not found: {self.config_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in configuration file: {e}")
            sys.exit(1)
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all configured metrics."""
        print(f"Starting metrics collection at {self.timestamp}")
        
        # Collect different metric categories
        self.collect_code_quality_metrics()
        self.collect_security_metrics()
        self.collect_performance_metrics()
        self.collect_development_metrics()
        self.collect_deployment_metrics()
        self.collect_reliability_metrics()
        self.collect_business_metrics()
        
        return self.metrics_data
    
    def collect_code_quality_metrics(self):
        """Collect code quality metrics."""
        print("Collecting code quality metrics...")
        
        metrics = {}
        
        # Test coverage
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "--cov=src", "--cov-report=json", "--cov-report=term"],
                capture_output=True, text=True, cwd="."
            )
            if result.returncode == 0 and os.path.exists("coverage.json"):
                with open("coverage.json", 'r') as f:
                    coverage_data = json.load(f)
                    metrics["test_coverage"] = round(coverage_data["totals"]["percent_covered"], 2)
        except Exception as e:
            print(f"Error collecting test coverage: {e}")
            metrics["test_coverage"] = 0
        
        # Code complexity
        try:
            result = subprocess.run(
                ["radon", "cc", "src/", "--json"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                complexity_data = json.loads(result.stdout)
                complexities = []
                for file_data in complexity_data.values():
                    for item in file_data:
                        if isinstance(item, dict) and 'complexity' in item:
                            complexities.append(item['complexity'])
                metrics["code_complexity"] = round(sum(complexities) / len(complexities), 2) if complexities else 0
        except Exception as e:
            print(f"Error collecting code complexity: {e}")
            metrics["code_complexity"] = 0
        
        # Maintainability index
        try:
            result = subprocess.run(
                ["radon", "mi", "src/", "--json"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                mi_data = json.loads(result.stdout)
                mi_scores = []
                for file_data in mi_data.values():
                    if isinstance(file_data, dict) and 'mi' in file_data:
                        mi_scores.append(file_data['mi'])
                metrics["maintainability_index"] = round(sum(mi_scores) / len(mi_scores), 2) if mi_scores else 0
        except Exception as e:
            print(f"Error collecting maintainability index: {e}")
            metrics["maintainability_index"] = 0
        
        self.metrics_data["code_quality"] = metrics
    
    def collect_security_metrics(self):
        """Collect security metrics."""
        print("Collecting security metrics...")
        
        metrics = {"vulnerabilities": {"critical": 0, "high": 0, "medium": 0}}
        
        # Bandit security scan
        try:
            result = subprocess.run(
                ["bandit", "-r", "src/", "-f", "json"],
                capture_output=True, text=True
            )
            if result.stdout:
                bandit_data = json.loads(result.stdout)
                for issue in bandit_data.get("results", []):
                    severity = issue.get("issue_severity", "").lower()
                    if severity in metrics["vulnerabilities"]:
                        metrics["vulnerabilities"][severity] += 1
        except Exception as e:
            print(f"Error running Bandit scan: {e}")
        
        # Safety dependency check
        try:
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True, text=True
            )
            if result.stdout and result.stdout.strip():
                safety_data = json.loads(result.stdout)
                metrics["dependency_vulnerabilities"] = len(safety_data)
            else:
                metrics["dependency_vulnerabilities"] = 0
        except Exception as e:
            print(f"Error running Safety check: {e}")
            metrics["dependency_vulnerabilities"] = 0
        
        # Secrets detection
        try:
            result = subprocess.run(
                ["detect-secrets", "scan", "--all-files"],
                capture_output=True, text=True
            )
            if result.stdout:
                secrets_data = json.loads(result.stdout)
                metrics["secrets_exposed"] = len(secrets_data.get("results", {}))
            else:
                metrics["secrets_exposed"] = 0
        except Exception as e:
            print(f"Error running secrets detection: {e}")
            metrics["secrets_exposed"] = 0
        
        self.metrics_data["security"] = metrics
    
    def collect_performance_metrics(self):
        """Collect performance metrics."""
        print("Collecting performance metrics...")
        
        metrics = {}
        
        # Build time from GitHub Actions
        try:
            if os.getenv("GITHUB_TOKEN"):
                metrics.update(self._get_github_build_metrics())
        except Exception as e:
            print(f"Error collecting GitHub metrics: {e}")
        
        # Test execution time
        try:
            start_time = time.time()
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "--tb=no", "-q"],
                capture_output=True, text=True
            )
            test_duration = time.time() - start_time
            metrics["test_execution_time"] = round(test_duration, 2)
        except Exception as e:
            print(f"Error measuring test execution time: {e}")
            metrics["test_execution_time"] = 0
        
        # Memory usage simulation
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            metrics["memory_usage"] = round(memory_info.rss / 1024 / 1024, 2)  # MB
        except Exception as e:
            print(f"Error collecting memory usage: {e}")
            metrics["memory_usage"] = 0
        
        self.metrics_data["performance"] = metrics
    
    def collect_development_metrics(self):
        """Collect development process metrics."""
        print("Collecting development metrics...")
        
        metrics = {}
        
        # Commit frequency
        try:
            since_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            result = subprocess.run(
                ["git", "log", "--since", since_date, "--oneline"],
                capture_output=True, text=True
            )
            commit_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            metrics["commit_frequency"] = round(commit_count / 7, 2)
        except Exception as e:
            print(f"Error collecting commit frequency: {e}")
            metrics["commit_frequency"] = 0
        
        # GitHub PR metrics
        try:
            if os.getenv("GITHUB_TOKEN"):
                metrics.update(self._get_github_pr_metrics())
        except Exception as e:
            print(f"Error collecting PR metrics: {e}")
        
        self.metrics_data["development"] = metrics
    
    def collect_deployment_metrics(self):
        """Collect deployment metrics."""
        print("Collecting deployment metrics...")
        
        metrics = {}
        
        # GitHub deployment metrics
        try:
            if os.getenv("GITHUB_TOKEN"):
                metrics.update(self._get_github_deployment_metrics())
        except Exception as e:
            print(f"Error collecting deployment metrics: {e}")
        
        self.metrics_data["deployment"] = metrics
    
    def collect_reliability_metrics(self):
        """Collect reliability metrics."""
        print("Collecting reliability metrics...")
        
        metrics = {}
        
        # Prometheus metrics (if available)
        try:
            prometheus_url = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
            metrics.update(self._get_prometheus_metrics(prometheus_url))
        except Exception as e:
            print(f"Error collecting Prometheus metrics: {e}")
        
        self.metrics_data["reliability"] = metrics
    
    def collect_business_metrics(self):
        """Collect business and application-specific metrics."""
        print("Collecting business metrics...")
        
        metrics = {}
        
        # Application-specific metrics would go here
        # For now, using placeholder values
        metrics["user_adoption"] = 0
        metrics["simulation_success_rate"] = 0
        metrics["average_simulation_time"] = 0
        metrics["model_accuracy"] = 0
        metrics["research_productivity"] = 0
        
        self.metrics_data["business"] = metrics
    
    def _get_github_build_metrics(self) -> Dict[str, Any]:
        """Get build metrics from GitHub Actions API."""
        repo = os.getenv("GITHUB_REPOSITORY", "danieleschmidt/photonic-neuromorphics-sim")
        token = os.getenv("GITHUB_TOKEN")
        
        headers = {"Authorization": f"token {token}"}
        url = f"https://api.github.com/repos/{repo}/actions/runs"
        
        response = requests.get(url, headers=headers, params={"per_page": 10})
        if response.status_code == 200:
            runs = response.json()["workflow_runs"]
            durations = []
            for run in runs:
                if run["status"] == "completed":
                    start = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00"))
                    end = datetime.fromisoformat(run["updated_at"].replace("Z", "+00:00"))
                    duration = (end - start).total_seconds()
                    durations.append(duration)
            
            return {
                "build_time": round(sum(durations) / len(durations), 2) if durations else 0
            }
        return {}
    
    def _get_github_pr_metrics(self) -> Dict[str, Any]:
        """Get pull request metrics from GitHub API."""
        repo = os.getenv("GITHUB_REPOSITORY", "danieleschmidt/photonic-neuromorphics-sim")
        token = os.getenv("GITHUB_TOKEN")
        
        headers = {"Authorization": f"token {token}"}
        url = f"https://api.github.com/repos/{repo}/pulls"
        
        response = requests.get(url, headers=headers, params={"state": "closed", "per_page": 20})
        if response.status_code == 200:
            prs = response.json()
            cycle_times = []
            
            for pr in prs:
                if pr["merged_at"]:
                    created = datetime.fromisoformat(pr["created_at"].replace("Z", "+00:00"))
                    merged = datetime.fromisoformat(pr["merged_at"].replace("Z", "+00:00"))
                    cycle_time = (merged - created).total_seconds() / 3600  # hours
                    cycle_times.append(cycle_time)
            
            return {
                "pull_request_cycle_time": round(sum(cycle_times) / len(cycle_times), 2) if cycle_times else 0
            }
        return {}
    
    def _get_github_deployment_metrics(self) -> Dict[str, Any]:
        """Get deployment metrics from GitHub API."""
        repo = os.getenv("GITHUB_REPOSITORY", "danieleschmidt/photonic-neuromorphics-sim")
        token = os.getenv("GITHUB_TOKEN")
        
        headers = {"Authorization": f"token {token}"}
        url = f"https://api.github.com/repos/{repo}/deployments"
        
        response = requests.get(url, headers=headers, params={"per_page": 20})
        if response.status_code == 200:
            deployments = response.json()
            
            # Calculate deployment frequency (last 7 days)
            week_ago = datetime.now() - timedelta(days=7)
            recent_deployments = [
                d for d in deployments
                if datetime.fromisoformat(d["created_at"].replace("Z", "+00:00")) > week_ago
            ]
            
            return {
                "deployment_frequency": len(recent_deployments),
                "deployment_success_rate": 95  # Placeholder - would need deployment status
            }
        return {}
    
    def _get_prometheus_metrics(self, prometheus_url: str) -> Dict[str, Any]:
        """Get metrics from Prometheus."""
        metrics = {}
        
        queries = {
            "uptime": "up",
            "error_rate": "rate(http_requests_total{status=~\"5..\"}[5m])",
            "response_time_p95": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
        }
        
        for metric_name, query in queries.items():
            try:
                url = f"{prometheus_url}/api/v1/query"
                response = requests.get(url, params={"query": query})
                if response.status_code == 200:
                    data = response.json()
                    if data["data"]["result"]:
                        value = float(data["data"]["result"][0]["value"][1])
                        metrics[metric_name] = round(value, 2)
            except Exception as e:
                print(f"Error querying Prometheus for {metric_name}: {e}")
        
        return metrics
    
    def save_metrics(self, output_path: str = "metrics_data.json"):
        """Save collected metrics to file."""
        output_data = {
            "timestamp": self.timestamp,
            "metrics": self.metrics_data,
            "collection_info": {
                "script_version": "1.0.0",
                "collection_duration": time.time(),
                "environment": os.getenv("ENVIRONMENT", "development")
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Metrics saved to {output_path}")
    
    def send_to_prometheus(self, pushgateway_url: str = "http://localhost:9091"):
        """Send metrics to Prometheus Pushgateway."""
        try:
            import prometheus_client
            
            registry = prometheus_client.CollectorRegistry()
            
            # Create Prometheus metrics
            for category, metrics in self.metrics_data.items():
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        gauge = prometheus_client.Gauge(
                            f"project_{category}_{metric_name}",
                            f"{category} {metric_name}",
                            registry=registry
                        )
                        gauge.set(value)
            
            # Push to gateway
            prometheus_client.push_to_gateway(
                pushgateway_url,
                job="metrics_collector",
                registry=registry
            )
            print(f"Metrics pushed to Prometheus at {pushgateway_url}")
            
        except Exception as e:
            print(f"Error sending metrics to Prometheus: {e}")
    
    def generate_summary_report(self) -> str:
        """Generate a summary report of collected metrics."""
        report = f"# Metrics Collection Report - {self.timestamp}\n\n"
        
        for category, metrics in self.metrics_data.items():
            report += f"## {category.title()}\n"
            for metric_name, value in metrics.items():
                if isinstance(value, dict):
                    report += f"### {metric_name}\n"
                    for sub_metric, sub_value in value.items():
                        report += f"- {sub_metric}: {sub_value}\n"
                else:
                    report += f"- {metric_name}: {value}\n"
            report += "\n"
        
        return report


def main():
    """Main function to run metrics collection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect project metrics")
    parser.add_argument("--config", default=".github/project-metrics.json",
                       help="Path to metrics configuration file")
    parser.add_argument("--output", default="metrics_data.json",
                       help="Output file for metrics data")
    parser.add_argument("--prometheus", help="Prometheus Pushgateway URL")
    parser.add_argument("--report", action="store_true",
                       help="Generate summary report")
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = MetricsCollector(args.config)
    
    # Collect metrics
    metrics = collector.collect_all_metrics()
    
    # Save metrics
    collector.save_metrics(args.output)
    
    # Send to Prometheus if specified
    if args.prometheus:
        collector.send_to_prometheus(args.prometheus)
    
    # Generate report if requested
    if args.report:
        report = collector.generate_summary_report()
        print("\n" + report)
        
        with open("metrics_report.md", "w") as f:
            f.write(report)
        print("Report saved to metrics_report.md")
    
    print("Metrics collection completed successfully!")


if __name__ == "__main__":
    main()