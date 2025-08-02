#!/usr/bin/env python3
"""
Code Quality Monitoring Script

This script monitors code quality metrics over time and provides
trend analysis and alerts for quality degradation.
"""

import argparse
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile
import os


class CodeQualityMonitor:
    """Monitors and tracks code quality metrics over time."""
    
    def __init__(self, repo_path: Path = None, db_path: Path = None):
        self.repo_path = repo_path or Path.cwd()
        self.db_path = db_path or (self.repo_path / ".quality_metrics.db")
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for storing metrics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    commit_hash TEXT,
                    branch TEXT,
                    lines_of_code INTEGER,
                    test_coverage REAL,
                    test_count INTEGER,
                    flake8_violations INTEGER,
                    mypy_errors INTEGER,
                    bandit_issues INTEGER,
                    complexity_score REAL,
                    maintainability_index REAL,
                    technical_debt_ratio REAL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_trends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    value REAL NOT NULL,
                    trend TEXT,
                    alert_level TEXT
                )
            """)
    
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
    
    def get_git_info(self) -> Dict[str, str]:
        """Get current Git commit and branch info."""
        commit_result = self.run_command(["git", "rev-parse", "HEAD"])
        branch_result = self.run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        
        return {
            "commit_hash": commit_result["stdout"].strip() if commit_result["success"] else "unknown",
            "branch": branch_result["stdout"].strip() if branch_result["success"] else "unknown"
        }
    
    def measure_lines_of_code(self) -> int:
        """Count lines of code in source files."""
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
        
        return total_loc
    
    def measure_test_coverage(self) -> Tuple[float, int]:
        """Measure test coverage and test count."""
        # Run pytest with coverage
        result = self.run_command([
            "pytest", "--cov=src", "--cov-report=term-missing", 
            "--co", "-q", "--tb=no"
        ])
        
        coverage = 0.0
        test_count = 0
        
        if result["success"]:
            # Extract coverage from output
            lines = result["stdout"].split('\n')
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
                
                # Count tests
                if 'test' in line.lower() and '::' in line:
                    test_count += 1
        
        return coverage, test_count
    
    def measure_flake8_violations(self) -> int:
        """Count Flake8 style violations."""
        result = self.run_command(["flake8", "src", "tests", "--count"])
        if result["success"]:
            try:
                return int(result["stdout"].strip())
            except ValueError:
                pass
        return 0
    
    def measure_mypy_errors(self) -> int:
        """Count MyPy type errors."""
        result = self.run_command(["mypy", "src", "--no-error-summary"])
        if not result["success"]:
            # Count errors in stderr
            error_lines = [line for line in result["stderr"].split('\n') if ': error:' in line]
            return len(error_lines)
        return 0
    
    def measure_bandit_issues(self) -> int:
        """Count Bandit security issues."""
        result = self.run_command(["bandit", "-r", "src", "-f", "json"])
        if result["success"]:
            try:
                bandit_data = json.loads(result["stdout"])
                return len(bandit_data.get("results", []))
            except json.JSONDecodeError:
                pass
        return 0
    
    def measure_complexity(self) -> float:
        """Measure cyclomatic complexity using radon."""
        result = self.run_command(["radon", "cc", "src", "-a", "-s"])
        if result["success"]:
            lines = result["stdout"].split('\n')
            for line in lines:
                if "Average complexity:" in line:
                    try:
                        return float(line.split(":")[-1].strip().split()[0])
                    except (ValueError, IndexError):
                        pass
        return 0.0
    
    def measure_maintainability(self) -> float:
        """Measure maintainability index using radon."""
        result = self.run_command(["radon", "mi", "src", "-s"])
        if result["success"]:
            lines = result["stdout"].split('\n')
            scores = []
            for line in lines:
                if " - " in line and "(" in line:
                    try:
                        score_part = line.split("(")[1].split(")")[0]
                        score = float(score_part)
                        scores.append(score)
                    except (ValueError, IndexError):
                        pass
            if scores:
                return sum(scores) / len(scores)
        return 0.0
    
    def calculate_technical_debt_ratio(self, metrics: Dict) -> float:
        """Calculate technical debt ratio based on various metrics."""
        # Simple formula: combination of violations and complexity
        violations = metrics.get("flake8_violations", 0) + metrics.get("mypy_errors", 0)
        loc = metrics.get("lines_of_code", 1)  # Avoid division by zero
        complexity = metrics.get("complexity_score", 1)
        
        # Technical debt ratio (lower is better)
        debt_ratio = (violations * complexity) / loc * 100
        return min(debt_ratio, 100.0)  # Cap at 100%
    
    def collect_metrics(self) -> Dict:
        """Collect all quality metrics."""
        print("📊 Collecting code quality metrics...")
        
        git_info = self.get_git_info()
        
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "commit_hash": git_info["commit_hash"],
            "branch": git_info["branch"],
            "lines_of_code": self.measure_lines_of_code(),
            "flake8_violations": self.measure_flake8_violations(),
            "mypy_errors": self.measure_mypy_errors(),
            "bandit_issues": self.measure_bandit_issues(),
            "complexity_score": self.measure_complexity(),
            "maintainability_index": self.measure_maintainability(),
        }
        
        # Measure test coverage (can be slow)
        coverage, test_count = self.measure_test_coverage()
        metrics["test_coverage"] = coverage
        metrics["test_count"] = test_count
        
        # Calculate derived metrics
        metrics["technical_debt_ratio"] = self.calculate_technical_debt_ratio(metrics)
        
        return metrics
    
    def store_metrics(self, metrics: Dict):
        """Store metrics in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO quality_metrics (
                    timestamp, commit_hash, branch, lines_of_code,
                    test_coverage, test_count, flake8_violations, mypy_errors,
                    bandit_issues, complexity_score, maintainability_index,
                    technical_debt_ratio
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics["timestamp"], metrics["commit_hash"], metrics["branch"],
                metrics["lines_of_code"], metrics["test_coverage"], metrics["test_count"],
                metrics["flake8_violations"], metrics["mypy_errors"], metrics["bandit_issues"],
                metrics["complexity_score"], metrics["maintainability_index"],
                metrics["technical_debt_ratio"]
            ))
    
    def get_historical_metrics(self, days: int = 30) -> List[Dict]:
        """Get historical metrics from database."""
        cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM quality_metrics 
                WHERE timestamp >= ? 
                ORDER BY timestamp DESC
            """, (cutoff_date,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def analyze_trends(self, metrics: Dict, historical: List[Dict]) -> Dict[str, Dict]:
        """Analyze trends in quality metrics."""
        if len(historical) < 2:
            return {}
        
        trends = {}
        
        # Key metrics to analyze
        key_metrics = [
            "test_coverage", "flake8_violations", "mypy_errors", 
            "complexity_score", "technical_debt_ratio", "lines_of_code"
        ]
        
        for metric in key_metrics:
            current_value = metrics.get(metric, 0)
            
            # Get recent values for trend analysis
            recent_values = [h.get(metric, 0) for h in historical[:10]]
            if not recent_values:
                continue
            
            avg_recent = sum(recent_values) / len(recent_values)
            
            # Calculate trend
            if current_value > avg_recent * 1.1:
                trend = "increasing"
            elif current_value < avg_recent * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
            
            # Determine alert level
            alert_level = self._determine_alert_level(metric, current_value, avg_recent)
            
            trends[metric] = {
                "current": current_value,
                "average": avg_recent,
                "trend": trend,
                "alert_level": alert_level
            }
        
        return trends
    
    def _determine_alert_level(self, metric: str, current: float, average: float) -> str:
        """Determine alert level for a metric."""
        # Define thresholds for different metrics
        thresholds = {
            "test_coverage": {"warning": 80, "critical": 70},
            "flake8_violations": {"warning": 10, "critical": 20},
            "mypy_errors": {"warning": 5, "critical": 15},
            "complexity_score": {"warning": 5, "critical": 10},
            "technical_debt_ratio": {"warning": 10, "critical": 20}
        }
        
        if metric not in thresholds:
            return "info"
        
        threshold = thresholds[metric]
        
        if metric == "test_coverage":
            # Lower is worse for coverage
            if current < threshold["critical"]:
                return "critical"
            elif current < threshold["warning"]:
                return "warning"
        else:
            # Higher is worse for violations/complexity
            if current > threshold["critical"]:
                return "critical"
            elif current > threshold["warning"]:
                return "warning"
        
        return "info"
    
    def generate_report(self, metrics: Dict, trends: Dict) -> str:
        """Generate a quality report."""
        lines = [
            "# Code Quality Report",
            f"Generated: {metrics['timestamp']}",
            f"Branch: {metrics['branch']}",
            f"Commit: {metrics['commit_hash'][:8]}",
            "",
            "## Current Metrics",
            f"- Lines of Code: {metrics['lines_of_code']:,}",
            f"- Test Coverage: {metrics['test_coverage']:.1f}%",
            f"- Test Count: {metrics['test_count']:,}",
            f"- Flake8 Violations: {metrics['flake8_violations']:,}",
            f"- MyPy Errors: {metrics['mypy_errors']:,}",
            f"- Bandit Issues: {metrics['bandit_issues']:,}",
            f"- Complexity Score: {metrics['complexity_score']:.2f}",
            f"- Maintainability Index: {metrics['maintainability_index']:.2f}",
            f"- Technical Debt Ratio: {metrics['technical_debt_ratio']:.2f}%",
            "",
            "## Trend Analysis"
        ]
        
        if trends:
            for metric, trend_data in trends.items():
                alert_emoji = {
                    "critical": "🔴",
                    "warning": "🟡",
                    "info": "🟢"
                }.get(trend_data["alert_level"], "ℹ️")
                
                trend_emoji = {
                    "increasing": "📈",
                    "decreasing": "📉",
                    "stable": "➡️"
                }.get(trend_data["trend"], "")
                
                lines.append(
                    f"- {metric.replace('_', ' ').title()}: "
                    f"{trend_data['current']:.2f} {trend_emoji} "
                    f"({trend_data['trend']}) {alert_emoji}"
                )
        else:
            lines.append("- Insufficient historical data for trend analysis")
        
        lines.extend([
            "",
            "## Quality Gates",
            f"- Test Coverage: {'✅' if metrics['test_coverage'] >= 80 else '❌'} (>= 80%)",
            f"- Code Style: {'✅' if metrics['flake8_violations'] == 0 else '❌'} (0 violations)",
            f"- Type Safety: {'✅' if metrics['mypy_errors'] == 0 else '❌'} (0 errors)",
            f"- Security: {'✅' if metrics['bandit_issues'] == 0 else '❌'} (0 issues)",
            f"- Complexity: {'✅' if metrics['complexity_score'] <= 5 else '❌'} (<= 5.0)",
            "",
            "## Recommendations"
        ])
        
        # Add recommendations based on metrics
        recommendations = []
        
        if metrics['test_coverage'] < 80:
            recommendations.append("- Increase test coverage to at least 80%")
        
        if metrics['flake8_violations'] > 0:
            recommendations.append("- Fix code style violations")
        
        if metrics['mypy_errors'] > 0:
            recommendations.append("- Resolve type checking errors")
        
        if metrics['complexity_score'] > 5:
            recommendations.append("- Refactor complex code to reduce complexity")
        
        if metrics['technical_debt_ratio'] > 10:
            recommendations.append("- Address technical debt accumulation")
        
        if not recommendations:
            recommendations.append("- Maintain current quality standards")
        
        lines.extend(recommendations)
        
        return '\n'.join(lines)
    
    def run_monitoring(self, generate_report: bool = False):
        """Run the complete quality monitoring workflow."""
        print("🔍 Starting code quality monitoring...")
        
        # Collect current metrics
        metrics = self.collect_metrics()
        
        # Store in database
        self.store_metrics(metrics)
        
        # Get historical data for trends
        historical = self.get_historical_metrics()
        
        # Analyze trends
        trends = self.analyze_trends(metrics, historical)
        
        if generate_report:
            report = self.generate_report(metrics, trends)
            print("\n" + report)
        
        # Generate dashboard data if requested
        dashboard_data = {
            "current_metrics": metrics,
            "trends": trends,
            "historical": historical[-30:]  # Last 30 entries
        }
        
        return dashboard_data


def main():
    parser = argparse.ArgumentParser(description="Monitor code quality metrics")
    parser.add_argument("--report", action="store_true",
                        help="Generate and display quality report")
    parser.add_argument("--dashboard", action="store_true",
                        help="Generate dashboard data JSON")
    parser.add_argument("--output", help="Output file for dashboard data")
    parser.add_argument("--repo-path", help="Repository path (default: current directory)")
    parser.add_argument("--db-path", help="Database path for storing metrics")
    
    args = parser.parse_args()
    
    repo_path = Path(args.repo_path) if args.repo_path else Path.cwd()
    db_path = Path(args.db_path) if args.db_path else None
    
    monitor = CodeQualityMonitor(repo_path, db_path)
    
    try:
        dashboard_data = monitor.run_monitoring(generate_report=args.report)
        
        if args.dashboard or args.output:
            output_data = json.dumps(dashboard_data, indent=2)
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(output_data)
                print(f"\n📊 Dashboard data saved to: {args.output}")
            else:
                print("\n📊 Dashboard Data:")
                print(output_data)
    
    except Exception as e:
        print(f"❌ Error in quality monitoring: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()