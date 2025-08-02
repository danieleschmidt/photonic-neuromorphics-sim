#!/usr/bin/env python3
"""
Code quality monitoring system for photonic neuromorphics simulation platform.
Tracks code quality metrics and generates reports.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple

import matplotlib.pyplot as plt
import pandas as pd


class CodeQualityMonitor:
    """Code quality monitoring and reporting system."""
    
    def __init__(self, repo_path: str = "."):
        """Initialize code quality monitor."""
        self.repo_path = Path(repo_path)
        self.metrics_history = []
        self.thresholds = {
            "test_coverage": {"good": 90, "warning": 70},
            "complexity": {"good": 5, "warning": 10},
            "duplication": {"good": 3, "warning": 10},
            "maintainability": {"good": 80, "warning": 60},
            "technical_debt": {"good": 0, "warning": 240}  # hours
        }
        
    def collect_quality_metrics(self) -> Dict[str, Any]:
        """Collect current code quality metrics."""
        print("Collecting code quality metrics...")
        
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "git_commit": self._get_current_commit(),
            "metrics": {}
        }
        
        # Test coverage
        metrics["metrics"]["test_coverage"] = self._measure_test_coverage()
        
        # Code complexity
        metrics["metrics"]["complexity"] = self._measure_complexity()
        
        # Code duplication
        metrics["metrics"]["duplication"] = self._measure_duplication()
        
        # Maintainability index
        metrics["metrics"]["maintainability"] = self._measure_maintainability()
        
        # Technical debt
        metrics["metrics"]["technical_debt"] = self._estimate_technical_debt()
        
        # Lines of code
        metrics["metrics"]["lines_of_code"] = self._count_lines_of_code()
        
        # Code quality score
        metrics["metrics"]["quality_score"] = self._calculate_quality_score(metrics["metrics"])
        
        return metrics
    
    def _get_current_commit(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"
    
    def _measure_test_coverage(self) -> Dict[str, Any]:
        """Measure test coverage."""
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "--cov=src", "--cov-report=json", "--tb=no"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            coverage_file = self.repo_path / "coverage.json"
            if result.returncode == 0 and coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                
                total_coverage = coverage_data["totals"]["percent_covered"]
                
                return {
                    "total_coverage": round(total_coverage, 2),
                    "covered_lines": coverage_data["totals"]["covered_lines"],
                    "missing_lines": coverage_data["totals"]["missing_lines"],
                    "total_lines": coverage_data["totals"]["num_statements"],
                    "status": self._get_status("test_coverage", total_coverage)
                }
            else:
                return {
                    "total_coverage": 0,
                    "status": "error",
                    "error": "Could not measure test coverage"
                }
        except Exception as e:
            return {
                "total_coverage": 0,
                "status": "error",
                "error": str(e)
            }
    
    def _measure_complexity(self) -> Dict[str, Any]:
        """Measure code complexity using radon."""
        try:
            result = subprocess.run(
                ["radon", "cc", "src/", "--json"],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                complexity_data = json.loads(result.stdout)
                
                all_complexities = []
                file_complexities = {}
                
                for file_path, file_data in complexity_data.items():
                    file_complexities[file_path] = []
                    for item in file_data:
                        if isinstance(item, dict) and 'complexity' in item:
                            complexity = item['complexity']
                            all_complexities.append(complexity)
                            file_complexities[file_path].append({
                                "name": item.get('name', 'unknown'),
                                "complexity": complexity,
                                "type": item.get('type', 'unknown')
                            })
                
                avg_complexity = sum(all_complexities) / len(all_complexities) if all_complexities else 0
                max_complexity = max(all_complexities) if all_complexities else 0
                
                return {
                    "average_complexity": round(avg_complexity, 2),
                    "max_complexity": max_complexity,
                    "total_functions": len(all_complexities),
                    "high_complexity_functions": len([c for c in all_complexities if c > 10]),
                    "file_complexities": file_complexities,
                    "status": self._get_status("complexity", avg_complexity)
                }
            else:
                return {
                    "average_complexity": 0,
                    "status": "error",
                    "error": "Could not measure complexity"
                }
        except Exception as e:
            return {
                "average_complexity": 0,
                "status": "error",
                "error": str(e)
            }
    
    def _measure_duplication(self) -> Dict[str, Any]:
        """Measure code duplication."""
        try:
            # Using simple approach - would integrate with tools like sonarqube in production
            python_files = list(self.repo_path.glob("src/**/*.py"))
            
            if not python_files:
                return {"duplication_percentage": 0, "status": "good"}
            
            # Simple duplication detection based on identical lines
            line_counts = {}
            total_lines = 0
            duplicate_lines = 0
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#') and len(line) > 10:
                            total_lines += 1
                            line_counts[line] = line_counts.get(line, 0) + 1
                            if line_counts[line] > 1:
                                duplicate_lines += 1
                except Exception:
                    continue
            
            duplication_percentage = (duplicate_lines / total_lines * 100) if total_lines > 0 else 0
            
            return {
                "duplication_percentage": round(duplication_percentage, 2),
                "duplicate_lines": duplicate_lines,
                "total_lines": total_lines,
                "status": self._get_status("duplication", duplication_percentage)
            }
            
        except Exception as e:
            return {
                "duplication_percentage": 0,
                "status": "error",
                "error": str(e)
            }
    
    def _measure_maintainability(self) -> Dict[str, Any]:
        """Measure maintainability index using radon."""
        try:
            result = subprocess.run(
                ["radon", "mi", "src/", "--json"],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                mi_data = json.loads(result.stdout)
                
                mi_scores = []
                file_scores = {}
                
                for file_path, file_data in mi_data.items():
                    if isinstance(file_data, dict) and 'mi' in file_data:
                        mi_score = file_data['mi']
                        mi_scores.append(mi_score)
                        file_scores[file_path] = {
                            "maintainability_index": mi_score,
                            "rank": file_data.get('rank', 'unknown')
                        }
                
                avg_mi = sum(mi_scores) / len(mi_scores) if mi_scores else 0
                min_mi = min(mi_scores) if mi_scores else 0
                
                return {
                    "average_maintainability": round(avg_mi, 2),
                    "min_maintainability": round(min_mi, 2),
                    "file_scores": file_scores,
                    "low_maintainability_files": len([mi for mi in mi_scores if mi < 50]),
                    "status": self._get_status("maintainability", avg_mi)
                }
            else:
                return {
                    "average_maintainability": 0,
                    "status": "error",
                    "error": "Could not measure maintainability"
                }
        except Exception as e:
            return {
                "average_maintainability": 0,
                "status": "error",
                "error": str(e)
            }
    
    def _estimate_technical_debt(self) -> Dict[str, Any]:
        """Estimate technical debt in hours."""
        try:
            # Simplified technical debt calculation
            debt_hours = 0
            debt_sources = {}
            
            # Debt from high complexity functions
            complexity_metrics = self._measure_complexity()
            if complexity_metrics.get("high_complexity_functions", 0) > 0:
                complexity_debt = complexity_metrics["high_complexity_functions"] * 2  # 2 hours per function
                debt_hours += complexity_debt
                debt_sources["high_complexity"] = complexity_debt
            
            # Debt from low test coverage
            coverage_metrics = self._measure_test_coverage()
            coverage = coverage_metrics.get("total_coverage", 0)
            if coverage < 80:
                coverage_debt = (80 - coverage) * 0.5  # 0.5 hours per % below 80%
                debt_hours += coverage_debt
                debt_sources["low_coverage"] = coverage_debt
            
            # Debt from code duplication
            duplication_metrics = self._measure_duplication()
            duplication = duplication_metrics.get("duplication_percentage", 0)
            if duplication > 5:
                duplication_debt = (duplication - 5) * 1  # 1 hour per % above 5%
                debt_hours += duplication_debt
                debt_sources["duplication"] = duplication_debt
            
            # Debt from TODO comments
            todo_count = self._count_todo_comments()
            if todo_count > 0:
                todo_debt = todo_count * 0.5  # 0.5 hours per TODO
                debt_hours += todo_debt
                debt_sources["todos"] = todo_debt
            
            return {
                "total_hours": round(debt_hours, 1),
                "debt_sources": debt_sources,
                "todo_count": todo_count,
                "status": self._get_status("technical_debt", debt_hours)
            }
            
        except Exception as e:
            return {
                "total_hours": 0,
                "status": "error",
                "error": str(e)
            }
    
    def _count_lines_of_code(self) -> Dict[str, Any]:
        """Count lines of code."""
        try:
            result = subprocess.run(
                ["cloc", "src/", "--json"],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                cloc_data = json.loads(result.stdout)
                
                total_lines = cloc_data.get("SUM", {}).get("code", 0)
                blank_lines = cloc_data.get("SUM", {}).get("blank", 0)
                comment_lines = cloc_data.get("SUM", {}).get("comment", 0)
                
                return {
                    "total_lines": total_lines,
                    "blank_lines": blank_lines,
                    "comment_lines": comment_lines,
                    "comment_ratio": round((comment_lines / total_lines * 100), 2) if total_lines > 0 else 0,
                    "languages": {k: v for k, v in cloc_data.items() if k not in ["header", "SUM"]}
                }
            else:
                # Fallback to simple line counting
                python_files = list(self.repo_path.glob("src/**/*.py"))
                total_lines = 0
                
                for py_file in python_files:
                    try:
                        with open(py_file, 'r', encoding='utf-8') as f:
                            total_lines += len(f.readlines())
                    except Exception:
                        continue
                
                return {
                    "total_lines": total_lines,
                    "blank_lines": 0,
                    "comment_lines": 0,
                    "comment_ratio": 0
                }
                
        except Exception as e:
            return {
                "total_lines": 0,
                "error": str(e)
            }
    
    def _count_todo_comments(self) -> int:
        """Count TODO comments in code."""
        try:
            python_files = list(self.repo_path.glob("src/**/*.py"))
            todo_count = 0
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        todo_count += content.upper().count("TODO")
                        todo_count += content.upper().count("FIXME")
                        todo_count += content.upper().count("HACK")
                except Exception:
                    continue
            
            return todo_count
        except Exception:
            return 0
    
    def _calculate_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score (0-100)."""
        try:
            weights = {
                "test_coverage": 0.3,
                "complexity": 0.25,
                "maintainability": 0.2,
                "duplication": 0.15,
                "technical_debt": 0.1
            }
            
            scores = {}
            
            # Test coverage score
            coverage = metrics.get("test_coverage", {}).get("total_coverage", 0)
            scores["test_coverage"] = min(100, coverage)
            
            # Complexity score (inverted - lower is better)
            complexity = metrics.get("complexity", {}).get("average_complexity", 10)
            scores["complexity"] = max(0, 100 - (complexity * 10))
            
            # Maintainability score
            maintainability = metrics.get("maintainability", {}).get("average_maintainability", 50)
            scores["maintainability"] = maintainability
            
            # Duplication score (inverted - lower is better)
            duplication = metrics.get("duplication", {}).get("duplication_percentage", 0)
            scores["duplication"] = max(0, 100 - (duplication * 5))
            
            # Technical debt score (inverted - lower is better)
            debt_hours = metrics.get("technical_debt", {}).get("total_hours", 0)
            scores["technical_debt"] = max(0, 100 - (debt_hours / 10))
            
            # Calculate weighted average
            total_score = sum(scores[metric] * weights[metric] for metric in weights.keys() if metric in scores)
            
            return round(total_score, 1)
            
        except Exception:
            return 0.0
    
    def _get_status(self, metric: str, value: float) -> str:
        """Get status (good/warning/poor) for a metric value."""
        thresholds = self.thresholds.get(metric, {})
        
        if metric in ["complexity", "duplication", "technical_debt"]:
            # Lower is better
            if value <= thresholds.get("good", 0):
                return "good"
            elif value <= thresholds.get("warning", 10):
                return "warning"
            else:
                return "poor"
        else:
            # Higher is better
            if value >= thresholds.get("good", 80):
                return "good"
            elif value >= thresholds.get("warning", 60):
                return "warning"
            else:
                return "poor"
    
    def save_metrics(self, metrics: Dict[str, Any], filename: str = "quality_metrics.json"):
        """Save metrics to file."""
        output_path = self.repo_path / filename
        
        # Load existing metrics if file exists
        if output_path.exists():
            try:
                with open(output_path, 'r') as f:
                    existing_data = json.load(f)
                    if isinstance(existing_data, list):
                        self.metrics_history = existing_data
                    else:
                        self.metrics_history = [existing_data]
            except Exception:
                self.metrics_history = []
        else:
            self.metrics_history = []
        
        # Add new metrics
        self.metrics_history.append(metrics)
        
        # Keep only last 100 entries
        self.metrics_history = self.metrics_history[-100:]
        
        # Save updated metrics
        with open(output_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        print(f"Metrics saved to {output_path}")
    
    def generate_trend_analysis(self, days: int = 30) -> Dict[str, Any]:
        """Generate trend analysis for the last N days."""
        if not self.metrics_history:
            return {"error": "No historical metrics available"}
        
        # Filter metrics for the specified time period
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_metrics = []
        
        for entry in self.metrics_history:
            try:
                entry_date = datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00"))
                if entry_date >= cutoff_date:
                    recent_metrics.append(entry)
            except Exception:
                continue
        
        if len(recent_metrics) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        # Calculate trends
        trends = {}
        metric_keys = [
            "test_coverage.total_coverage",
            "complexity.average_complexity", 
            "maintainability.average_maintainability",
            "duplication.duplication_percentage",
            "technical_debt.total_hours",
            "quality_score"
        ]
        
        for key in metric_keys:
            values = []
            timestamps = []
            
            for entry in recent_metrics:
                try:
                    keys = key.split(".")
                    value = entry["metrics"]
                    for k in keys:
                        value = value[k]
                    values.append(value)
                    timestamps.append(datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00")))
                except Exception:
                    continue
            
            if len(values) >= 2:
                # Calculate trend (simple linear regression slope)
                n = len(values)
                sum_x = sum(range(n))
                sum_y = sum(values)
                sum_xy = sum(i * values[i] for i in range(n))
                sum_x2 = sum(i * i for i in range(n))
                
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if (n * sum_x2 - sum_x * sum_x) != 0 else 0
                
                trends[key] = {
                    "slope": round(slope, 4),
                    "direction": "improving" if slope > 0.01 else "declining" if slope < -0.01 else "stable",
                    "current_value": values[-1],
                    "previous_value": values[0],
                    "change": round(values[-1] - values[0], 2),
                    "data_points": len(values)
                }
        
        return {
            "period_days": days,
            "data_points": len(recent_metrics),
            "trends": trends,
            "summary": self._generate_trend_summary(trends)
        }
    
    def _generate_trend_summary(self, trends: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of trends."""
        improving = []
        declining = []
        stable = []
        
        for metric, trend_data in trends.items():
            direction = trend_data["direction"]
            if direction == "improving":
                improving.append(metric)
            elif direction == "declining":
                declining.append(metric)
            else:
                stable.append(metric)
        
        return {
            "improving_metrics": improving,
            "declining_metrics": declining,
            "stable_metrics": stable,
            "overall_trend": "improving" if len(improving) > len(declining) else "declining" if len(declining) > len(improving) else "stable"
        }
    
    def generate_quality_report(self) -> str:
        """Generate a comprehensive quality report."""
        if not self.metrics_history:
            return "No metrics data available for report generation."
        
        latest_metrics = self.metrics_history[-1]
        metrics = latest_metrics["metrics"]
        
        report = f"# Code Quality Report\n\n"
        report += f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
        report += f"**Commit:** {latest_metrics.get('git_commit', 'unknown')[:8]}\n"
        report += f"**Overall Quality Score:** {metrics.get('quality_score', 0)}/100\n\n"
        
        # Test Coverage
        coverage = metrics.get("test_coverage", {})
        report += f"## 🧪 Test Coverage\n\n"
        report += f"- **Total Coverage:** {coverage.get('total_coverage', 0)}%\n"
        report += f"- **Covered Lines:** {coverage.get('covered_lines', 0)}\n"
        report += f"- **Missing Lines:** {coverage.get('missing_lines', 0)}\n"
        report += f"- **Status:** {coverage.get('status', 'unknown').title()}\n\n"
        
        # Code Complexity
        complexity = metrics.get("complexity", {})
        report += f"## 🔄 Code Complexity\n\n"
        report += f"- **Average Complexity:** {complexity.get('average_complexity', 0)}\n"
        report += f"- **Max Complexity:** {complexity.get('max_complexity', 0)}\n"
        report += f"- **High Complexity Functions:** {complexity.get('high_complexity_functions', 0)}\n"
        report += f"- **Status:** {complexity.get('status', 'unknown').title()}\n\n"
        
        # Maintainability
        maintainability = metrics.get("maintainability", {})
        report += f"## 🔧 Maintainability\n\n"
        report += f"- **Average Maintainability Index:** {maintainability.get('average_maintainability', 0)}\n"
        report += f"- **Minimum Maintainability Index:** {maintainability.get('min_maintainability', 0)}\n"
        report += f"- **Low Maintainability Files:** {maintainability.get('low_maintainability_files', 0)}\n"
        report += f"- **Status:** {maintainability.get('status', 'unknown').title()}\n\n"
        
        # Code Duplication
        duplication = metrics.get("duplication", {})
        report += f"## 📋 Code Duplication\n\n"
        report += f"- **Duplication Percentage:** {duplication.get('duplication_percentage', 0)}%\n"
        report += f"- **Duplicate Lines:** {duplication.get('duplicate_lines', 0)}\n"
        report += f"- **Status:** {duplication.get('status', 'unknown').title()}\n\n"
        
        # Technical Debt
        debt = metrics.get("technical_debt", {})
        report += f"## 💳 Technical Debt\n\n"
        report += f"- **Total Debt:** {debt.get('total_hours', 0)} hours\n"
        report += f"- **TODO Comments:** {debt.get('todo_count', 0)}\n"
        if debt.get('debt_sources'):
            report += f"- **Debt Sources:**\n"
            for source, hours in debt['debt_sources'].items():
                report += f"  - {source.replace('_', ' ').title()}: {hours} hours\n"
        report += f"- **Status:** {debt.get('status', 'unknown').title()}\n\n"
        
        # Lines of Code
        loc = metrics.get("lines_of_code", {})
        report += f"## 📏 Lines of Code\n\n"
        report += f"- **Total Lines:** {loc.get('total_lines', 0)}\n"
        report += f"- **Comment Ratio:** {loc.get('comment_ratio', 0)}%\n\n"
        
        # Trends (if available)
        if len(self.metrics_history) > 1:
            trends = self.generate_trend_analysis(30)
            if "trends" in trends:
                report += f"## 📈 30-Day Trends\n\n"
                trend_summary = trends.get("summary", {})
                report += f"- **Overall Trend:** {trend_summary.get('overall_trend', 'unknown').title()}\n"
                
                if trend_summary.get("improving_metrics"):
                    report += f"- **Improving:** {', '.join(trend_summary['improving_metrics'])}\n"
                if trend_summary.get("declining_metrics"):
                    report += f"- **Declining:** {', '.join(trend_summary['declining_metrics'])}\n"
                
                report += "\n"
        
        return report
    
    def create_quality_dashboard(self, output_file: str = "quality_dashboard.png"):
        """Create a visual quality dashboard."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            if not self.metrics_history:
                print("No metrics data available for dashboard")
                return
            
            latest_metrics = self.metrics_history[-1]["metrics"]
            
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Code Quality Dashboard', fontsize=16, fontweight='bold')
            
            # Quality Score Gauge
            score = latest_metrics.get("quality_score", 0)
            self._create_gauge(ax1, score, "Overall Quality Score", 0, 100)
            
            # Test Coverage
            coverage = latest_metrics.get("test_coverage", {}).get("total_coverage", 0)
            self._create_gauge(ax2, coverage, "Test Coverage (%)", 0, 100)
            
            # Complexity vs Maintainability
            complexity = latest_metrics.get("complexity", {}).get("average_complexity", 0)
            maintainability = latest_metrics.get("maintainability", {}).get("average_maintainability", 0)
            ax3.scatter([complexity], [maintainability], s=100, c='blue', alpha=0.7)
            ax3.set_xlabel('Average Complexity')
            ax3.set_ylabel('Maintainability Index')
            ax3.set_title('Complexity vs Maintainability')
            ax3.grid(True, alpha=0.3)
            
            # Trends (if available)
            if len(self.metrics_history) > 1:
                dates = []
                scores = []
                
                for entry in self.metrics_history[-30:]:  # Last 30 entries
                    try:
                        dates.append(datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00")))
                        scores.append(entry["metrics"].get("quality_score", 0))
                    except Exception:
                        continue
                
                if dates and scores:
                    ax4.plot(dates, scores, marker='o', linewidth=2, markersize=4)
                    ax4.set_title('Quality Score Trend')
                    ax4.set_ylabel('Quality Score')
                    ax4.grid(True, alpha=0.3)
                    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Quality dashboard saved to {output_file}")
            
        except ImportError:
            print("Matplotlib not available, skipping dashboard creation")
        except Exception as e:
            print(f"Error creating dashboard: {e}")
    
    def _create_gauge(self, ax, value, title, min_val, max_val):
        """Create a gauge chart."""
        # Normalize value
        norm_value = (value - min_val) / (max_val - min_val)
        
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        # Background
        ax.plot(theta, r, 'lightgray', linewidth=10)
        
        # Value arc
        value_theta = theta[:int(norm_value * len(theta))]
        value_r = r[:len(value_theta)]
        
        # Color based on value
        if norm_value >= 0.8:
            color = 'green'
        elif norm_value >= 0.6:
            color = 'orange'
        else:
            color = 'red'
        
        ax.plot(value_theta, value_r, color, linewidth=10)
        
        # Add value text
        ax.text(0, 0, f'{value:.1f}', ha='center', va='center', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.2)
        ax.set_xlim(-0.2, np.pi + 0.2)
        ax.set_title(title, fontweight='bold')
        ax.axis('off')


def main():
    """Main function for code quality monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor code quality metrics")
    parser.add_argument("--repo-path", default=".", help="Path to repository")
    parser.add_argument("--output", default="quality_metrics.json", help="Output file for metrics")
    parser.add_argument("--report", action="store_true", help="Generate quality report")
    parser.add_argument("--dashboard", action="store_true", help="Create visual dashboard")
    parser.add_argument("--trends", type=int, default=30, help="Analyze trends for N days")
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = CodeQualityMonitor(args.repo_path)
    
    # Collect metrics
    metrics = monitor.collect_quality_metrics()
    
    # Save metrics
    monitor.save_metrics(metrics, args.output)
    
    # Print summary
    quality_score = metrics["metrics"].get("quality_score", 0)
    print(f"\nCode Quality Summary:")
    print(f"Overall Score: {quality_score}/100")
    
    for metric_name, metric_data in metrics["metrics"].items():
        if isinstance(metric_data, dict) and "status" in metric_data:
            status_emoji = "✅" if metric_data["status"] == "good" else "⚠️" if metric_data["status"] == "warning" else "❌"
            print(f"{status_emoji} {metric_name.replace('_', ' ').title()}: {metric_data['status']}")
    
    # Generate report if requested
    if args.report:
        report = monitor.generate_quality_report()
        report_file = "quality_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\nQuality report saved to {report_file}")
    
    # Create dashboard if requested
    if args.dashboard:
        monitor.create_quality_dashboard()
    
    # Show trends if requested
    if args.trends:
        trends = monitor.generate_trend_analysis(args.trends)
        if "trends" in trends:
            print(f"\n{args.trends}-Day Trends:")
            summary = trends["summary"]
            print(f"Overall trend: {summary.get('overall_trend', 'unknown')}")
            
            if summary.get("improving_metrics"):
                print(f"Improving: {', '.join(summary['improving_metrics'])}")
            if summary.get("declining_metrics"):
                print(f"Declining: {', '.join(summary['declining_metrics'])}")


if __name__ == "__main__":
    main()