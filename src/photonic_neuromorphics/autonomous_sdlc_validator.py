#!/usr/bin/env python3
"""
Autonomous SDLC Validation Framework

A comprehensive dependency-free validation system for photonic neuromorphics simulation
that provides intelligent code analysis, validation, and enhancement recommendations.
"""

import ast
import os
import re
import sys
import json
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from collections import defaultdict


@dataclass
class ValidationResult:
    """Represents a validation result with severity and recommendations."""
    category: str
    severity: str  # 'error', 'warning', 'info', 'success'
    message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    recommendation: Optional[str] = None
    performance_impact: Optional[str] = None


@dataclass
class MetricResult:
    """Represents a code metric measurement."""
    metric_name: str
    value: float
    unit: str
    threshold: Optional[float] = None
    status: str = "unknown"  # 'pass', 'fail', 'warning'


@dataclass
class ArchitectureAnalysis:
    """Architecture analysis results."""
    total_modules: int
    total_classes: int
    total_functions: int
    complexity_score: float
    maintainability_index: float
    test_coverage_estimate: float
    design_patterns: List[str]
    architecture_quality: str


class CodeComplexityAnalyzer:
    """Analyzes code complexity using AST parsing."""

    def __init__(self):
        self.complexity_scores = {}
        self.function_metrics = {}

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze complexity of a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source)
            
            metrics = {
                'cyclomatic_complexity': self._calculate_cyclomatic_complexity(tree),
                'cognitive_complexity': self._calculate_cognitive_complexity(tree),
                'lines_of_code': len(source.splitlines()),
                'comment_ratio': self._calculate_comment_ratio(source),
                'function_count': len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
                'class_count': len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
                'import_count': len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]),
            }
            
            return metrics
            
        except Exception as e:
            return {'error': str(e)}

    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.With):
                complexity += 1
            elif isinstance(node, ast.Assert):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
                
        return complexity

    def _calculate_cognitive_complexity(self, tree: ast.AST) -> int:
        """Calculate cognitive complexity."""
        complexity = 0
        nesting_level = 0
        
        def calculate_nesting(node, level=0):
            nonlocal complexity, nesting_level
            
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1 + level
                level += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1 + level
                level += 1
            elif isinstance(node, ast.With):
                complexity += 1 + level
                level += 1
            
            for child in ast.iter_child_nodes(node):
                calculate_nesting(child, level)
        
        calculate_nesting(tree)
        return complexity

    def _calculate_comment_ratio(self, source: str) -> float:
        """Calculate ratio of comment lines to total lines."""
        lines = source.splitlines()
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        docstring_lines = source.count('"""') * 3  # Rough estimate
        
        total_comment_lines = comment_lines + docstring_lines
        total_lines = len(lines)
        
        return total_comment_lines / total_lines if total_lines > 0 else 0.0


class SecurityAnalyzer:
    """Analyzes code for security vulnerabilities."""

    def __init__(self):
        self.security_patterns = {
            'command_injection': [
                r'os\.system\(',
                r'subprocess\.call\(',
                r'subprocess\.run\(',
                r'eval\(',
                r'exec\(',
            ],
            'hardcoded_secrets': [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']',
            ],
            'sql_injection': [
                r'\.execute\(["\'][^"\']*%[^"\']*["\']',
                r'\.execute\(["\'][^"\']*\+[^"\']*["\']',
            ],
            'unsafe_imports': [
                r'import\s+pickle',
                r'from\s+pickle\s+import',
                r'import\s+marshal',
            ]
        }

    def analyze_file(self, file_path: str) -> List[ValidationResult]:
        """Analyze a file for security issues."""
        results = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.splitlines()
            
            for category, patterns in self.security_patterns.items():
                for pattern in patterns:
                    for line_num, line in enumerate(lines, 1):
                        if re.search(pattern, line, re.IGNORECASE):
                            results.append(ValidationResult(
                                category='security',
                                severity='warning',
                                message=f'Potential {category.replace("_", " ")} vulnerability detected',
                                file_path=file_path,
                                line_number=line_num,
                                recommendation=f'Review and sanitize {category.replace("_", " ")} usage'
                            ))
                            
        except Exception as e:
            results.append(ValidationResult(
                category='security',
                severity='error',
                message=f'Failed to analyze file: {str(e)}',
                file_path=file_path
            ))
            
        return results


class PerformanceAnalyzer:
    """Analyzes code for performance issues."""

    def __init__(self):
        self.performance_patterns = {
            'inefficient_loops': [
                r'for\s+\w+\s+in\s+range\(len\(',
                r'for\s+\w+\s+in\s+\w+\.keys\(\):',
            ],
            'string_concatenation': [
                r'\s*\+\s*["\'][^"\']*["\']',
                r'["\'][^"\']*["\']\s*\+',
            ],
            'global_variables': [
                r'global\s+\w+',
            ],
            'nested_loops': [
                r'for\s+.*:\s*\n.*for\s+.*:',
            ]
        }

    def analyze_file(self, file_path: str) -> List[ValidationResult]:
        """Analyze a file for performance issues."""
        results = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.splitlines()
            
            for category, patterns in self.performance_patterns.items():
                for pattern in patterns:
                    for line_num, line in enumerate(lines, 1):
                        if re.search(pattern, line):
                            severity = 'warning' if 'global' in category else 'info'
                            results.append(ValidationResult(
                                category='performance',
                                severity=severity,
                                message=f'Potential performance issue: {category.replace("_", " ")}',
                                file_path=file_path,
                                line_number=line_num,
                                recommendation=self._get_performance_recommendation(category),
                                performance_impact='medium'
                            ))
                            
        except Exception as e:
            results.append(ValidationResult(
                category='performance',
                severity='error',
                message=f'Failed to analyze file: {str(e)}',
                file_path=file_path
            ))
            
        return results

    def _get_performance_recommendation(self, category: str) -> str:
        """Get performance recommendation for a specific issue."""
        recommendations = {
            'inefficient_loops': 'Consider using enumerate() or direct iteration',
            'string_concatenation': 'Use f-strings or join() for better performance',
            'global_variables': 'Minimize global variable usage for better performance',
            'nested_loops': 'Consider optimizing nested loops or using vectorized operations'
        }
        return recommendations.get(category, 'Review for optimization opportunities')


class TestabilityAnalyzer:
    """Analyzes code testability and test coverage."""

    def analyze_project(self, project_path: str) -> Dict[str, Any]:
        """Analyze project testability."""
        test_files = []
        source_files = []
        
        for root, dirs, files in os.walk(project_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    if 'test' in file.lower() or 'test' in root.lower():
                        test_files.append(file_path)
                    elif not file.startswith('__'):
                        source_files.append(file_path)
        
        test_coverage_estimate = len(test_files) / len(source_files) if source_files else 0
        
        return {
            'test_files_count': len(test_files),
            'source_files_count': len(source_files),
            'test_coverage_estimate': test_coverage_estimate,
            'testability_score': min(test_coverage_estimate * 2, 1.0),
            'test_files': test_files,
            'source_files': source_files
        }


class AutonomousSDLCValidator:
    """Main autonomous SDLC validation framework."""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.complexity_analyzer = CodeComplexityAnalyzer()
        self.security_analyzer = SecurityAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer()
        self.testability_analyzer = TestabilityAnalyzer()
        
        self.validation_results = []
        self.metrics = []
        self.start_time = time.time()

    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive SDLC validation."""
        print("🚀 Starting Autonomous SDLC Validation...")
        
        validation_report = {
            'timestamp': time.time(),
            'project_path': str(self.project_path),
            'validation_results': [],
            'metrics': [],
            'architecture_analysis': None,
            'recommendations': [],
            'overall_score': 0.0,
            'execution_time': 0.0
        }
        
        try:
            # Architecture Analysis
            print("📊 Analyzing architecture...")
            validation_report['architecture_analysis'] = self._analyze_architecture()
            
            # Code Quality Analysis
            print("🔍 Analyzing code quality...")
            self._analyze_code_quality()
            
            # Security Analysis
            print("🛡️ Analyzing security...")
            self._analyze_security()
            
            # Performance Analysis
            print("⚡ Analyzing performance...")
            self._analyze_performance()
            
            # Testability Analysis
            print("🧪 Analyzing testability...")
            self._analyze_testability()
            
            # Generate recommendations
            print("💡 Generating recommendations...")
            validation_report['recommendations'] = self._generate_recommendations()
            
            # Calculate overall score
            validation_report['overall_score'] = self._calculate_overall_score()
            
            validation_report['validation_results'] = [asdict(r) for r in self.validation_results]
            validation_report['metrics'] = [asdict(m) for m in self.metrics]
            validation_report['execution_time'] = time.time() - self.start_time
            
            print(f"✅ Validation completed in {validation_report['execution_time']:.2f}s")
            
        except Exception as e:
            validation_report['error'] = str(e)
            print(f"❌ Validation failed: {str(e)}")
            
        return validation_report

    def _analyze_architecture(self) -> ArchitectureAnalysis:
        """Analyze project architecture."""
        total_modules = 0
        total_classes = 0
        total_functions = 0
        total_complexity = 0
        
        design_patterns = set()
        
        for py_file in self.project_path.rglob('*.py'):
            if py_file.name.startswith('__'):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                tree = ast.parse(source)
                total_modules += 1
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        total_classes += 1
                        # Detect design patterns
                        if 'Factory' in node.name:
                            design_patterns.add('Factory Pattern')
                        elif 'Builder' in node.name:
                            design_patterns.add('Builder Pattern')
                        elif 'Singleton' in node.name:
                            design_patterns.add('Singleton Pattern')
                        elif 'Observer' in node.name:
                            design_patterns.add('Observer Pattern')
                    elif isinstance(node, ast.FunctionDef):
                        total_functions += 1
                
                metrics = self.complexity_analyzer.analyze_file(str(py_file))
                if 'cyclomatic_complexity' in metrics:
                    total_complexity += metrics['cyclomatic_complexity']
                    
            except Exception:
                continue
        
        avg_complexity = total_complexity / total_modules if total_modules > 0 else 0
        maintainability_index = max(0, 100 - avg_complexity * 2)
        
        testability_info = self.testability_analyzer.analyze_project(str(self.project_path))
        
        # Determine architecture quality
        quality_score = (
            min(maintainability_index / 100, 1.0) * 0.4 +
            min(testability_info['testability_score'], 1.0) * 0.3 +
            min(len(design_patterns) / 5, 1.0) * 0.3
        )
        
        if quality_score > 0.8:
            architecture_quality = "Excellent"
        elif quality_score > 0.6:
            architecture_quality = "Good"
        elif quality_score > 0.4:
            architecture_quality = "Average"
        else:
            architecture_quality = "Needs Improvement"
        
        return ArchitectureAnalysis(
            total_modules=total_modules,
            total_classes=total_classes,
            total_functions=total_functions,
            complexity_score=avg_complexity,
            maintainability_index=maintainability_index,
            test_coverage_estimate=testability_info['test_coverage_estimate'],
            design_patterns=list(design_patterns),
            architecture_quality=architecture_quality
        )

    def _analyze_code_quality(self):
        """Analyze code quality metrics."""
        total_complexity = 0
        total_files = 0
        
        for py_file in self.project_path.rglob('*.py'):
            if py_file.name.startswith('__'):
                continue
                
            metrics = self.complexity_analyzer.analyze_file(str(py_file))
            
            if 'error' not in metrics:
                total_files += 1
                complexity = metrics.get('cyclomatic_complexity', 0)
                total_complexity += complexity
                
                # Check complexity thresholds
                if complexity > 20:
                    self.validation_results.append(ValidationResult(
                        category='code_quality',
                        severity='error',
                        message=f'High cyclomatic complexity ({complexity})',
                        file_path=str(py_file),
                        recommendation='Break down complex functions into smaller ones'
                    ))
                elif complexity > 10:
                    self.validation_results.append(ValidationResult(
                        category='code_quality',
                        severity='warning',
                        message=f'Moderate cyclomatic complexity ({complexity})',
                        file_path=str(py_file),
                        recommendation='Consider refactoring for better maintainability'
                    ))
                
                # Check comment ratio
                comment_ratio = metrics.get('comment_ratio', 0)
                if comment_ratio < 0.1:
                    self.validation_results.append(ValidationResult(
                        category='code_quality',
                        severity='info',
                        message=f'Low comment ratio ({comment_ratio:.2%})',
                        file_path=str(py_file),
                        recommendation='Add more documentation and comments'
                    ))
        
        avg_complexity = total_complexity / total_files if total_files > 0 else 0
        
        self.metrics.append(MetricResult(
            metric_name='average_cyclomatic_complexity',
            value=avg_complexity,
            unit='complexity_points',
            threshold=10.0,
            status='pass' if avg_complexity <= 10 else 'fail'
        ))

    def _analyze_security(self):
        """Analyze security vulnerabilities."""
        for py_file in self.project_path.rglob('*.py'):
            if py_file.name.startswith('__'):
                continue
                
            security_results = self.security_analyzer.analyze_file(str(py_file))
            self.validation_results.extend(security_results)

    def _analyze_performance(self):
        """Analyze performance issues."""
        for py_file in self.project_path.rglob('*.py'):
            if py_file.name.startswith('__'):
                continue
                
            performance_results = self.performance_analyzer.analyze_file(str(py_file))
            self.validation_results.extend(performance_results)

    def _analyze_testability(self):
        """Analyze testability and test coverage."""
        testability_info = self.testability_analyzer.analyze_project(str(self.project_path))
        
        self.metrics.append(MetricResult(
            metric_name='test_coverage_estimate',
            value=testability_info['test_coverage_estimate'],
            unit='ratio',
            threshold=0.8,
            status='pass' if testability_info['test_coverage_estimate'] >= 0.8 else 'warning'
        ))
        
        if testability_info['test_coverage_estimate'] < 0.5:
            self.validation_results.append(ValidationResult(
                category='testability',
                severity='warning',
                message=f'Low test coverage estimate ({testability_info["test_coverage_estimate"]:.2%})',
                recommendation='Increase test coverage by adding more unit tests'
            ))

    def _generate_recommendations(self) -> List[str]:
        """Generate comprehensive recommendations."""
        recommendations = []
        
        # Count issues by severity
        error_count = sum(1 for r in self.validation_results if r.severity == 'error')
        warning_count = sum(1 for r in self.validation_results if r.severity == 'warning')
        
        if error_count > 0:
            recommendations.append(f"🔴 Address {error_count} critical error(s) immediately")
        
        if warning_count > 0:
            recommendations.append(f"🟡 Review and fix {warning_count} warning(s)")
        
        # Architecture recommendations
        arch_analysis = None
        for metric in self.metrics:
            if metric.metric_name == 'average_cyclomatic_complexity' and metric.value > 15:
                recommendations.append("🏗️ Refactor complex modules to improve maintainability")
        
        # Security recommendations
        security_issues = [r for r in self.validation_results if r.category == 'security']
        if security_issues:
            recommendations.append("🛡️ Review and remediate security vulnerabilities")
        
        # Performance recommendations
        performance_issues = [r for r in self.validation_results if r.category == 'performance']
        if len(performance_issues) > 5:
            recommendations.append("⚡ Optimize performance bottlenecks for better efficiency")
        
        # Test coverage recommendations
        test_coverage = next((m.value for m in self.metrics if m.metric_name == 'test_coverage_estimate'), 0)
        if test_coverage < 0.7:
            recommendations.append("🧪 Increase test coverage to ensure reliability")
        
        if not recommendations:
            recommendations.append("✅ Code quality is excellent! Continue maintaining best practices")
        
        return recommendations

    def _calculate_overall_score(self) -> float:
        """Calculate overall code quality score."""
        # Base score
        score = 100.0
        
        # Deduct points for issues
        for result in self.validation_results:
            if result.severity == 'error':
                score -= 10.0
            elif result.severity == 'warning':
                score -= 5.0
            elif result.severity == 'info':
                score -= 1.0
        
        # Bonus points for good metrics
        for metric in self.metrics:
            if metric.status == 'pass':
                score += 2.0
        
        return max(0.0, min(100.0, score))

    def generate_report(self, validation_data: Dict[str, Any], output_path: str = None) -> str:
        """Generate a comprehensive validation report."""
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append("🚀 AUTONOMOUS SDLC VALIDATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Project: {validation_data['project_path']}")
        report_lines.append(f"Validation Time: {time.ctime(validation_data['timestamp'])}")
        report_lines.append(f"Execution Time: {validation_data['execution_time']:.2f}s")
        report_lines.append(f"Overall Score: {validation_data['overall_score']:.1f}/100")
        report_lines.append("")
        
        # Architecture Analysis
        if validation_data['architecture_analysis']:
            arch = validation_data['architecture_analysis']
            report_lines.append("📊 ARCHITECTURE ANALYSIS")
            report_lines.append("-" * 40)
            report_lines.append(f"Modules: {arch['total_modules']}")
            report_lines.append(f"Classes: {arch['total_classes']}")
            report_lines.append(f"Functions: {arch['total_functions']}")
            report_lines.append(f"Complexity Score: {arch['complexity_score']:.1f}")
            report_lines.append(f"Maintainability Index: {arch['maintainability_index']:.1f}")
            report_lines.append(f"Test Coverage Estimate: {arch['test_coverage_estimate']:.1%}")
            report_lines.append(f"Architecture Quality: {arch['architecture_quality']}")
            if arch['design_patterns']:
                report_lines.append(f"Design Patterns: {', '.join(arch['design_patterns'])}")
            report_lines.append("")
        
        # Metrics Summary
        if validation_data['metrics']:
            report_lines.append("📈 METRICS SUMMARY")
            report_lines.append("-" * 40)
            for metric in validation_data['metrics']:
                status_emoji = "✅" if metric['status'] == 'pass' else "⚠️" if metric['status'] == 'warning' else "❌"
                report_lines.append(f"{status_emoji} {metric['metric_name']}: {metric['value']:.2f} {metric['unit']}")
            report_lines.append("")
        
        # Issues Summary
        if validation_data['validation_results']:
            issues_by_category = defaultdict(list)
            for result in validation_data['validation_results']:
                issues_by_category[result['category']].append(result)
            
            report_lines.append("🔍 ISSUES SUMMARY")
            report_lines.append("-" * 40)
            
            for category, issues in issues_by_category.items():
                error_count = sum(1 for i in issues if i['severity'] == 'error')
                warning_count = sum(1 for i in issues if i['severity'] == 'warning')
                info_count = sum(1 for i in issues if i['severity'] == 'info')
                
                report_lines.append(f"{category.upper()}: {error_count} errors, {warning_count} warnings, {info_count} info")
            report_lines.append("")
        
        # Recommendations
        if validation_data['recommendations']:
            report_lines.append("💡 RECOMMENDATIONS")
            report_lines.append("-" * 40)
            for rec in validation_data['recommendations']:
                report_lines.append(f"• {rec}")
            report_lines.append("")
        
        # Detailed Issues
        if validation_data['validation_results']:
            report_lines.append("📋 DETAILED ISSUES")
            report_lines.append("-" * 40)
            
            for result in sorted(validation_data['validation_results'], 
                               key=lambda x: (x['severity'] == 'info', x['severity'] == 'warning', x['category'])):
                severity_emoji = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(result['severity'], "❓")
                
                location = ""
                if result['file_path']:
                    file_name = os.path.basename(result['file_path'])
                    location = f" ({file_name}"
                    if result['line_number']:
                        location += f":{result['line_number']}"
                    location += ")"
                
                report_lines.append(f"{severity_emoji} [{result['category'].upper()}] {result['message']}{location}")
                
                if result['recommendation']:
                    report_lines.append(f"   💡 {result['recommendation']}")
                
                report_lines.append("")
        
        report_lines.append("=" * 80)
        
        report_content = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"📄 Report saved to: {output_path}")
        
        return report_content


def main():
    """Main entry point for autonomous SDLC validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Autonomous SDLC Validation Framework")
    parser.add_argument("project_path", nargs="?", default=".", help="Path to project directory")
    parser.add_argument("--output", "-o", help="Output file for detailed report")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    validator = AutonomousSDLCValidator(args.project_path)
    validation_data = validator.run_comprehensive_validation()
    
    if args.json:
        print(json.dumps(validation_data, indent=2))
    elif not args.quiet:
        report = validator.generate_report(validation_data, args.output)
        if not args.output:
            print(report)
    
    # Exit with appropriate code
    error_count = sum(1 for r in validation_data.get('validation_results', []) 
                     if r.get('severity') == 'error')
    sys.exit(1 if error_count > 0 else 0)


if __name__ == "__main__":
    main()