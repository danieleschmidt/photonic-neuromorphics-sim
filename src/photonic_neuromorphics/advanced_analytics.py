"""
Advanced Analytics and Insights for Photonic Neuromorphics

Comprehensive analytics framework providing deep insights into photonic neural
network performance, optimization opportunities, and predictive intelligence.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
import logging
import time
import json
from collections import defaultdict, deque
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

from .core import PhotonicSNN, OpticalParameters
from .exceptions import ValidationError, PhotonicNeuromorphicsException
from .autonomous_learning import LearningMetrics, AutonomousLearningFramework
from .quantum_photonic_interface import HybridQuantumPhotonic
from .realtime_adaptive_optimization import PerformanceMetrics, RealTimeOptimizer
from .distributed_computing import NodeManager, ComputeTask


@dataclass
class AnalyticsMetric:
    """Individual analytics metric."""
    name: str
    value: float
    unit: str = ""
    category: str = "general"
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0  # 0-1 confidence in measurement
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InsightReport:
    """Generated insight report."""
    insight_type: str
    title: str
    description: str
    severity: str  # low, medium, high, critical
    confidence: float
    recommendations: List[str] = field(default_factory=list)
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class PerformanceAnalyzer:
    """Advanced performance analysis for photonic systems."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.insights_history = deque(maxlen=100)
        self.logger = logging.getLogger(__name__)
        
        # Analysis components
        self.scaler = StandardScaler()
        self.anomaly_detector = None
        self.trend_models = {}
        
    def record_metrics(self, metrics: Dict[str, AnalyticsMetric]) -> None:
        """Record new metrics for analysis."""
        timestamp = time.time()
        self.metrics_history.append({
            'timestamp': timestamp,
            'metrics': metrics
        })
        
        # Perform real-time analysis
        if len(self.metrics_history) >= 10:
            self._analyze_real_time()
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.metrics_history:
            return {"error": "No metrics data available"}
        
        report = {
            'summary': self._generate_summary_statistics(),
            'trends': self._analyze_trends(),
            'anomalies': self._detect_anomalies(),
            'correlations': self._analyze_correlations(),
            'insights': self._generate_insights(),
            'recommendations': self._generate_recommendations(),
            'timestamp': time.time()
        }
        
        return report
    
    def _generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics for all metrics."""
        if not self.metrics_history:
            return {}
        
        # Extract metric values by category
        metric_data = defaultdict(list)
        
        for record in self.metrics_history:
            for metric_name, metric in record['metrics'].items():
                metric_data[metric_name].append(metric.value)
        
        summary = {}
        for metric_name, values in metric_data.items():
            if values:
                summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75),
                    'count': len(values),
                    'latest': values[-1] if values else 0
                }
        
        return summary
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        if len(self.metrics_history) < 5:
            return {}
        
        trends = {}
        
        # Extract time series data
        timestamps = [record['timestamp'] for record in self.metrics_history]
        timestamps = np.array(timestamps) - timestamps[0]  # Normalize to start at 0
        
        for metric_name in self._get_all_metric_names():
            values = []
            for record in self.metrics_history:
                if metric_name in record['metrics']:
                    values.append(record['metrics'][metric_name].value)
                else:
                    values.append(np.nan)
            
            if len(values) >= 5 and not all(np.isnan(values)):
                # Remove NaN values
                valid_indices = ~np.isnan(values)
                valid_times = timestamps[valid_indices]
                valid_values = np.array(values)[valid_indices]
                
                if len(valid_values) >= 3:
                    # Linear trend analysis
                    slope, intercept, r_value, p_value, std_err = stats.linregress(valid_times, valid_values)
                    
                    trends[metric_name] = {
                        'slope': slope,
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
                        'trend_strength': abs(r_value),
                        'is_significant': p_value < 0.05
                    }
        
        return trends
    
    def _detect_anomalies(self) -> Dict[str, Any]:
        """Detect anomalies in performance metrics."""
        anomalies = {}
        
        for metric_name in self._get_all_metric_names():
            values = []
            timestamps = []
            
            for record in self.metrics_history:
                if metric_name in record['metrics']:
                    values.append(record['metrics'][metric_name].value)
                    timestamps.append(record['timestamp'])
            
            if len(values) >= 10:
                values = np.array(values)
                
                # Z-score based anomaly detection
                z_scores = np.abs(stats.zscore(values))
                anomaly_threshold = 3.0
                
                anomalous_indices = np.where(z_scores > anomaly_threshold)[0]
                
                if len(anomalous_indices) > 0:
                    anomalies[metric_name] = {
                        'count': len(anomalous_indices),
                        'percentage': len(anomalous_indices) / len(values) * 100,
                        'most_recent': bool(len(anomalous_indices) > 0 and anomalous_indices[-1] >= len(values) - 5),
                        'severity': 'high' if len(anomalous_indices) / len(values) > 0.1 else 'medium'
                    }
        
        return anomalies
    
    def _analyze_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between different metrics."""
        correlations = {}
        
        # Build correlation matrix
        metric_names = self._get_all_metric_names()
        if len(metric_names) < 2:
            return correlations
        
        # Create data matrix
        data_matrix = []
        for metric_name in metric_names:
            values = []
            for record in self.metrics_history:
                if metric_name in record['metrics']:
                    values.append(record['metrics'][metric_name].value)
                else:
                    values.append(np.nan)
            data_matrix.append(values)
        
        data_matrix = np.array(data_matrix).T
        
        # Calculate correlations
        try:
            df = pd.DataFrame(data_matrix, columns=metric_names)
            corr_matrix = df.corr()
            
            # Find strong correlations
            strong_correlations = []
            for i, metric1 in enumerate(metric_names):
                for j, metric2 in enumerate(metric_names):
                    if i < j:  # Avoid duplicates
                        corr_value = corr_matrix.iloc[i, j]
                        if abs(corr_value) > 0.7 and not np.isnan(corr_value):
                            strong_correlations.append({
                                'metric1': metric1,
                                'metric2': metric2,
                                'correlation': corr_value,
                                'strength': 'strong' if abs(corr_value) > 0.8 else 'moderate'
                            })
            
            correlations = {
                'strong_correlations': strong_correlations,
                'correlation_matrix': corr_matrix.to_dict()
            }
            
        except Exception as e:
            self.logger.error(f"Correlation analysis failed: {e}")
        
        return correlations
    
    def _generate_insights(self) -> List[InsightReport]:
        """Generate actionable insights from analysis."""
        insights = []
        
        # Trend-based insights
        trends = self._analyze_trends()
        for metric_name, trend_data in trends.items():
            if trend_data['is_significant']:
                if trend_data['trend_direction'] == 'decreasing' and 'accuracy' in metric_name.lower():
                    insights.append(InsightReport(
                        insight_type="performance_degradation",
                        title=f"Declining {metric_name}",
                        description=f"{metric_name} is showing a significant downward trend (R² = {trend_data['r_squared']:.3f})",
                        severity="high",
                        confidence=min(0.9, trend_data['trend_strength']),
                        recommendations=[
                            "Review recent changes to network configuration",
                            "Consider retraining or parameter optimization",
                            "Check for data quality issues"
                        ]
                    ))
                
                elif trend_data['trend_direction'] == 'increasing' and 'latency' in metric_name.lower():
                    insights.append(InsightReport(
                        insight_type="performance_degradation",
                        title=f"Increasing {metric_name}",
                        description=f"{metric_name} is showing a significant upward trend (R² = {trend_data['r_squared']:.3f})",
                        severity="medium",
                        confidence=min(0.9, trend_data['trend_strength']),
                        recommendations=[
                            "Optimize network architecture for speed",
                            "Consider distributed processing",
                            "Review system resource utilization"
                        ]
                    ))
        
        # Anomaly-based insights
        anomalies = self._detect_anomalies()
        for metric_name, anomaly_data in anomalies.items():
            if anomaly_data['most_recent']:
                insights.append(InsightReport(
                    insight_type="anomaly_detection",
                    title=f"Recent Anomaly in {metric_name}",
                    description=f"Detected {anomaly_data['count']} anomalous values in {metric_name} ({anomaly_data['percentage']:.1f}% of data)",
                    severity=anomaly_data['severity'],
                    confidence=0.8,
                    recommendations=[
                        "Investigate recent system changes",
                        "Check for external factors affecting performance",
                        "Consider adjusting anomaly detection thresholds"
                    ]
                ))
        
        # Performance efficiency insights
        summary = self._generate_summary_statistics()
        if 'accuracy' in summary and 'energy_efficiency' in summary:
            acc_mean = summary['accuracy']['mean']
            energy_mean = summary['energy_efficiency']['mean']
            
            if acc_mean > 0.9 and energy_mean < 50:
                insights.append(InsightReport(
                    insight_type="optimization_opportunity",
                    title="Energy Efficiency Opportunity",
                    description=f"High accuracy ({acc_mean:.3f}) but low energy efficiency ({energy_mean:.1f})",
                    severity="medium",
                    confidence=0.7,
                    recommendations=[
                        "Optimize optical parameters for better energy efficiency",
                        "Consider network pruning techniques",
                        "Implement dynamic voltage scaling"
                    ]
                ))
        
        return insights
    
    def _generate_recommendations(self) -> List[str]:
        """Generate general recommendations based on overall analysis."""
        recommendations = []
        
        # Analyze overall system health
        summary = self._generate_summary_statistics()
        anomalies = self._detect_anomalies()
        
        # General performance recommendations
        if len(anomalies) > len(summary) * 0.3:
            recommendations.append("High anomaly rate detected - consider reviewing system stability")
        
        if 'accuracy' in summary:
            acc_std = summary['accuracy']['std']
            if acc_std > 0.1:
                recommendations.append("High accuracy variance - consider more stable training procedures")
        
        if 'latency' in summary:
            latency_mean = summary['latency']['mean']
            if latency_mean > 0.1:  # 100ms
                recommendations.append("High latency detected - consider optimization or distributed processing")
        
        # Add general best practices
        recommendations.extend([
            "Regular model retraining to maintain performance",
            "Continuous monitoring of optical parameter drift",
            "Periodic benchmarking against baseline performance"
        ])
        
        return recommendations
    
    def _analyze_real_time(self) -> None:
        """Perform real-time analysis on recent data."""
        if len(self.metrics_history) < 10:
            return
        
        # Check for recent performance drops
        recent_records = list(self.metrics_history)[-5:]
        older_records = list(self.metrics_history)[-10:-5]
        
        for metric_name in self._get_all_metric_names():
            recent_values = []
            older_values = []
            
            for record in recent_records:
                if metric_name in record['metrics']:
                    recent_values.append(record['metrics'][metric_name].value)
            
            for record in older_records:
                if metric_name in record['metrics']:
                    older_values.append(record['metrics'][metric_name].value)
            
            if len(recent_values) >= 3 and len(older_values) >= 3:
                recent_mean = np.mean(recent_values)
                older_mean = np.mean(older_values)
                
                # Check for significant drops in accuracy or efficiency
                if 'accuracy' in metric_name.lower() or 'efficiency' in metric_name.lower():
                    if recent_mean < older_mean * 0.9:  # 10% drop
                        self.logger.warning(f"Performance drop detected in {metric_name}: "
                                          f"{older_mean:.3f} → {recent_mean:.3f}")
    
    def _get_all_metric_names(self) -> List[str]:
        """Get all unique metric names from history."""
        metric_names = set()
        for record in self.metrics_history:
            metric_names.update(record['metrics'].keys())
        return list(metric_names)


class OptimizationAnalyzer:
    """Analyzer for optimization effectiveness and opportunities."""
    
    def __init__(self):
        self.optimization_history = []
        self.logger = logging.getLogger(__name__)
    
    def record_optimization_attempt(self,
                                  optimization_type: str,
                                  parameters_before: Dict[str, Any],
                                  parameters_after: Dict[str, Any],
                                  performance_before: PerformanceMetrics,
                                  performance_after: PerformanceMetrics) -> None:
        """Record an optimization attempt for analysis."""
        
        improvement = performance_after.composite_score() - performance_before.composite_score()
        
        record = {
            'timestamp': time.time(),
            'optimization_type': optimization_type,
            'parameters_before': parameters_before,
            'parameters_after': parameters_after,
            'performance_before': performance_before,
            'performance_after': performance_after,
            'improvement': improvement,
            'success': improvement > 0.01  # Consider successful if > 1% improvement
        }
        
        self.optimization_history.append(record)
    
    def analyze_optimization_effectiveness(self) -> Dict[str, Any]:
        """Analyze the effectiveness of different optimization strategies."""
        if not self.optimization_history:
            return {}
        
        analysis = {}
        
        # Group by optimization type
        by_type = defaultdict(list)
        for record in self.optimization_history:
            by_type[record['optimization_type']].append(record)
        
        for opt_type, records in by_type.items():
            improvements = [r['improvement'] for r in records]
            successes = sum(1 for r in records if r['success'])
            
            analysis[opt_type] = {
                'total_attempts': len(records),
                'success_rate': successes / len(records) if records else 0,
                'average_improvement': np.mean(improvements),
                'best_improvement': max(improvements) if improvements else 0,
                'worst_outcome': min(improvements) if improvements else 0,
                'consistency': 1.0 - np.std(improvements) if len(improvements) > 1 else 1.0
            }
        
        # Overall statistics
        all_improvements = [r['improvement'] for r in self.optimization_history]
        analysis['overall'] = {
            'total_optimizations': len(self.optimization_history),
            'overall_success_rate': sum(1 for r in self.optimization_history if r['success']) / len(self.optimization_history),
            'average_improvement': np.mean(all_improvements),
            'improvement_trend': self._calculate_improvement_trend()
        }
        
        return analysis
    
    def _calculate_improvement_trend(self) -> str:
        """Calculate if optimization effectiveness is improving over time."""
        if len(self.optimization_history) < 10:
            return "insufficient_data"
        
        # Split into earlier and later halves
        mid_point = len(self.optimization_history) // 2
        earlier_half = self.optimization_history[:mid_point]
        later_half = self.optimization_history[mid_point:]
        
        earlier_avg = np.mean([r['improvement'] for r in earlier_half])
        later_avg = np.mean([r['improvement'] for r in later_half])
        
        if later_avg > earlier_avg * 1.1:
            return "improving"
        elif later_avg < earlier_avg * 0.9:
            return "declining"
        else:
            return "stable"


class SystemHealthAnalyzer:
    """Comprehensive system health analysis."""
    
    def __init__(self):
        self.health_history = deque(maxlen=1000)
        self.alert_thresholds = {
            'accuracy': {'critical': 0.5, 'warning': 0.8},
            'latency': {'critical': 1.0, 'warning': 0.5},
            'memory_usage': {'critical': 0.95, 'warning': 0.85},
            'error_rate': {'critical': 0.1, 'warning': 0.05}
        }
        self.logger = logging.getLogger(__name__)
    
    def assess_system_health(self, 
                           performance_metrics: PerformanceMetrics,
                           node_manager: Optional[NodeManager] = None) -> Dict[str, Any]:
        """Assess overall system health."""
        
        health_score = 0.0
        health_components = {}
        alerts = []
        
        # Performance health assessment
        perf_health = self._assess_performance_health(performance_metrics)
        health_components['performance'] = perf_health
        health_score += perf_health['score'] * 0.4
        
        # Resource health assessment
        resource_health = self._assess_resource_health(performance_metrics)
        health_components['resources'] = resource_health
        health_score += resource_health['score'] * 0.3
        
        # Distributed system health (if available)
        if node_manager:
            cluster_health = self._assess_cluster_health(node_manager)
            health_components['cluster'] = cluster_health
            health_score += cluster_health['score'] * 0.3
        else:
            health_score = health_score / 0.7  # Normalize without cluster component
        
        # Generate alerts
        alerts.extend(self._generate_health_alerts(health_components))
        
        # Record health assessment
        health_record = {
            'timestamp': time.time(),
            'health_score': health_score,
            'components': health_components,
            'alerts': alerts
        }
        self.health_history.append(health_record)
        
        return {
            'overall_health_score': health_score,
            'health_grade': self._get_health_grade(health_score),
            'components': health_components,
            'alerts': alerts,
            'recommendations': self._generate_health_recommendations(health_components)
        }
    
    def _assess_performance_health(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Assess performance-related health."""
        score = 100.0
        issues = []
        
        # Check accuracy
        if metrics.accuracy < self.alert_thresholds['accuracy']['critical']:
            score -= 40
            issues.append("Critical accuracy degradation")
        elif metrics.accuracy < self.alert_thresholds['accuracy']['warning']:
            score -= 20
            issues.append("Low accuracy warning")
        
        # Check latency
        if metrics.latency > self.alert_thresholds['latency']['critical']:
            score -= 30
            issues.append("Critical latency issues")
        elif metrics.latency > self.alert_thresholds['latency']['warning']:
            score -= 15
            issues.append("High latency warning")
        
        # Check error rate
        if metrics.error_rate > self.alert_thresholds['error_rate']['critical']:
            score -= 25
            issues.append("Critical error rate")
        elif metrics.error_rate > self.alert_thresholds['error_rate']['warning']:
            score -= 10
            issues.append("Elevated error rate")
        
        return {
            'score': max(0, score) / 100.0,
            'status': 'healthy' if score > 80 else 'warning' if score > 50 else 'critical',
            'issues': issues
        }
    
    def _assess_resource_health(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Assess resource utilization health."""
        score = 100.0
        issues = []
        
        # Check memory usage
        if metrics.memory_usage > self.alert_thresholds['memory_usage']['critical']:
            score -= 40
            issues.append("Critical memory usage")
        elif metrics.memory_usage > self.alert_thresholds['memory_usage']['warning']:
            score -= 20
            issues.append("High memory usage")
        
        # Check GPU utilization (simplified)
        if metrics.gpu_utilization > 0.95:
            score -= 20
            issues.append("GPU overutilization")
        elif metrics.gpu_utilization < 0.3:
            score -= 10
            issues.append("Low GPU utilization")
        
        # Check power consumption vs performance
        if metrics.power_consumption > 100 and metrics.throughput < 500:  # Poor efficiency
            score -= 15
            issues.append("Poor power efficiency")
        
        return {
            'score': max(0, score) / 100.0,
            'status': 'healthy' if score > 80 else 'warning' if score > 50 else 'critical',
            'issues': issues
        }
    
    def _assess_cluster_health(self, node_manager: NodeManager) -> Dict[str, Any]:
        """Assess distributed cluster health."""
        cluster_status = node_manager.get_cluster_status()
        
        score = 100.0
        issues = []
        
        # Check node availability
        if cluster_status['online_nodes'] == 0:
            score = 0
            issues.append("No nodes available")
        elif cluster_status['online_nodes'] < cluster_status['total_nodes'] * 0.5:
            score -= 40
            issues.append("More than half of nodes offline")
        elif cluster_status['online_nodes'] < cluster_status['total_nodes']:
            score -= 20
            issues.append("Some nodes offline")
        
        # Check task queue health
        if cluster_status['pending_tasks'] > 100:
            score -= 30
            issues.append("Large task backlog")
        elif cluster_status['pending_tasks'] > 50:
            score -= 15
            issues.append("Growing task queue")
        
        # Check average load
        if cluster_status['average_load'] > 0.9:
            score -= 25
            issues.append("Cluster overloaded")
        elif cluster_status['average_load'] > 0.7:
            score -= 10
            issues.append("High cluster load")
        
        return {
            'score': max(0, score) / 100.0,
            'status': 'healthy' if score > 80 else 'warning' if score > 50 else 'critical',
            'issues': issues,
            'cluster_metrics': cluster_status
        }
    
    def _generate_health_alerts(self, health_components: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate health alerts based on component status."""
        alerts = []
        
        for component_name, component_data in health_components.items():
            if component_data['status'] == 'critical':
                alerts.append({
                    'type': 'critical',
                    'component': component_name,
                    'message': f"Critical issues detected in {component_name}",
                    'issues': component_data.get('issues', [])
                })
            elif component_data['status'] == 'warning':
                alerts.append({
                    'type': 'warning',
                    'component': component_name,
                    'message': f"Warning conditions in {component_name}",
                    'issues': component_data.get('issues', [])
                })
        
        return alerts
    
    def _generate_health_recommendations(self, health_components: Dict[str, Any]) -> List[str]:
        """Generate health improvement recommendations."""
        recommendations = []
        
        for component_name, component_data in health_components.items():
            if component_data['status'] in ['warning', 'critical']:
                if component_name == 'performance':
                    recommendations.extend([
                        "Review and optimize network architecture",
                        "Consider retraining with updated data",
                        "Analyze and tune hyperparameters"
                    ])
                elif component_name == 'resources':
                    recommendations.extend([
                        "Monitor and optimize resource utilization",
                        "Consider scaling up hardware resources",
                        "Implement more efficient algorithms"
                    ])
                elif component_name == 'cluster':
                    recommendations.extend([
                        "Check network connectivity to offline nodes",
                        "Balance load distribution across nodes",
                        "Consider adding more compute capacity"
                    ])
        
        return list(set(recommendations))  # Remove duplicates
    
    def _get_health_grade(self, health_score: float) -> str:
        """Convert health score to letter grade."""
        if health_score >= 0.9:
            return "A"
        elif health_score >= 0.8:
            return "B"
        elif health_score >= 0.7:
            return "C"
        elif health_score >= 0.6:
            return "D"
        else:
            return "F"


class AdvancedAnalyticsFramework:
    """Comprehensive analytics framework."""
    
    def __init__(self):
        self.performance_analyzer = PerformanceAnalyzer()
        self.optimization_analyzer = OptimizationAnalyzer()
        self.health_analyzer = SystemHealthAnalyzer()
        
        self.analytics_history = []
        self.logger = logging.getLogger(__name__)
    
    def analyze_system(self,
                      performance_metrics: PerformanceMetrics,
                      node_manager: Optional[NodeManager] = None) -> Dict[str, Any]:
        """Perform comprehensive system analysis."""
        
        analysis_start = time.time()
        
        # Convert performance metrics to analytics metrics
        analytics_metrics = self._convert_to_analytics_metrics(performance_metrics)
        self.performance_analyzer.record_metrics(analytics_metrics)
        
        # Generate performance report
        performance_report = self.performance_analyzer.generate_performance_report()
        
        # Assess system health
        health_assessment = self.health_analyzer.assess_system_health(
            performance_metrics, node_manager
        )
        
        # Analyze optimization effectiveness
        optimization_analysis = self.optimization_analyzer.analyze_optimization_effectiveness()
        
        # Compile comprehensive report
        comprehensive_report = {
            'analysis_timestamp': time.time(),
            'analysis_duration': time.time() - analysis_start,
            'performance_analysis': performance_report,
            'health_assessment': health_assessment,
            'optimization_analysis': optimization_analysis,
            'executive_summary': self._generate_executive_summary(
                performance_report, health_assessment, optimization_analysis
            )
        }
        
        # Record analysis
        self.analytics_history.append(comprehensive_report)
        
        return comprehensive_report
    
    def _convert_to_analytics_metrics(self, perf_metrics: PerformanceMetrics) -> Dict[str, AnalyticsMetric]:
        """Convert performance metrics to analytics metrics."""
        return {
            'accuracy': AnalyticsMetric(
                name='accuracy',
                value=perf_metrics.accuracy,
                category='performance'
            ),
            'throughput': AnalyticsMetric(
                name='throughput',
                value=perf_metrics.throughput,
                unit='samples/sec',
                category='performance'
            ),
            'latency': AnalyticsMetric(
                name='latency',
                value=perf_metrics.latency,
                unit='seconds',
                category='performance'
            ),
            'energy_efficiency': AnalyticsMetric(
                name='energy_efficiency',
                value=perf_metrics.energy_efficiency,
                unit='TOPS/W',
                category='efficiency'
            ),
            'memory_usage': AnalyticsMetric(
                name='memory_usage',
                value=perf_metrics.memory_usage,
                unit='MB',
                category='resources'
            ),
            'error_rate': AnalyticsMetric(
                name='error_rate',
                value=perf_metrics.error_rate,
                category='reliability'
            )
        }
    
    def _generate_executive_summary(self,
                                  performance_report: Dict[str, Any],
                                  health_assessment: Dict[str, Any],
                                  optimization_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of analysis."""
        
        # Key findings
        key_findings = []
        
        # Health status
        health_grade = health_assessment.get('health_grade', 'Unknown')
        key_findings.append(f"System Health Grade: {health_grade}")
        
        # Performance trends
        trends = performance_report.get('trends', {})
        accuracy_trend = trends.get('accuracy', {})
        if accuracy_trend.get('is_significant'):
            direction = accuracy_trend.get('trend_direction', 'unknown')
            key_findings.append(f"Accuracy trend: {direction}")
        
        # Optimization effectiveness
        opt_overall = optimization_analysis.get('overall', {})
        success_rate = opt_overall.get('overall_success_rate', 0)
        if success_rate > 0:
            key_findings.append(f"Optimization success rate: {success_rate:.1%}")
        
        # Action items
        action_items = []
        
        # From health alerts
        alerts = health_assessment.get('alerts', [])
        critical_alerts = [a for a in alerts if a['type'] == 'critical']
        if critical_alerts:
            action_items.append("Address critical system health issues immediately")
        
        # From insights
        insights = performance_report.get('insights', [])
        high_severity_insights = [i for i in insights if i.severity == 'high']
        if high_severity_insights:
            action_items.append("Investigate high-severity performance issues")
        
        # Recommendations
        recommendations = health_assessment.get('recommendations', [])
        top_recommendations = recommendations[:3]  # Top 3 recommendations
        
        return {
            'key_findings': key_findings,
            'action_items': action_items,
            'top_recommendations': top_recommendations,
            'overall_status': health_assessment.get('health_grade', 'Unknown'),
            'requires_immediate_attention': len(critical_alerts) > 0 or len(high_severity_insights) > 0
        }


def create_advanced_analytics_demo() -> AdvancedAnalyticsFramework:
    """Create demonstration advanced analytics framework."""
    return AdvancedAnalyticsFramework()


def run_advanced_analytics_demo():
    """Run advanced analytics demonstration."""
    print("📊 Advanced Analytics & Insights Demo")
    print("=" * 40)
    
    # Create analytics framework
    analytics = create_advanced_analytics_demo()
    
    # Simulate performance data over time
    print("Generating simulated performance data...")
    
    for i in range(50):
        # Simulate varying performance metrics
        base_accuracy = 0.85 + 0.1 * np.sin(i * 0.2) + np.random.normal(0, 0.02)
        base_latency = 0.01 + 0.005 * i * 0.1 + np.random.normal(0, 0.001)
        
        metrics = PerformanceMetrics(
            accuracy=max(0, min(1, base_accuracy)),
            throughput=800 + np.random.normal(0, 50),
            latency=max(0.001, base_latency),
            energy_efficiency=100 + np.random.normal(0, 20),
            memory_usage=0.3 + np.random.uniform(0, 0.4),
            error_rate=max(0, np.random.exponential(0.01))
        )
        
        # Record metrics
        analytics_metrics = analytics._convert_to_analytics_metrics(metrics)
        analytics.performance_analyzer.record_metrics(analytics_metrics)
        
        time.sleep(0.01)  # Small delay to simulate real-time
    
    print(f"Recorded {len(analytics.performance_analyzer.metrics_history)} performance samples")
    
    # Generate comprehensive analysis
    print("\n🔍 Performing Comprehensive Analysis...")
    
    # Create mock performance metrics for current analysis
    current_metrics = PerformanceMetrics(
        accuracy=0.82,
        throughput=750,
        latency=0.015,
        energy_efficiency=85,
        memory_usage=0.65,
        error_rate=0.03
    )
    
    # Perform analysis
    analysis_report = analytics.analyze_system(current_metrics)
    
    # Display results
    print("\n📈 Analysis Results:")
    
    # Executive summary
    exec_summary = analysis_report['executive_summary']
    print(f"\nExecutive Summary:")
    print(f"  Overall Status: {exec_summary['overall_status']}")
    print(f"  Requires Immediate Attention: {exec_summary['requires_immediate_attention']}")
    
    print(f"\nKey Findings:")
    for finding in exec_summary['key_findings']:
        print(f"  • {finding}")
    
    # Health assessment
    health = analysis_report['health_assessment']
    print(f"\nSystem Health Assessment:")
    print(f"  Health Grade: {health['health_grade']}")
    print(f"  Health Score: {health['overall_health_score']:.3f}")
    
    if health['alerts']:
        print(f"\nActive Alerts:")
        for alert in health['alerts']:
            print(f"  {alert['type'].upper()}: {alert['message']}")
    
    # Performance insights
    performance = analysis_report['performance_analysis']
    insights = performance.get('insights', [])
    if insights:
        print(f"\nPerformance Insights:")
        for insight in insights[:3]:  # Show top 3
            print(f"  • {insight.title} (Confidence: {insight.confidence:.2f})")
            print(f"    {insight.description}")
    
    # Recommendations
    if exec_summary['top_recommendations']:
        print(f"\nTop Recommendations:")
        for i, rec in enumerate(exec_summary['top_recommendations'], 1):
            print(f"  {i}. {rec}")
    
    # Trends analysis
    trends = performance.get('trends', {})
    if trends:
        print(f"\nTrend Analysis:")
        for metric, trend_data in list(trends.items())[:3]:
            if trend_data.get('is_significant'):
                direction = trend_data['trend_direction']
                strength = trend_data['trend_strength']
                print(f"  • {metric}: {direction} trend (strength: {strength:.3f})")
    
    print(f"\n📊 Analysis completed in {analysis_report['analysis_duration']:.3f} seconds")
    
    return analytics


if __name__ == "__main__":
    run_advanced_analytics_demo()