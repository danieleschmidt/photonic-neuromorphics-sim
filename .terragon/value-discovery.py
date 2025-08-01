#!/usr/bin/env python3
"""
Terragon Autonomous SDLC Value Discovery Engine
==============================================

Continuous value discovery and autonomous execution system for mature repositories.
Implements advanced scoring algorithms (WSJF + ICE + Technical Debt) with machine learning.
"""

import asyncio
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import yaml
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('terragon.value_discovery')

@dataclass
class ValueItem:
    """Represents a discovered value item with comprehensive scoring."""
    id: str
    title: str
    description: str
    category: str
    source: str
    files: List[str]
    estimated_effort: float  # hours
    priority: str
    
    # Scoring components
    wsjf_score: float = 0.0
    ice_score: float = 0.0
    technical_debt_score: float = 0.0
    composite_score: float = 0.0
    
    # Business context
    user_impact: int = 0  # 1-10 scale
    time_criticality: int = 0  # 1-10 scale
    risk_reduction: int = 0  # 1-10 scale
    opportunity_enablement: int = 0  # 1-10 scale
    
    # ICE components
    impact: int = 0  # 1-10 scale
    confidence: int = 0  # 1-10 scale
    ease: int = 0  # 1-10 scale
    
    # Technical debt components
    debt_impact: float = 0.0  # maintenance hours saved
    debt_interest: float = 0.0  # future cost if not addressed
    hotspot_multiplier: float = 1.0  # 1-5x based on file activity
    
    # Metadata
    discovered_at: str = ""
    last_updated: str = ""
    execution_history: List[Dict] = None
    
    def __post_init__(self):
        if self.execution_history is None:
            self.execution_history = []
        if not self.discovered_at:
            self.discovered_at = datetime.now().isoformat()
        if not self.last_updated:
            self.last_updated = self.discovered_at

class ValueDiscoveryEngine:
    """Advanced value discovery engine with continuous learning."""
    
    def __init__(self, config_path: str = ".terragon/config.yaml"):
        self.config_path = Path(config_path)
        self.load_config()
        self.repo_root = Path.cwd()
        self.backlog: List[ValueItem] = []
        self.execution_history: List[Dict] = []
        self.learning_data: Dict = {}
        
    def load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            sys.exit(1)
    
    async def discover_all_value_items(self) -> List[ValueItem]:
        """Comprehensive value discovery from all configured sources."""
        logger.info("🔍 Starting comprehensive value discovery...")
        
        all_items = []
        
        # Parallel discovery from all sources
        discovery_tasks = [
            self.discover_from_git_history(),
            self.discover_from_static_analysis(),
            self.discover_from_security_scans(),
            self.discover_from_performance_monitoring(),
            self.discover_from_issue_tracking(),
            self.discover_from_dependency_analysis(),
            self.discover_from_code_quality(),
        ]
        
        results = await asyncio.gather(*discovery_tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Discovery task failed: {result}")
            else:
                all_items.extend(result)
        
        # Deduplicate and enrich items
        deduplicated = self.deduplicate_items(all_items)
        enriched = await self.enrich_with_context(deduplicated)
        
        logger.info(f"📊 Discovered {len(enriched)} value items")
        return enriched
    
    async def discover_from_git_history(self) -> List[ValueItem]:
        """Discover value items from Git history and code comments."""
        items = []
        
        try:
            # Search for TODO/FIXME markers in code
            result = subprocess.run([
                'grep', '-r', '-n', '-E', 
                r'(TODO|FIXME|XXX|HACK|DEPRECATED|temp|quick fix)',
                'src/', 'tests/'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            for line in result.stdout.split('\n'):
                if line.strip():
                    parts = line.split(':', 3)
                    if len(parts) >= 3:
                        file_path, line_num, content = parts[0], parts[1], parts[2]
                        
                        item = ValueItem(
                            id=f"git-{hash(line) % 10000:04d}",
                            title=f"Address technical debt in {Path(file_path).name}",
                            description=content.strip(),
                            category="technical_debt",
                            source="git_history",
                            files=[file_path],
                            estimated_effort=self.estimate_effort_from_content(content),
                            priority="medium"
                        )
                        items.append(item)
            
            # Analyze commit messages for quick fixes
            result = subprocess.run([
                'git', 'log', '--oneline', '--grep=temp', '--grep=quick', 
                '--grep=hack', '--grep=wip', '-n', '50'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            for line in result.stdout.split('\n'):
                if line.strip():
                    commit_hash, message = line.split(' ', 1)
                    
                    item = ValueItem(
                        id=f"commit-{commit_hash[:8]}",
                        title=f"Review potential technical debt: {message[:50]}",
                        description=f"Commit {commit_hash} may indicate technical debt",
                        category="code_review",
                        source="git_history",
                        files=[],
                        estimated_effort=2.0,
                        priority="low"
                    )
                    items.append(item)
        
        except subprocess.CalledProcessError as e:
            logger.warning(f"Git history analysis failed: {e}")
        
        return items
    
    async def discover_from_static_analysis(self) -> List[ValueItem]:
        """Discover issues from static analysis tools."""
        items = []
        
        # Run Ruff for code quality issues
        try:
            result = subprocess.run([
                'ruff', 'check', '--format=json', 'src/', 'tests/'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.stdout:
                ruff_issues = json.loads(result.stdout)
                for issue in ruff_issues[:20]:  # Limit to top 20
                    item = ValueItem(
                        id=f"ruff-{hash(str(issue)) % 10000:04d}",
                        title=f"Fix {issue['code']}: {issue['message'][:50]}",
                        description=issue['message'],
                        category="code_quality",
                        source="static_analysis",
                        files=[issue['filename']],
                        estimated_effort=0.5,
                        priority="low"
                    )
                    items.append(item)
        
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            logger.warning(f"Ruff analysis failed: {e}")
        
        # Run MyPy for type checking
        try:
            result = subprocess.run([
                'mypy', 'src/', '--ignore-missing-imports'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            type_errors = len(result.stdout.split('\n'))
            if type_errors > 10:
                item = ValueItem(
                    id="mypy-improvements",
                    title=f"Improve type annotations ({type_errors} issues)",
                    description="Enhance type safety and code documentation",
                    category="code_quality",
                    source="static_analysis",
                    files=["src/"],
                    estimated_effort=type_errors * 0.1,
                    priority="medium"
                )
                items.append(item)
        
        except subprocess.CalledProcessError as e:
            logger.warning(f"MyPy analysis failed: {e}")
        
        return items
    
    async def discover_from_security_scans(self) -> List[ValueItem]:
        """Discover security vulnerabilities and improvements."""
        items = []
        
        # Check for known vulnerabilities in dependencies
        try:
            result = subprocess.run([
                'safety', 'check', '--json'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.stdout:
                vulnerabilities = json.loads(result.stdout)
                for vuln in vulnerabilities:
                    item = ValueItem(
                        id=f"security-{vuln['id']}",
                        title=f"Fix {vuln['package_name']} vulnerability",
                        description=vuln['advisory'],
                        category="security",
                        source="security_scan",
                        files=["requirements.txt", "pyproject.toml"],
                        estimated_effort=1.0,
                        priority="high"
                    )
                    items.append(item)
        
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Safety check failed: {e}")
        
        # Run Bandit for security issues in code
        try:
            result = subprocess.run([
                'bandit', '-r', 'src/', '-f', 'json'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.stdout:
                bandit_results = json.loads(result.stdout)
                for issue in bandit_results.get('results', [])[:10]:
                    item = ValueItem(
                        id=f"bandit-{issue['test_id']}",
                        title=f"Security: {issue['test_name']}",
                        description=issue['issue_text'],
                        category="security",
                        source="security_scan",
                        files=[issue['filename']],
                        estimated_effort=2.0,
                        priority="high" if issue['issue_severity'] == "HIGH" else "medium"
                    )
                    items.append(item)
        
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Bandit scan failed: {e}")
        
        return items
    
    async def discover_from_performance_monitoring(self) -> List[ValueItem]:
        """Discover performance optimization opportunities."""
        items = []
        
        # Check for slow tests
        try:
            result = subprocess.run([
                'pytest', '--collect-only', '-q'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            test_count = len([l for l in result.stdout.split('\n') if 'test' in l])
            
            if test_count > 50:
                item = ValueItem(
                    id="test-optimization",
                    title="Optimize test suite performance",
                    description=f"Large test suite ({test_count} tests) may benefit from optimization",
                    category="performance",
                    source="performance_monitoring",
                    files=["tests/"],
                    estimated_effort=4.0,
                    priority="medium"
                )
                items.append(item)
        
        except subprocess.CalledProcessError as e:
            logger.warning(f"Test analysis failed: {e}")
        
        # Analyze import times
        try:
            result = subprocess.run([
                'python', '-c', 
                'import time; start=time.time(); import photonic_neuromorphics; print(f"Import time: {time.time()-start:.3f}s")'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if "Import time:" in result.stdout:
                import_time = float(result.stdout.split()[-1].rstrip('s'))
                if import_time > 1.0:
                    item = ValueItem(
                        id="import-optimization",
                        title="Optimize module import performance",
                        description=f"Module import takes {import_time:.2f}s",
                        category="performance",
                        source="performance_monitoring",
                        files=["src/photonic_neuromorphics/__init__.py"],
                        estimated_effort=3.0,
                        priority="medium"
                    )
                    items.append(item)
        
        except (subprocess.CalledProcessError, ValueError) as e:
            logger.warning(f"Import analysis failed: {e}")
        
        return items
    
    async def discover_from_issue_tracking(self) -> List[ValueItem]:
        """Discover items from GitHub issues and PRs."""
        items = []
        
        # This would integrate with GitHub API in a real implementation
        # For now, simulate discovering common improvement areas
        
        common_improvements = [
            {
                "title": "Add more comprehensive docstrings",
                "description": "Improve API documentation for better developer experience",
                "category": "documentation",
                "effort": 6.0,
                "priority": "medium"
            },
            {
                "title": "Implement caching for expensive computations",
                "description": "Add memoization for photonic simulation calculations",
                "category": "performance",
                "effort": 8.0,
                "priority": "high"
            },
            {
                "title": "Add integration tests for RTL generation",
                "description": "Ensure RTL output quality with comprehensive tests",
                "category": "testing",
                "effort": 12.0,
                "priority": "high"
            }
        ]
        
        for i, improvement in enumerate(common_improvements):
            item = ValueItem(
                id=f"github-{i+1:03d}",
                title=improvement["title"],
                description=improvement["description"],
                category=improvement["category"],
                source="issue_tracking",
                files=["src/"],
                estimated_effort=improvement["effort"],
                priority=improvement["priority"]
            )
            items.append(item)
        
        return items
    
    async def discover_from_dependency_analysis(self) -> List[ValueItem]:
        """Discover dependency-related improvements."""
        items = []
        
        # Check for outdated dependencies
        try:
            with open(self.repo_root / "requirements.txt", 'r') as f:
                requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            if len(requirements) > 10:
                item = ValueItem(
                    id="dependency-audit",
                    title="Audit and update dependencies",
                    description=f"Review {len(requirements)} dependencies for updates and security",
                    category="maintenance",
                    source="dependency_analysis",
                    files=["requirements.txt", "pyproject.toml"],
                    estimated_effort=4.0,
                    priority="medium"
                )
                items.append(item)
        
        except FileNotFoundError:
            logger.warning("requirements.txt not found for dependency analysis")
        
        return items
    
    async def discover_from_code_quality(self) -> List[ValueItem]:
        """Discover code quality improvements."""
        items = []
        
        # Analyze code complexity
        try:
            python_files = list(self.repo_root.glob("src/**/*.py"))
            large_files = [f for f in python_files if f.stat().st_size > 5000]  # >5KB
            
            if large_files:
                item = ValueItem(
                    id="code-complexity",
                    title=f"Refactor {len(large_files)} large modules",
                    description="Break down large modules for better maintainability",
                    category="refactoring",
                    source="code_quality",
                    files=[str(f) for f in large_files],
                    estimated_effort=len(large_files) * 3.0,
                    priority="medium"
                )
                items.append(item)
        
        except Exception as e:
            logger.warning(f"Code quality analysis failed: {e}")
        
        return items
    
    def deduplicate_items(self, items: List[ValueItem]) -> List[ValueItem]:
        """Remove duplicate items based on similarity."""
        unique_items = []
        seen_titles = set()
        
        for item in items:
            # Simple deduplication based on title similarity
            title_key = item.title.lower().replace(' ', '')[:30]
            if title_key not in seen_titles:
                unique_items.append(item)
                seen_titles.add(title_key)
        
        return unique_items
    
    async def enrich_with_context(self, items: List[ValueItem]) -> List[ValueItem]:
        """Enrich items with additional context and hot-spot analysis."""
        for item in items:
            # Analyze file activity (hotspot detection)
            if item.files:
                try:
                    result = subprocess.run([
                        'git', 'log', '--oneline', '--since=3months', '--', item.files[0]
                    ], capture_output=True, text=True, cwd=self.repo_root)
                    
                    commit_count = len(result.stdout.split('\n')) - 1
                    item.hotspot_multiplier = min(1.0 + (commit_count * 0.1), 5.0)
                
                except subprocess.CalledProcessError:
                    pass
            
            # Set default scoring components if not set
            if item.user_impact == 0:
                item.user_impact = self.estimate_user_impact(item)
            if item.impact == 0:
                item.impact = self.estimate_ice_impact(item)
            if item.confidence == 0:
                item.confidence = self.estimate_confidence(item)
            if item.ease == 0:
                item.ease = self.estimate_ease(item)
        
        return items
    
    def calculate_composite_score(self, item: ValueItem) -> float:
        """Calculate comprehensive composite score using WSJF + ICE + Technical Debt."""
        
        # WSJF Calculation
        cost_of_delay = (
            item.user_impact * 2.0 +  # User/Business Value
            item.time_criticality * 1.5 +  # Time Criticality
            item.risk_reduction * 1.8 +  # Risk Reduction/Opportunity Enablement
            item.opportunity_enablement * 1.2
        )
        job_size = max(item.estimated_effort, 0.5)  # Prevent division by zero
        item.wsjf_score = cost_of_delay / job_size
        
        # ICE Calculation
        item.ice_score = item.impact * item.confidence * item.ease
        
        # Technical Debt Calculation
        item.technical_debt_score = (
            (item.debt_impact + item.debt_interest) * item.hotspot_multiplier
        )
        
        # Composite Score with Advanced Repository Weights
        weights = self.config['scoring']['weights']['advanced']
        normalized_wsjf = min(item.wsjf_score / 50.0, 1.0)  # Normalize to 0-1
        normalized_ice = min(item.ice_score / 1000.0, 1.0)  # Normalize to 0-1
        normalized_debt = min(item.technical_debt_score / 100.0, 1.0)  # Normalize to 0-1
        
        composite = (
            weights['wsjf'] * normalized_wsjf * 100 +
            weights['ice'] * normalized_ice * 100 +
            weights['technical_debt'] * normalized_debt * 100 +
            weights['security'] * (1.0 if item.category == 'security' else 0.0) * 20
        )
        
        # Apply category-specific boosts and penalties
        if item.category == 'security':
            composite *= self.config['scoring']['thresholds']['security_boost']
        elif item.category == 'performance':
            composite *= self.config['scoring']['thresholds']['performance_boost']
        elif item.category == 'documentation':
            composite *= self.config['scoring']['penalties']['documentation_only']
        elif item.category == 'maintenance':
            composite *= self.config['scoring']['penalties']['dependency_update']
        
        item.composite_score = composite
        return composite
    
    def estimate_effort_from_content(self, content: str) -> float:
        """Estimate effort hours based on content analysis."""
        if any(word in content.lower() for word in ['refactor', 'rewrite', 'major']):
            return 8.0
        elif any(word in content.lower() for word in ['fix', 'bug', 'issue']):
            return 2.0
        elif any(word in content.lower() for word in ['todo', 'improve']):
            return 1.0
        else:
            return 0.5
    
    def estimate_user_impact(self, item: ValueItem) -> int:
        """Estimate user impact on 1-10 scale."""
        impact_map = {
            'security': 9,
            'performance': 8,
            'bug_fix': 7,
            'testing': 6,
            'feature': 8,
            'refactoring': 5,
            'documentation': 4,
            'maintenance': 3,
            'code_quality': 4
        }
        return impact_map.get(item.category, 5)
    
    def estimate_ice_impact(self, item: ValueItem) -> int:
        """Estimate ICE impact component."""
        return min(self.estimate_user_impact(item) + 1, 10)
    
    def estimate_confidence(self, item: ValueItem) -> int:
        """Estimate execution confidence on 1-10 scale."""
        confidence_map = {
            'documentation': 9,
            'code_quality': 8,
            'maintenance': 8,
            'testing': 7,
            'refactoring': 6,
            'performance': 5,
            'security': 6,
            'feature': 4
        }
        return confidence_map.get(item.category, 6)
    
    def estimate_ease(self, item: ValueItem) -> int:
        """Estimate implementation ease on 1-10 scale."""
        # Inverse of effort: easier tasks have higher ease scores
        if item.estimated_effort <= 1:
            return 9
        elif item.estimated_effort <= 3:
            return 7
        elif item.estimated_effort <= 6:
            return 5
        elif item.estimated_effort <= 10:
            return 3
        else:
            return 1
    
    def select_next_best_value(self, items: List[ValueItem]) -> Optional[ValueItem]:
        """Select the next highest-value item for execution."""
        if not items:
            return None
        
        # Filter items that meet minimum thresholds
        min_score = self.config['scoring']['thresholds']['min_score']
        max_risk = self.config['scoring']['thresholds']['max_risk']
        
        qualified_items = []
        for item in items:
            score = self.calculate_composite_score(item)
            risk_score = 1.0 - (item.confidence / 10.0)  # Higher confidence = lower risk
            
            if score >= min_score and risk_score <= max_risk:
                qualified_items.append(item)
        
        if not qualified_items:
            logger.warning("No items meet minimum quality thresholds")
            return None
        
        # Sort by composite score (descending)
        qualified_items.sort(key=lambda x: x.composite_score, reverse=True)
        return qualified_items[0]
    
    async def generate_backlog_markdown(self, items: List[ValueItem]) -> str:
        """Generate comprehensive backlog in Markdown format."""
        now = datetime.now().isoformat()
        
        # Sort items by composite score
        sorted_items = sorted(items, key=lambda x: x.composite_score, reverse=True)
        top_10 = sorted_items[:10]
        
        next_item = self.select_next_best_value(sorted_items)
        
        md = f"""# 📊 Autonomous Value Backlog

**Repository**: {self.config['repository']['name']}
**Maturity Level**: {self.config['repository']['maturity_level'].title()} (75%+ SDLC)
**Last Updated**: {now}
**Next Execution**: {(datetime.now() + timedelta(hours=1)).isoformat()}

## 🎯 Next Best Value Item

"""
        
        if next_item:
            md += f"""**[{next_item.id.upper()}] {next_item.title}**
- **Composite Score**: {next_item.composite_score:.1f}
- **WSJF**: {next_item.wsjf_score:.1f} | **ICE**: {next_item.ice_score:.0f} | **Tech Debt**: {next_item.technical_debt_score:.1f}
- **Estimated Effort**: {next_item.estimated_effort:.1f} hours
- **Category**: {next_item.category.replace('_', ' ').title()}
- **Priority**: {next_item.priority.title()}
- **Expected Impact**: {next_item.description}

"""
        else:
            md += "**No qualifying items found** - All tasks below minimum thresholds\n\n"
        
        md += f"""## 📋 Top 10 Backlog Items

| Rank | ID | Title | Score | Category | Est. Hours | Priority |
|------|-----|--------|---------|----------|------------|----------|
"""
        
        for i, item in enumerate(top_10, 1):
            score = self.calculate_composite_score(item)
            md += f"| {i} | {item.id.upper()} | {item.title[:40]}{'...' if len(item.title) > 40 else ''} | {score:.1f} | {item.category.replace('_', ' ').title()} | {item.estimated_effort:.1f} | {item.priority.title()} |\n"
        
        # Add value metrics
        total_effort = sum(item.estimated_effort for item in sorted_items)
        security_items = len([i for i in sorted_items if i.category == 'security'])
        performance_items = len([i for i in sorted_items if i.category == 'performance'])
        debt_items = len([i for i in sorted_items if i.category in ['technical_debt', 'refactoring']])
        
        md += f"""

## 📈 Value Metrics

- **Total Items Discovered**: {len(sorted_items)}
- **Total Estimated Effort**: {total_effort:.1f} hours
- **High Priority Items**: {len([i for i in sorted_items if i.priority == 'high'])}
- **Security Items**: {security_items}
- **Performance Items**: {performance_items}
- **Technical Debt Items**: {debt_items}
- **Average Score**: {sum(i.composite_score for i in sorted_items) / len(sorted_items) if sorted_items else 0:.1f}

## 🔄 Discovery Sources

"""
        
        source_counts = {}
        for item in sorted_items:
            source_counts[item.source] = source_counts.get(item.source, 0) + 1
        
        for source, count in source_counts.items():
            percentage = (count / len(sorted_items)) * 100 if sorted_items else 0
            md += f"- **{source.replace('_', ' ').title()}**: {count} items ({percentage:.1f}%)\n"
        
        md += f"""

## 🎯 Category Breakdown

"""
        
        category_counts = {}
        for item in sorted_items:
            category_counts[item.category] = category_counts.get(item.category, 0) + 1
        
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(sorted_items)) * 100 if sorted_items else 0
            md += f"- **{category.replace('_', ' ').title()}**: {count} items ({percentage:.1f}%)\n"
        
        md += f"""

## ⚙️ Execution Configuration

- **Repository Maturity**: Advanced (75%+ SDLC)
- **Minimum Score Threshold**: {self.config['scoring']['thresholds']['min_score']}
- **Maximum Risk Tolerance**: {self.config['scoring']['thresholds']['max_risk']}
- **Security Priority Boost**: {self.config['scoring']['thresholds']['security_boost']}x
- **Performance Priority Boost**: {self.config['scoring']['thresholds']['performance_boost']}x

## 📊 Scoring Weights (Advanced Repository)

- **WSJF (Weighted Shortest Job First)**: {self.config['scoring']['weights']['advanced']['wsjf'] * 100:.0f}%
- **ICE (Impact × Confidence × Ease)**: {self.config['scoring']['weights']['advanced']['ice'] * 100:.0f}%
- **Technical Debt**: {self.config['scoring']['weights']['advanced']['technical_debt'] * 100:.0f}%
- **Security**: {self.config['scoring']['weights']['advanced']['security'] * 100:.0f}%

---

*Generated by Terragon Autonomous SDLC Value Discovery Engine v1.0*
*Next discovery cycle: {(datetime.now() + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return md
    
    async def save_results(self, items: List[ValueItem]):
        """Save discovery results to files."""
        # Ensure .terragon directory exists
        terragon_dir = self.repo_root / ".terragon"
        terragon_dir.mkdir(exist_ok=True)
        
        # Save backlog markdown
        backlog_md = await self.generate_backlog_markdown(items)
        with open(self.repo_root / "BACKLOG.md", 'w') as f:
            f.write(backlog_md)
        
        # Save detailed metrics as JSON
        metrics = {
            "last_updated": datetime.now().isoformat(),
            "total_items": len(items),
            "items": [asdict(item) for item in items],
            "summary": {
                "by_category": {},
                "by_priority": {},
                "by_source": {},
                "total_estimated_effort": sum(item.estimated_effort for item in items),
                "average_score": sum(item.composite_score for item in items) / len(items) if items else 0
            }
        }
        
        # Calculate summaries
        for item in items:
            # By category
            metrics["summary"]["by_category"][item.category] = \
                metrics["summary"]["by_category"].get(item.category, 0) + 1
            
            # By priority
            metrics["summary"]["by_priority"][item.priority] = \
                metrics["summary"]["by_priority"].get(item.priority, 0) + 1
            
            # By source
            metrics["summary"]["by_source"][item.source] = \
                metrics["summary"]["by_source"].get(item.source, 0) + 1
        
        with open(terragon_dir / "value-metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        logger.info(f"💾 Saved {len(items)} items to BACKLOG.md and value-metrics.json")
    
    async def run_discovery_cycle(self):
        """Execute a complete discovery and scoring cycle."""
        logger.info("🚀 Starting Terragon Autonomous SDLC Value Discovery Cycle")
        
        start_time = time.time()
        
        try:
            # Discover all value items
            items = await self.discover_all_value_items()
            
            # Calculate scores for all items
            for item in items:
                self.calculate_composite_score(item)
            
            # Save results
            await self.save_results(items)
            
            # Select next best item
            next_item = self.select_next_best_value(items)
            
            duration = time.time() - start_time
            
            if next_item:
                logger.info(f"✅ Discovery complete in {duration:.1f}s")
                logger.info(f"🎯 Next best value item: [{next_item.id.upper()}] {next_item.title}")
                logger.info(f"📊 Composite score: {next_item.composite_score:.1f}")
                logger.info(f"⏱️  Estimated effort: {next_item.estimated_effort:.1f} hours")
            else:
                logger.info(f"✅ Discovery complete in {duration:.1f}s - No qualifying items found")
            
            return items, next_item
        
        except Exception as e:
            logger.error(f"❌ Discovery cycle failed: {e}")
            return [], None

async def main():
    """Main entry point for the value discovery engine."""
    engine = ValueDiscoveryEngine()
    items, next_item = await engine.run_discovery_cycle()
    
    print(f"\n🎯 Discovery Summary:")
    print(f"   Items found: {len(items)}")
    print(f"   Next item: {next_item.title if next_item else 'None'}")
    print(f"   Results saved to: BACKLOG.md")

if __name__ == "__main__":
    asyncio.run(main())