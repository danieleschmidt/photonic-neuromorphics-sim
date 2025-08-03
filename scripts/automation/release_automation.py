#!/usr/bin/env python3
"""
Release Automation Script

This script automates the release process including version bumping,
changelog generation, tagging, and GitHub release creation.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
import semver


class ReleaseAutomator:
    """Handles automated release management with semantic versioning."""
    
    def __init__(self, repo_path: Path = None):
        self.repo_path = repo_path or Path.cwd()
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo_name = os.getenv("GITHUB_REPOSITORY")
        
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
    
    def get_current_version(self) -> Optional[str]:
        """Get current version from git tags."""
        result = self.run_command(["git", "tag", "--sort=-version:refname"])
        if result["success"]:
            tags = [tag.strip() for tag in result["stdout"].split('\n') if tag.strip()]
            for tag in tags:
                # Look for semantic version tags
                if re.match(r'^v?\d+\.\d+\.\d+', tag):
                    return tag.lstrip('v')
        return "0.0.0"
    
    def analyze_commits_since_tag(self, since_tag: str = None) -> Dict[str, List[str]]:
        """Analyze commits since last tag to determine version bump type."""
        if since_tag:
            commit_range = f"{since_tag}..HEAD"
        else:
            commit_range = "HEAD"
        
        result = self.run_command(["git", "log", "--oneline", commit_range])
        if not result["success"]:
            return {"patch": [], "minor": [], "major": [], "other": []}
        
        commits = result["stdout"].strip().split('\n') if result["stdout"].strip() else []
        
        categorized = {
            "patch": [],
            "minor": [],
            "major": [],
            "other": []
        }
        
        for commit in commits:
            if not commit.strip():
                continue
                
            commit_msg = commit.split(' ', 1)[1] if ' ' in commit else commit
            
            # Categorize based on conventional commits
            if self._is_breaking_change(commit_msg):
                categorized["major"].append(commit)
            elif self._is_feature(commit_msg):
                categorized["minor"].append(commit)
            elif self._is_fix_or_patch(commit_msg):
                categorized["patch"].append(commit)
            else:
                categorized["other"].append(commit)
        
        return categorized
    
    def _is_breaking_change(self, commit_msg: str) -> bool:
        """Check if commit represents a breaking change."""
        breaking_indicators = [
            "BREAKING CHANGE", "breaking change", "!:",
            "major:", "MAJOR:"
        ]
        return any(indicator in commit_msg for indicator in breaking_indicators)
    
    def _is_feature(self, commit_msg: str) -> bool:
        """Check if commit represents a feature."""
        feature_indicators = [
            "feat:", "feature:", "minor:", "add:", "implement:"
        ]
        return any(commit_msg.lower().startswith(indicator.lower()) for indicator in feature_indicators)
    
    def _is_fix_or_patch(self, commit_msg: str) -> bool:
        """Check if commit represents a fix or patch."""
        fix_indicators = [
            "fix:", "patch:", "bugfix:", "hotfix:", "docs:", "style:",
            "refactor:", "test:", "chore:", "perf:", "ci:", "build:"
        ]
        return any(commit_msg.lower().startswith(indicator.lower()) for indicator in fix_indicators)
    
    def determine_version_bump(self, commit_analysis: Dict[str, List[str]]) -> str:
        """Determine the type of version bump needed."""
        if commit_analysis["major"]:
            return "major"
        elif commit_analysis["minor"]:
            return "minor"
        elif commit_analysis["patch"]:
            return "patch"
        else:
            return "patch"  # Default to patch for any changes
    
    def bump_version(self, current_version: str, bump_type: str) -> str:
        """Bump version according to semantic versioning."""
        try:
            if bump_type == "major":
                return semver.bump_major(current_version)
            elif bump_type == "minor":
                return semver.bump_minor(current_version)
            elif bump_type == "patch":
                return semver.bump_patch(current_version)
            else:
                raise ValueError(f"Invalid bump type: {bump_type}")
        except ValueError:
            # Fallback for non-semver versions
            parts = current_version.split('.')
            if len(parts) != 3:
                return "1.0.0"
            
            major, minor, patch = map(int, parts)
            
            if bump_type == "major":
                return f"{major + 1}.0.0"
            elif bump_type == "minor":
                return f"{major}.{minor + 1}.0"
            else:  # patch
                return f"{major}.{minor}.{patch + 1}"
    
    def update_version_files(self, new_version: str):
        """Update version in project files."""
        version_files = [
            ("pyproject.toml", r'version = ".*"', f'version = "{new_version}"'),
            ("setup.py", r'version=".*"', f'version="{new_version}"'),
            ("package.json", r'"version": ".*"', f'"version": "{new_version}"'),
            ("__init__.py", r'__version__ = ".*"', f'__version__ = "{new_version}"')
        ]
        
        updated_files = []
        
        for file_name, pattern, replacement in version_files:
            file_path = self.repo_path / file_name
            if file_path.exists():
                with open(file_path, 'r') as f:
                    content = f.read()
                
                if re.search(pattern, content):
                    new_content = re.sub(pattern, replacement, content)
                    with open(file_path, 'w') as f:
                        f.write(new_content)
                    updated_files.append(file_name)
        
        # Also check in src/ for __init__.py
        src_init = self.repo_path / "src" / "photonic_neuromorphics" / "__init__.py"
        if src_init.exists():
            with open(src_init, 'r') as f:
                content = f.read()
            
            if '__version__' in content:
                new_content = re.sub(
                    r'__version__ = ".*"', 
                    f'__version__ = "{new_version}"', 
                    content
                )
                with open(src_init, 'w') as f:
                    f.write(new_content)
                updated_files.append("src/photonic_neuromorphics/__init__.py")
        
        return updated_files
    
    def generate_changelog(self, version: str, commit_analysis: Dict[str, List[str]], 
                          previous_version: str = None) -> str:
        """Generate changelog for the release."""
        lines = [
            f"# Changelog - Version {version}",
            f"Released: {datetime.now().strftime('%Y-%m-%d')}",
            ""
        ]
        
        if previous_version:
            lines.append(f"Changes since {previous_version}:")
        else:
            lines.append("Changes in this release:")
        
        lines.append("")
        
        # Breaking Changes
        if commit_analysis["major"]:
            lines.append("## 🚨 Breaking Changes")
            for commit in commit_analysis["major"]:
                lines.append(f"- {commit}")
            lines.append("")
        
        # New Features
        if commit_analysis["minor"]:
            lines.append("## 🎉 New Features")
            for commit in commit_analysis["minor"]:
                lines.append(f"- {commit}")
            lines.append("")
        
        # Bug Fixes
        if commit_analysis["patch"]:
            lines.append("## 🐛 Bug Fixes & Improvements")
            for commit in commit_analysis["patch"]:
                lines.append(f"- {commit}")
            lines.append("")
        
        # Other Changes
        if commit_analysis["other"]:
            lines.append("## 🔧 Other Changes")
            for commit in commit_analysis["other"]:
                lines.append(f"- {commit}")
            lines.append("")
        
        return '\n'.join(lines)
    
    def update_changelog_file(self, version: str, changelog_entry: str):
        """Update CHANGELOG.md file with new entry."""
        changelog_path = self.repo_path / "CHANGELOG.md"
        
        if changelog_path.exists():
            with open(changelog_path, 'r') as f:
                existing_content = f.read()
        else:
            existing_content = "# Changelog\n\nAll notable changes to this project will be documented in this file.\n\n"
        
        # Insert new changelog entry after the header
        lines = existing_content.split('\n')
        header_end = 0
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith('#'):
                header_end = i
                break
        
        new_lines = lines[:header_end] + [''] + changelog_entry.split('\n') + [''] + lines[header_end:]
        
        with open(changelog_path, 'w') as f:
            f.write('\n'.join(new_lines))
    
    def create_git_tag(self, version: str, changelog: str) -> bool:
        """Create an annotated git tag for the release."""
        tag_name = f"v{version}"
        
        # Create annotated tag with changelog as message
        result = self.run_command([
            "git", "tag", "-a", tag_name, "-m", f"Release {version}\n\n{changelog}"
        ])
        
        if result["success"]:
            print(f"✅ Created tag: {tag_name}")
            return True
        else:
            print(f"❌ Failed to create tag: {result.get('stderr', 'Unknown error')}")
            return False
    
    def push_changes(self, version: str) -> bool:
        """Push commits and tags to remote repository."""
        # Push commits
        result = self.run_command(["git", "push", "origin", "HEAD"])
        if not result["success"]:
            print(f"❌ Failed to push commits: {result.get('stderr', 'Unknown error')}")
            return False
        
        # Push tags
        result = self.run_command(["git", "push", "origin", f"v{version}"])
        if not result["success"]:
            print(f"❌ Failed to push tag: {result.get('stderr', 'Unknown error')}")
            return False
        
        print("✅ Pushed changes and tags to remote")
        return True
    
    def create_github_release(self, version: str, changelog: str) -> Optional[str]:
        """Create a GitHub release."""
        if not self.github_token or not self.repo_name:
            print("⚠️ GitHub token or repository name not configured, skipping GitHub release")
            return None
        
        tag_name = f"v{version}"
        
        # Create release via GitHub API
        try:
            url = f"https://api.github.com/repos/{self.repo_name}/releases"
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            data = {
                "tag_name": tag_name,
                "name": f"Release {version}",
                "body": changelog,
                "draft": False,
                "prerelease": False
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            if response.status_code == 201:
                release_data = response.json()
                release_url = release_data["html_url"]
                print(f"✅ Created GitHub release: {release_url}")
                return release_url
            else:
                print(f"❌ Failed to create GitHub release: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Error creating GitHub release: {e}")
        
        return None
    
    def run_pre_release_checks(self) -> bool:
        """Run pre-release validation checks."""
        print("🔍 Running pre-release checks...")
        
        checks_passed = True
        
        # Check git status
        result = self.run_command(["git", "status", "--porcelain"])
        if result["success"] and result["stdout"].strip():
            print("❌ Working directory is not clean")
            checks_passed = False
        else:
            print("✅ Working directory is clean")
        
        # Check if on main/master branch
        result = self.run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        if result["success"]:
            branch = result["stdout"].strip()
            if branch not in ["main", "master"]:
                print(f"⚠️ Not on main/master branch (current: {branch})")
        
        # Run tests if available
        if (self.repo_path / "pytest.ini").exists():
            print("🧪 Running tests...")
            result = self.run_command(["pytest", "--tb=short", "-q"])
            if result["success"]:
                print("✅ All tests passed")
            else:
                print("❌ Tests failed")
                checks_passed = False
        
        # Check build
        if (self.repo_path / "pyproject.toml").exists():
            print("🏗️ Testing package build...")
            result = self.run_command(["python", "-m", "build", "--outdir", "/tmp/release-test"])
            if result["success"]:
                print("✅ Package builds successfully")
            else:
                print("❌ Package build failed")
                checks_passed = False
        
        return checks_passed
    
    def run_release_workflow(self, bump_type: str = None, dry_run: bool = False):
        """Run the complete release workflow."""
        print("🚀 Starting release automation workflow...")
        
        # Pre-release checks
        if not self.run_pre_release_checks():
            print("❌ Pre-release checks failed!")
            if not dry_run:
                return False
        
        # Get current version
        current_version = self.get_current_version()
        print(f"📋 Current version: {current_version}")
        
        # Analyze commits since last release
        last_tag = f"v{current_version}" if current_version != "0.0.0" else None
        commit_analysis = self.analyze_commits_since_tag(last_tag)
        
        # Determine version bump
        if not bump_type:
            bump_type = self.determine_version_bump(commit_analysis)
        
        print(f"📈 Detected changes: {sum(len(v) for v in commit_analysis.values())} commits")
        print(f"🎯 Version bump type: {bump_type}")
        
        # Calculate new version
        new_version = self.bump_version(current_version, bump_type)
        print(f"🔖 New version: {new_version}")
        
        if dry_run:
            print("🔍 Dry run mode - no changes will be made")
            changelog = self.generate_changelog(new_version, commit_analysis, current_version)
            print("\n📝 Generated changelog:")
            print(changelog)
            return True
        
        # Update version files
        updated_files = self.update_version_files(new_version)
        if updated_files:
            print(f"📝 Updated version in: {', '.join(updated_files)}")
        
        # Generate changelog
        changelog = self.generate_changelog(new_version, commit_analysis, current_version)
        self.update_changelog_file(new_version, changelog)
        print("📝 Updated CHANGELOG.md")
        
        # Commit changes
        result = self.run_command(["git", "add", "-A"])
        if not result["success"]:
            print("❌ Failed to stage changes")
            return False
        
        commit_msg = f"chore(release): bump version to {new_version}"
        result = self.run_command(["git", "commit", "-m", commit_msg])
        if not result["success"]:
            print("❌ Failed to commit changes")
            return False
        
        print(f"✅ Committed version bump")
        
        # Create git tag
        if not self.create_git_tag(new_version, changelog):
            return False
        
        # Push changes
        if not self.push_changes(new_version):
            return False
        
        # Create GitHub release
        self.create_github_release(new_version, changelog)
        
        print(f"🎉 Release {new_version} completed successfully!")
        return True


def main():
    parser = argparse.ArgumentParser(description="Automate release process")
    parser.add_argument("--bump", choices=["major", "minor", "patch"],
                        help="Force specific version bump type")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without making changes")
    parser.add_argument("--repo-path", help="Repository path (default: current directory)")
    
    args = parser.parse_args()
    
    repo_path = Path(args.repo_path) if args.repo_path else Path.cwd()
    automator = ReleaseAutomator(repo_path)
    
    try:
        success = automator.run_release_workflow(
            bump_type=args.bump,
            dry_run=args.dry_run
        )
        
        if not success and not args.dry_run:
            sys.exit(1)
    
    except Exception as e:
        print(f"❌ Error in release automation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()