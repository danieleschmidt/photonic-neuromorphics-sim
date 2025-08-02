#!/usr/bin/env python3
"""
Automated release management system for photonic neuromorphics simulation platform.
Handles version bumping, changelog generation, and release automation.
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import requests
import semantic_version


class ReleaseAutomation:
    """Automated release management system."""
    
    def __init__(self, repo_path: str = "."):
        """Initialize release automation."""
        self.repo_path = Path(repo_path)
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo_name = os.getenv("GITHUB_REPOSITORY", "danieleschmidt/photonic-neuromorphics-sim")
        self.current_version = self._get_current_version()
        
    def _get_current_version(self) -> str:
        """Get current version from various sources."""
        # Try pyproject.toml first
        pyproject_path = self.repo_path / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path, 'r') as f:
                content = f.read()
                version_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
                if version_match:
                    return version_match.group(1)
        
        # Try setup.py
        setup_path = self.repo_path / "setup.py"
        if setup_path.exists():
            with open(setup_path, 'r') as f:
                content = f.read()
                version_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
                if version_match:
                    return version_match.group(1)
        
        # Try package __init__.py
        init_path = self.repo_path / "src" / "photonic_neuromorphics" / "__init__.py"
        if init_path.exists():
            with open(init_path, 'r') as f:
                content = f.read()
                version_match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
                if version_match:
                    return version_match.group(1)
        
        # Try git tags
        try:
            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            if result.returncode == 0:
                tag = result.stdout.strip()
                return tag.lstrip('v')
        except Exception:
            pass
        
        return "0.1.0"  # Default version
    
    def analyze_commits_for_release_type(self, since_version: Optional[str] = None) -> str:
        """Analyze commits to determine release type (major, minor, patch)."""
        if since_version:
            git_range = f"v{since_version}..HEAD"
        else:
            git_range = "HEAD~10..HEAD"  # Last 10 commits if no version
        
        try:
            result = subprocess.run(
                ["git", "log", git_range, "--oneline"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            if result.returncode != 0:
                return "patch"
            
            commits = result.stdout.strip().split('\n')
            if not commits or commits == ['']:
                return "patch"
            
            has_breaking = False
            has_feature = False
            
            for commit in commits:
                commit_lower = commit.lower()
                
                # Check for breaking changes
                if any(keyword in commit_lower for keyword in [
                    "breaking", "break:", "major:", "!:", "breaking change"
                ]):
                    has_breaking = True
                
                # Check for features
                if any(keyword in commit_lower for keyword in [
                    "feat:", "feature:", "add:", "new:"
                ]):
                    has_feature = True
            
            if has_breaking:
                return "major"
            elif has_feature:
                return "minor"
            else:
                return "patch"
                
        except Exception:
            return "patch"
    
    def bump_version(self, bump_type: str) -> str:
        """Bump version based on type (major, minor, patch)."""
        try:
            current_sem_ver = semantic_version.Version(self.current_version)
            
            if bump_type == "major":
                new_version = current_sem_ver.next_major()
            elif bump_type == "minor":
                new_version = current_sem_ver.next_minor()
            elif bump_type == "patch":
                new_version = current_sem_ver.next_patch()
            else:
                raise ValueError(f"Invalid bump type: {bump_type}")
            
            return str(new_version)
            
        except Exception as e:
            print(f"Error bumping version: {e}")
            # Fallback to manual version bumping
            parts = self.current_version.split('.')
            if len(parts) >= 3:
                major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
                
                if bump_type == "major":
                    return f"{major + 1}.0.0"
                elif bump_type == "minor":
                    return f"{major}.{minor + 1}.0"
                else:  # patch
                    return f"{major}.{minor}.{patch + 1}"
            else:
                return "0.1.0"
    
    def update_version_files(self, new_version: str) -> List[str]:
        """Update version in all relevant files."""
        updated_files = []
        
        # Update pyproject.toml
        pyproject_path = self.repo_path / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path, 'r') as f:
                content = f.read()
            
            updated_content = re.sub(
                r'version\s*=\s*["\'][^"\']+["\']',
                f'version = "{new_version}"',
                content
            )
            
            if updated_content != content:
                with open(pyproject_path, 'w') as f:
                    f.write(updated_content)
                updated_files.append(str(pyproject_path))
        
        # Update setup.py
        setup_path = self.repo_path / "setup.py"
        if setup_path.exists():
            with open(setup_path, 'r') as f:
                content = f.read()
            
            updated_content = re.sub(
                r'version\s*=\s*["\'][^"\']+["\']',
                f'version="{new_version}"',
                content
            )
            
            if updated_content != content:
                with open(setup_path, 'w') as f:
                    f.write(updated_content)
                updated_files.append(str(setup_path))
        
        # Update package __init__.py
        init_path = self.repo_path / "src" / "photonic_neuromorphics" / "__init__.py"
        if init_path.exists():
            with open(init_path, 'r') as f:
                content = f.read()
            
            updated_content = re.sub(
                r'__version__\s*=\s*["\'][^"\']+["\']',
                f'__version__ = "{new_version}"',
                content
            )
            
            if updated_content != content:
                with open(init_path, 'w') as f:
                    f.write(updated_content)
                updated_files.append(str(init_path))
        
        return updated_files
    
    def generate_changelog(self, new_version: str, since_version: Optional[str] = None) -> str:
        """Generate changelog for the release."""
        changelog_content = f"## [{new_version}] - {datetime.now().strftime('%Y-%m-%d')}\n\n"
        
        if since_version:
            git_range = f"v{since_version}..HEAD"
        else:
            git_range = "HEAD~20..HEAD"
        
        try:
            result = subprocess.run(
                ["git", "log", git_range, "--oneline", "--no-merges"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            if result.returncode != 0:
                changelog_content += "- Initial release\n"
                return changelog_content
            
            commits = result.stdout.strip().split('\n')
            if not commits or commits == ['']:
                changelog_content += "- Maintenance release\n"
                return changelog_content
            
            # Categorize commits
            features = []
            fixes = []
            docs = []
            tests = []
            chores = []
            breaking = []
            
            for commit in commits:
                commit_msg = commit.split(' ', 1)[1] if ' ' in commit else commit
                commit_lower = commit_msg.lower()
                
                if any(keyword in commit_lower for keyword in ["breaking", "break:", "major:", "!:"]):
                    breaking.append(commit_msg)
                elif any(keyword in commit_lower for keyword in ["feat:", "feature:", "add:", "new:"]):
                    features.append(commit_msg)
                elif any(keyword in commit_lower for keyword in ["fix:", "bug:", "patch:"]):
                    fixes.append(commit_msg)
                elif any(keyword in commit_lower for keyword in ["docs:", "doc:", "documentation:"]):
                    docs.append(commit_msg)
                elif any(keyword in commit_lower for keyword in ["test:", "tests:"]):
                    tests.append(commit_msg)
                else:
                    chores.append(commit_msg)
            
            # Generate changelog sections
            if breaking:
                changelog_content += "### ⚠️ BREAKING CHANGES\n\n"
                for item in breaking:
                    changelog_content += f"- {item}\n"
                changelog_content += "\n"
            
            if features:
                changelog_content += "### ✨ Features\n\n"
                for item in features:
                    changelog_content += f"- {item}\n"
                changelog_content += "\n"
            
            if fixes:
                changelog_content += "### 🐛 Bug Fixes\n\n"
                for item in fixes:
                    changelog_content += f"- {item}\n"
                changelog_content += "\n"
            
            if docs:
                changelog_content += "### 📚 Documentation\n\n"
                for item in docs:
                    changelog_content += f"- {item}\n"
                changelog_content += "\n"
            
            if tests:
                changelog_content += "### 🧪 Tests\n\n"
                for item in tests:
                    changelog_content += f"- {item}\n"
                changelog_content += "\n"
            
            if chores:
                changelog_content += "### 🔧 Maintenance\n\n"
                for item in chores[:5]:  # Limit chores to 5 items
                    changelog_content += f"- {item}\n"
                if len(chores) > 5:
                    changelog_content += f"- ... and {len(chores) - 5} more maintenance items\n"
                changelog_content += "\n"
            
        except Exception as e:
            print(f"Error generating changelog: {e}")
            changelog_content += "- Release updates\n"
        
        return changelog_content
    
    def update_changelog_file(self, new_content: str) -> str:
        """Update CHANGELOG.md file with new release content."""
        changelog_path = self.repo_path / "CHANGELOG.md"
        
        if changelog_path.exists():
            with open(changelog_path, 'r') as f:
                existing_content = f.read()
            
            # Insert new content after header
            if "# Changelog" in existing_content:
                parts = existing_content.split("# Changelog", 1)
                if len(parts) == 2:
                    updated_content = f"# Changelog\n\n{new_content}\n{parts[1]}"
                else:
                    updated_content = f"# Changelog\n\n{new_content}\n{existing_content}"
            else:
                updated_content = f"# Changelog\n\n{new_content}\n{existing_content}"
        else:
            updated_content = f"# Changelog\n\n{new_content}"
        
        with open(changelog_path, 'w') as f:
            f.write(updated_content)
        
        return str(changelog_path)
    
    def run_pre_release_checks(self) -> Tuple[bool, List[str]]:
        """Run pre-release checks to ensure release readiness."""
        checks_passed = True
        issues = []
        
        print("Running pre-release checks...")
        
        # Check if working directory is clean
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            if result.stdout.strip():
                issues.append("Working directory is not clean")
                checks_passed = False
        except Exception:
            issues.append("Could not check git status")
            checks_passed = False
        
        # Check if tests pass
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "--tb=no", "-q"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            if result.returncode != 0:
                issues.append("Tests are failing")
                checks_passed = False
        except Exception:
            issues.append("Could not run tests")
        
        # Check if build succeeds
        try:
            result = subprocess.run(
                ["python", "-m", "build"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            if result.returncode != 0:
                issues.append("Build is failing")
                checks_passed = False
        except Exception:
            issues.append("Could not run build")
        
        # Check for security vulnerabilities
        try:
            result = subprocess.run(
                ["safety", "check"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            if result.returncode != 0:
                issues.append("Security vulnerabilities detected")
                checks_passed = False
        except Exception:
            pass  # Optional check
        
        return checks_passed, issues
    
    def create_release_commit(self, new_version: str, updated_files: List[str]) -> bool:
        """Create a release commit with version bump and changelog."""
        try:
            # Add updated files
            for file_path in updated_files:
                subprocess.run(
                    ["git", "add", file_path],
                    cwd=self.repo_path, check=True
                )
            
            # Create commit
            commit_message = f"release: bump version to {new_version}\n\n" \
                           f"- Update version files\n" \
                           f"- Update CHANGELOG.md\n" \
                           f"- Prepare for {new_version} release"
            
            subprocess.run(
                ["git", "commit", "-m", commit_message],
                cwd=self.repo_path, check=True
            )
            
            print(f"Created release commit for version {new_version}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error creating release commit: {e}")
            return False
    
    def create_git_tag(self, version: str, changelog: str) -> bool:
        """Create an annotated git tag for the release."""
        try:
            tag_message = f"Release {version}\n\n{changelog}"
            
            subprocess.run(
                ["git", "tag", "-a", f"v{version}", "-m", tag_message],
                cwd=self.repo_path, check=True
            )
            
            print(f"Created tag v{version}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error creating tag: {e}")
            return False
    
    def push_release(self, version: str) -> bool:
        """Push release commit and tag to remote."""
        try:
            # Push commit
            subprocess.run(
                ["git", "push", "origin", "main"],
                cwd=self.repo_path, check=True
            )
            
            # Push tag
            subprocess.run(
                ["git", "push", "origin", f"v{version}"],
                cwd=self.repo_path, check=True
            )
            
            print(f"Pushed release {version} to remote")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error pushing release: {e}")
            return False
    
    def create_github_release(self, version: str, changelog: str, is_prerelease: bool = False) -> Optional[str]:
        """Create a GitHub release."""
        if not self.github_token:
            print("No GitHub token available, skipping GitHub release")
            return None
        
        try:
            headers = {
                "Authorization": f"token {self.github_token}",
                "Content-Type": "application/json"
            }
            
            release_data = {
                "tag_name": f"v{version}",
                "name": f"Release {version}",
                "body": changelog,
                "draft": False,
                "prerelease": is_prerelease
            }
            
            url = f"https://api.github.com/repos/{self.repo_name}/releases"
            response = requests.post(url, headers=headers, json=release_data)
            
            if response.status_code == 201:
                release_url = response.json()["html_url"]
                print(f"Created GitHub release: {release_url}")
                return release_url
            else:
                print(f"Failed to create GitHub release: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Error creating GitHub release: {e}")
            return None
    
    def publish_to_pypi(self, test_pypi: bool = False) -> bool:
        """Publish package to PyPI."""
        try:
            # Build package
            result = subprocess.run(
                ["python", "-m", "build"],
                cwd=self.repo_path, check=True,
                capture_output=True, text=True
            )
            
            # Upload to PyPI
            if test_pypi:
                upload_cmd = ["python", "-m", "twine", "upload", "--repository", "testpypi", "dist/*"]
            else:
                upload_cmd = ["python", "-m", "twine", "upload", "dist/*"]
            
            result = subprocess.run(
                upload_cmd,
                cwd=self.repo_path, check=True,
                capture_output=True, text=True
            )
            
            repository = "Test PyPI" if test_pypi else "PyPI"
            print(f"Successfully published to {repository}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error publishing to PyPI: {e}")
            return False
    
    def create_release(self, bump_type: str = "auto", dry_run: bool = False, 
                      publish_pypi: bool = False, test_pypi: bool = False) -> Dict[str, any]:
        """Create a complete release."""
        print(f"Starting release process...")
        print(f"Current version: {self.current_version}")
        
        # Determine bump type
        if bump_type == "auto":
            bump_type = self.analyze_commits_for_release_type()
            print(f"Auto-detected bump type: {bump_type}")
        
        # Calculate new version
        new_version = self.bump_version(bump_type)
        print(f"New version: {new_version}")
        
        if dry_run:
            print("DRY RUN - No changes will be made")
            changelog = self.generate_changelog(new_version)
            return {
                "success": True,
                "version": new_version,
                "bump_type": bump_type,
                "changelog": changelog,
                "dry_run": True
            }
        
        # Run pre-release checks
        checks_passed, issues = self.run_pre_release_checks()
        if not checks_passed:
            print("Pre-release checks failed:")
            for issue in issues:
                print(f"  - {issue}")
            return {
                "success": False,
                "error": "Pre-release checks failed",
                "issues": issues
            }
        
        try:
            # Update version files
            updated_files = self.update_version_files(new_version)
            print(f"Updated version files: {updated_files}")
            
            # Generate and update changelog
            changelog = self.generate_changelog(new_version, self.current_version)
            changelog_file = self.update_changelog_file(changelog)
            updated_files.append(changelog_file)
            
            # Create release commit
            if not self.create_release_commit(new_version, updated_files):
                raise Exception("Failed to create release commit")
            
            # Create git tag
            if not self.create_git_tag(new_version, changelog):
                raise Exception("Failed to create git tag")
            
            # Push to remote
            if not self.push_release(new_version):
                raise Exception("Failed to push release")
            
            # Create GitHub release
            github_url = self.create_github_release(new_version, changelog)
            
            # Publish to PyPI if requested
            pypi_success = True
            if publish_pypi or test_pypi:
                pypi_success = self.publish_to_pypi(test_pypi)
            
            print(f"✅ Release {new_version} completed successfully!")
            
            return {
                "success": True,
                "version": new_version,
                "bump_type": bump_type,
                "changelog": changelog,
                "github_url": github_url,
                "pypi_published": pypi_success,
                "updated_files": updated_files
            }
            
        except Exception as e:
            print(f"❌ Release failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "version": new_version
            }
    
    def rollback_release(self, version: str) -> bool:
        """Rollback a release by deleting tag and resetting commit."""
        try:
            print(f"Rolling back release {version}...")
            
            # Delete local tag
            subprocess.run(
                ["git", "tag", "-d", f"v{version}"],
                cwd=self.repo_path, check=True
            )
            
            # Delete remote tag
            subprocess.run(
                ["git", "push", "origin", "--delete", f"v{version}"],
                cwd=self.repo_path, check=True
            )
            
            # Reset to previous commit
            subprocess.run(
                ["git", "reset", "--hard", "HEAD~1"],
                cwd=self.repo_path, check=True
            )
            
            # Force push
            subprocess.run(
                ["git", "push", "origin", "main", "--force"],
                cwd=self.repo_path, check=True
            )
            
            print(f"✅ Rollback of release {version} completed")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Rollback failed: {e}")
            return False


def main():
    """Main function for release automation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated release management")
    parser.add_argument("--bump", choices=["major", "minor", "patch", "auto"],
                       default="auto", help="Version bump type")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without making changes")
    parser.add_argument("--publish-pypi", action="store_true",
                       help="Publish to PyPI")
    parser.add_argument("--test-pypi", action="store_true",
                       help="Publish to Test PyPI")
    parser.add_argument("--rollback", help="Rollback a specific version")
    parser.add_argument("--repo-path", default=".",
                       help="Path to repository")
    
    args = parser.parse_args()
    
    # Initialize release automation
    release_manager = ReleaseAutomation(args.repo_path)
    
    if args.rollback:
        # Rollback release
        success = release_manager.rollback_release(args.rollback)
        sys.exit(0 if success else 1)
    else:
        # Create release
        result = release_manager.create_release(
            bump_type=args.bump,
            dry_run=args.dry_run,
            publish_pypi=args.publish_pypi,
            test_pypi=args.test_pypi
        )
        
        if result["success"]:
            print(f"\n🎉 Release {result['version']} successful!")
            if result.get("github_url"):
                print(f"GitHub Release: {result['github_url']}")
        else:
            print(f"\n💥 Release failed: {result['error']}")
            sys.exit(1)


if __name__ == "__main__":
    main()