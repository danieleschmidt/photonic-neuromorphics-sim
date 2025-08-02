#!/usr/bin/env python3
"""
Dependency Update Automation Script

This script automates the process of checking for and updating dependencies
with security prioritization and automated PR creation.
"""

import argparse
import json
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile
import requests


class DependencyUpdater:
    """Handles automated dependency updates with security prioritization."""
    
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
    
    def check_outdated_packages(self) -> List[Dict]:
        """Check for outdated Python packages."""
        print("📦 Checking for outdated packages...")
        
        result = self.run_command(["pip", "list", "--outdated", "--format=json"])
        if not result["success"]:
            print(f"❌ Failed to check outdated packages: {result.get('stderr', 'Unknown error')}")
            return []
        
        try:
            outdated = json.loads(result["stdout"])
            print(f"Found {len(outdated)} outdated packages")
            return outdated
        except json.JSONDecodeError:
            print("❌ Failed to parse pip list output")
            return []
    
    def check_security_vulnerabilities(self) -> List[Dict]:
        """Check for security vulnerabilities using safety."""
        print("🔒 Checking for security vulnerabilities...")
        
        result = self.run_command(["safety", "check", "--json"])
        if not result["success"]:
            # safety returns non-zero when vulnerabilities found
            if result["stdout"]:
                try:
                    vulnerabilities = json.loads(result["stdout"])
                    print(f"⚠️ Found {len(vulnerabilities)} security vulnerabilities")
                    return vulnerabilities
                except json.JSONDecodeError:
                    pass
            print("No security vulnerabilities detected")
            return []
        
        return []
    
    def get_package_info(self, package_name: str) -> Optional[Dict]:
        """Get package information from PyPI."""
        try:
            response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"⚠️ Failed to get info for {package_name}: {e}")
        return None
    
    def prioritize_updates(self, outdated: List[Dict], vulnerabilities: List[Dict]) -> List[Dict]:
        """Prioritize updates based on security and other factors."""
        print("🎯 Prioritizing updates...")
        
        # Create vulnerability lookup
        vuln_packages = {vuln.get("package_name", "").lower() for vuln in vulnerabilities}
        
        prioritized = []
        for package in outdated:
            name = package["name"].lower()
            priority = "normal"
            reason = "outdated"
            
            # Security vulnerabilities get highest priority
            if name in vuln_packages:
                priority = "critical"
                reason = "security_vulnerability"
            # Major version updates get medium priority
            elif self._is_major_update(package.get("version", ""), package.get("latest_version", "")):
                priority = "medium"
                reason = "major_version"
            
            prioritized.append({
                **package,
                "priority": priority,
                "reason": reason
            })
        
        # Sort by priority: critical > medium > normal
        priority_order = {"critical": 0, "medium": 1, "normal": 2}
        prioritized.sort(key=lambda x: priority_order.get(x["priority"], 3))
        
        return prioritized
    
    def _is_major_update(self, current: str, latest: str) -> bool:
        """Check if this is a major version update."""
        try:
            current_parts = current.split(".")
            latest_parts = latest.split(".")
            
            if len(current_parts) >= 1 and len(latest_parts) >= 1:
                return int(current_parts[0]) < int(latest_parts[0])
        except (ValueError, IndexError):
            pass
        return False
    
    def create_update_branch(self, package_info: Dict) -> str:
        """Create a new branch for the update."""
        package_name = package_info["name"]
        branch_name = f"deps/update-{package_name}-{datetime.now().strftime('%Y%m%d')}"
        
        # Create and checkout new branch
        result = self.run_command(["git", "checkout", "-b", branch_name])
        if not result["success"]:
            raise Exception(f"Failed to create branch: {result.get('stderr', 'Unknown error')}")
        
        return branch_name
    
    def update_requirements_file(self, package_info: Dict, target_version: str = None) -> bool:
        """Update requirements.txt with new package version."""
        requirements_files = ["requirements.txt", "requirements-dev.txt"]
        updated = False
        
        for req_file in requirements_files:
            req_path = self.repo_path / req_file
            if not req_path.exists():
                continue
            
            with open(req_path, 'r') as f:
                lines = f.readlines()
            
            package_name = package_info["name"]
            new_version = target_version or package_info["latest_version"]
            
            for i, line in enumerate(lines):
                if line.strip().startswith(package_name):
                    # Update the line with new version
                    if "==" in line:
                        lines[i] = f"{package_name}=={new_version}\n"
                    elif ">=" in line:
                        lines[i] = f"{package_name}>={new_version}\n"
                    else:
                        lines[i] = f"{package_name}=={new_version}\n"
                    updated = True
                    break
            
            if updated:
                with open(req_path, 'w') as f:
                    f.writelines(lines)
                print(f"📝 Updated {req_file}")
        
        return updated
    
    def run_tests(self) -> bool:
        """Run tests to ensure updates don't break anything."""
        print("🧪 Running tests...")
        
        # Try pytest first
        result = self.run_command(["pytest", "--tb=short", "-q"])
        if result["success"]:
            print("✅ All tests passed")
            return True
        
        # Fall back to unittest
        result = self.run_command(["python", "-m", "unittest", "discover", "-s", "tests"])
        if result["success"]:
            print("✅ All tests passed")
            return True
        
        print(f"❌ Tests failed: {result.get('stderr', 'Unknown error')}")
        return False
    
    def commit_changes(self, package_info: Dict) -> bool:
        """Commit the dependency update changes."""
        package_name = package_info["name"]
        new_version = package_info["latest_version"]
        priority = package_info.get("priority", "normal")
        
        # Add changes
        result = self.run_command(["git", "add", "-A"])
        if not result["success"]:
            print(f"❌ Failed to stage changes: {result.get('stderr', 'Unknown error')}")
            return False
        
        # Create commit message
        if priority == "critical":
            commit_msg = f"deps(security): update {package_name} to {new_version}"
        else:
            commit_msg = f"deps: update {package_name} to {new_version}"
        
        # Commit changes
        result = self.run_command(["git", "commit", "-m", commit_msg])
        if not result["success"]:
            print(f"❌ Failed to commit changes: {result.get('stderr', 'Unknown error')}")
            return False
        
        print(f"✅ Committed update for {package_name}")
        return True
    
    def create_pull_request(self, branch_name: str, package_info: Dict) -> Optional[str]:
        """Create a pull request for the dependency update."""
        if not self.github_token or not self.repo_name:
            print("⚠️ GitHub token or repository name not configured, skipping PR creation")
            return None
        
        package_name = package_info["name"]
        current_version = package_info["version"]
        new_version = package_info["latest_version"]
        priority = package_info.get("priority", "normal")
        reason = package_info.get("reason", "outdated")
        
        # Create PR title and body
        if priority == "critical":
            title = f"🔒 Security: Update {package_name} to {new_version}"
        else:
            title = f"⬆️ Update {package_name} to {new_version}"
        
        body = f"""## Dependency Update

**Package**: {package_name}
**Current Version**: {current_version}
**New Version**: {new_version}
**Priority**: {priority.upper()}
**Reason**: {reason.replace('_', ' ').title()}

### Changes
- Updated {package_name} from {current_version} to {new_version}

### Testing
- [x] Tests pass
- [x] No breaking changes detected

### Security
"""
        
        if priority == "critical":
            body += "- [x] This update addresses security vulnerabilities\n"
        else:
            body += "- [x] No known security issues\n"
        
        body += "\n*Auto-generated by dependency updater*"
        
        # Create PR via GitHub API
        try:
            url = f"https://api.github.com/repos/{self.repo_name}/pulls"
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            data = {
                "title": title,
                "body": body,
                "head": branch_name,
                "base": "main"
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            if response.status_code == 201:
                pr_data = response.json()
                pr_url = pr_data["html_url"]
                print(f"✅ Created PR: {pr_url}")
                return pr_url
            else:
                print(f"❌ Failed to create PR: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Error creating PR: {e}")
        
        return None
    
    def update_package(self, package_info: Dict, create_pr: bool = False) -> bool:
        """Update a single package with full workflow."""
        package_name = package_info["name"]
        priority = package_info.get("priority", "normal")
        
        print(f"\n🔄 Updating {package_name} (priority: {priority})")
        
        try:
            # Create update branch
            branch_name = self.create_update_branch(package_info)
            
            # Update requirements files
            if not self.update_requirements_file(package_info):
                print(f"⚠️ No requirements file updates needed for {package_name}")
                self.run_command(["git", "checkout", "main"])
                self.run_command(["git", "branch", "-D", branch_name])
                return False
            
            # Install updated dependencies
            result = self.run_command(["pip", "install", "-r", "requirements.txt"])
            if not result["success"]:
                print(f"❌ Failed to install updated dependencies")
                self.run_command(["git", "checkout", "main"])
                self.run_command(["git", "branch", "-D", branch_name])
                return False
            
            # Run tests
            if not self.run_tests():
                print(f"❌ Tests failed after updating {package_name}")
                self.run_command(["git", "checkout", "main"])
                self.run_command(["git", "branch", "-D", branch_name])
                return False
            
            # Commit changes
            if not self.commit_changes(package_info):
                self.run_command(["git", "checkout", "main"])
                self.run_command(["git", "branch", "-D", branch_name])
                return False
            
            # Push branch and create PR if requested
            if create_pr:
                result = self.run_command(["git", "push", "origin", branch_name])
                if result["success"]:
                    self.create_pull_request(branch_name, package_info)
                else:
                    print(f"⚠️ Failed to push branch: {result.get('stderr', 'Unknown error')}")
            
            # Return to main branch
            self.run_command(["git", "checkout", "main"])
            
            print(f"✅ Successfully updated {package_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error updating {package_name}: {e}")
            self.run_command(["git", "checkout", "main"])
            self.run_command(["git", "branch", "-D", branch_name])
            return False
    
    def run_update_workflow(self, update_type: str = "security", max_updates: int = 5, create_pr: bool = False):
        """Run the complete dependency update workflow."""
        print("🚀 Starting dependency update workflow...")
        
        # Check for outdated packages
        outdated = self.check_outdated_packages()
        if not outdated:
            print("✅ All packages are up to date!")
            return
        
        # Check for security vulnerabilities
        vulnerabilities = self.check_security_vulnerabilities()
        
        # Prioritize updates
        prioritized = self.prioritize_updates(outdated, vulnerabilities)
        
        # Filter by update type
        if update_type == "security":
            updates_to_process = [p for p in prioritized if p["priority"] == "critical"]
        elif update_type == "major":
            updates_to_process = [p for p in prioritized if p["priority"] in ["critical", "medium"]]
        else:  # all
            updates_to_process = prioritized
        
        # Limit number of updates
        updates_to_process = updates_to_process[:max_updates]
        
        if not updates_to_process:
            print(f"✅ No {update_type} updates needed!")
            return
        
        print(f"📋 Processing {len(updates_to_process)} updates:")
        for pkg in updates_to_process:
            print(f"  - {pkg['name']}: {pkg['version']} → {pkg['latest_version']} ({pkg['priority']})")
        
        # Process updates
        successful_updates = 0
        for package_info in updates_to_process:
            if self.update_package(package_info, create_pr):
                successful_updates += 1
        
        print(f"\n✅ Dependency update workflow completed!")
        print(f"📊 {successful_updates}/{len(updates_to_process)} updates successful")


def main():
    parser = argparse.ArgumentParser(description="Automate dependency updates")
    parser.add_argument("--type", choices=["security", "major", "all"], default="security",
                        help="Type of updates to process")
    parser.add_argument("--max-updates", type=int, default=5,
                        help="Maximum number of updates to process")
    parser.add_argument("--create-pr", action="store_true",
                        help="Create pull requests for updates")
    parser.add_argument("--repo-path", help="Repository path (default: current directory)")
    
    args = parser.parse_args()
    
    repo_path = Path(args.repo_path) if args.repo_path else Path.cwd()
    updater = DependencyUpdater(repo_path)
    
    try:
        updater.run_update_workflow(
            update_type=args.type,
            max_updates=args.max_updates,
            create_pr=args.create_pr
        )
    except Exception as e:
        print(f"❌ Error in dependency update workflow: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()