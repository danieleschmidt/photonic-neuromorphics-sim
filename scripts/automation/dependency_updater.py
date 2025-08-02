#!/usr/bin/env python3
"""
Automated dependency update system for photonic neuromorphics simulation platform.
Handles Python packages, Docker images, and GitHub Actions updates.
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any

import requests
import yaml


class DependencyUpdater:
    """Automated dependency update system."""
    
    def __init__(self, repo_path: str = "."):
        """Initialize dependency updater."""
        self.repo_path = Path(repo_path)
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo_name = os.getenv("GITHUB_REPOSITORY", "danieleschmidt/photonic-neuromorphics-sim")
        self.updates_found = []
        
    def check_all_dependencies(self) -> Dict[str, List[Dict]]:
        """Check all types of dependencies for updates."""
        print("Checking all dependencies for updates...")
        
        updates = {
            "python": self.check_python_dependencies(),
            "docker": self.check_docker_dependencies(),
            "github_actions": self.check_github_actions_dependencies()
        }
        
        return updates
    
    def check_python_dependencies(self) -> List[Dict]:
        """Check Python dependencies for updates."""
        print("Checking Python dependencies...")
        
        updates = []
        
        # Check for outdated packages
        try:
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True, text=True
            )
            
            if result.returncode == 0 and result.stdout:
                outdated = json.loads(result.stdout)
                
                for package in outdated:
                    update_info = {
                        "type": "python",
                        "name": package["name"],
                        "current_version": package["version"],
                        "latest_version": package["latest_version"],
                        "update_type": self._classify_update_type(
                            package["version"], 
                            package["latest_version"]
                        )
                    }
                    updates.append(update_info)
                    
        except Exception as e:
            print(f"Error checking Python dependencies: {e}")
        
        # Check for security vulnerabilities
        try:
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True, text=True
            )
            
            if result.stdout and result.stdout.strip():
                vulnerabilities = json.loads(result.stdout)
                
                for vuln in vulnerabilities:
                    update_info = {
                        "type": "python_security",
                        "name": vuln["package_name"],
                        "current_version": vuln["installed_version"],
                        "vulnerable_spec": vuln["vulnerable_spec"],
                        "vulnerability_id": vuln["vulnerability_id"],
                        "severity": "high",
                        "update_type": "security"
                    }
                    updates.append(update_info)
                    
        except Exception as e:
            print(f"Error checking Python security vulnerabilities: {e}")
        
        return updates
    
    def check_docker_dependencies(self) -> List[Dict]:
        """Check Docker dependencies for updates."""
        print("Checking Docker dependencies...")
        
        updates = []
        dockerfile_path = self.repo_path / "Dockerfile"
        
        if not dockerfile_path.exists():
            return updates
        
        try:
            with open(dockerfile_path, 'r') as f:
                dockerfile_content = f.read()
            
            # Extract FROM statements
            from_pattern = r'^FROM\s+([^\s]+)(?:\s+as\s+\w+)?'
            from_matches = re.findall(from_pattern, dockerfile_content, re.MULTILINE | re.IGNORECASE)
            
            for image in from_matches:
                if ':' in image:
                    image_name, tag = image.split(':', 1)
                else:
                    image_name, tag = image, 'latest'
                
                # Check for newer versions
                latest_tag = self._get_latest_docker_tag(image_name)
                
                if latest_tag and latest_tag != tag:
                    update_info = {
                        "type": "docker",
                        "name": image_name,
                        "current_version": tag,
                        "latest_version": latest_tag,
                        "update_type": self._classify_update_type(tag, latest_tag)
                    }
                    updates.append(update_info)
                    
        except Exception as e:
            print(f"Error checking Docker dependencies: {e}")
        
        return updates
    
    def check_github_actions_dependencies(self) -> List[Dict]:
        """Check GitHub Actions dependencies for updates."""
        print("Checking GitHub Actions dependencies...")
        
        updates = []
        workflows_dir = self.repo_path / ".github" / "workflows"
        
        if not workflows_dir.exists():
            return updates
        
        try:
            for workflow_file in workflows_dir.glob("*.yml"):
                with open(workflow_file, 'r') as f:
                    workflow_content = yaml.safe_load(f)
                
                updates.extend(self._check_workflow_actions(workflow_file.name, workflow_content))
                
        except Exception as e:
            print(f"Error checking GitHub Actions dependencies: {e}")
        
        return updates
    
    def _check_workflow_actions(self, workflow_name: str, workflow: Dict) -> List[Dict]:
        """Check actions in a workflow for updates."""
        updates = []
        
        def check_job_actions(job_content):
            if not isinstance(job_content, dict) or 'steps' not in job_content:
                return
            
            for step in job_content['steps']:
                if 'uses' in step:
                    action = step['uses']
                    
                    # Parse action reference
                    if '@' in action:
                        action_name, version = action.rsplit('@', 1)
                        latest_version = self._get_latest_action_version(action_name)
                        
                        if latest_version and latest_version != version:
                            update_info = {
                                "type": "github_action",
                                "name": action_name,
                                "current_version": version,
                                "latest_version": latest_version,
                                "workflow": workflow_name,
                                "update_type": self._classify_update_type(version, latest_version)
                            }
                            updates.append(update_info)
        
        # Check jobs
        if 'jobs' in workflow:
            for job_name, job_content in workflow['jobs'].items():
                if isinstance(job_content, dict):
                    check_job_actions(job_content)
                    
                    # Check reusable workflows
                    if 'uses' in job_content:
                        workflow_ref = job_content['uses']
                        if '@' in workflow_ref:
                            workflow_name_ref, version = workflow_ref.rsplit('@', 1)
                            # Note: Checking reusable workflow versions would require additional API calls
        
        return updates
    
    def _get_latest_docker_tag(self, image_name: str) -> str:
        """Get the latest tag for a Docker image."""
        try:
            # For Docker Hub images
            if '/' not in image_name or not image_name.startswith(('ghcr.io', 'gcr.io', 'quay.io')):
                # Docker Hub API
                if '/' not in image_name:
                    image_name = f"library/{image_name}"
                
                url = f"https://registry-1.docker.io/v2/{image_name}/tags/list"
                response = requests.get(url)
                
                if response.status_code == 200:
                    tags = response.json().get('tags', [])
                    # Filter and sort semantic version tags
                    version_tags = [tag for tag in tags if re.match(r'^\d+\.\d+', tag)]
                    if version_tags:
                        return sorted(version_tags, key=self._version_key, reverse=True)[0]
                        
        except Exception as e:
            print(f"Error getting latest Docker tag for {image_name}: {e}")
        
        return None
    
    def _get_latest_action_version(self, action_name: str) -> str:
        """Get the latest version for a GitHub Action."""
        try:
            if not self.github_token:
                return None
            
            headers = {"Authorization": f"token {self.github_token}"}
            url = f"https://api.github.com/repos/{action_name}/releases/latest"
            
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                latest_release = response.json()
                return latest_release.get('tag_name', '').lstrip('v')
                
        except Exception as e:
            print(f"Error getting latest version for action {action_name}: {e}")
        
        return None
    
    def _classify_update_type(self, current: str, latest: str) -> str:
        """Classify the type of update (major, minor, patch, security)."""
        try:
            current_parts = [int(x) for x in current.split('.') if x.isdigit()]
            latest_parts = [int(x) for x in latest.split('.') if x.isdigit()]
            
            # Pad with zeros to make comparison easier
            max_len = max(len(current_parts), len(latest_parts))
            current_parts.extend([0] * (max_len - len(current_parts)))
            latest_parts.extend([0] * (max_len - len(latest_parts)))
            
            if len(current_parts) >= 1 and len(latest_parts) >= 1:
                if latest_parts[0] > current_parts[0]:
                    return "major"
                elif len(current_parts) >= 2 and len(latest_parts) >= 2 and latest_parts[1] > current_parts[1]:
                    return "minor"
                elif len(current_parts) >= 3 and len(latest_parts) >= 3 and latest_parts[2] > current_parts[2]:
                    return "patch"
            
        except (ValueError, IndexError):
            pass
        
        return "unknown"
    
    def _version_key(self, version: str) -> Tuple[int, ...]:
        """Create a sortable key from a version string."""
        try:
            return tuple(int(x) for x in version.split('.') if x.isdigit())
        except ValueError:
            return (0,)
    
    def create_update_branch(self, updates: List[Dict], update_type: str = "all") -> bool:
        """Create a branch with dependency updates."""
        if not updates:
            print("No updates to apply")
            return False
        
        # Filter updates based on type
        filtered_updates = self._filter_updates(updates, update_type)
        
        if not filtered_updates:
            print(f"No {update_type} updates found")
            return False
        
        # Create branch
        branch_name = f"dependency-updates/{update_type}-{datetime.now().strftime('%Y%m%d')}"
        
        try:
            subprocess.run(["git", "checkout", "-b", branch_name], check=True)
            print(f"Created branch: {branch_name}")
            
            # Apply updates
            changes_made = False
            
            for update in filtered_updates:
                if self._apply_update(update):
                    changes_made = True
            
            if changes_made:
                # Commit changes
                subprocess.run(["git", "add", "."], check=True)
                
                commit_message = self._generate_commit_message(filtered_updates)
                subprocess.run(["git", "commit", "-m", commit_message], check=True)
                
                print(f"Committed updates to {branch_name}")
                return True
            else:
                # No changes made, delete branch
                subprocess.run(["git", "checkout", "main"], check=True)
                subprocess.run(["git", "branch", "-D", branch_name], check=True)
                print("No changes applied, branch deleted")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"Error creating update branch: {e}")
            return False
    
    def _filter_updates(self, updates: List[Dict], update_type: str) -> List[Dict]:
        """Filter updates based on type."""
        if update_type == "all":
            return updates
        elif update_type == "security":
            return [u for u in updates if u.get("update_type") == "security" or u.get("type") == "python_security"]
        elif update_type == "minor":
            return [u for u in updates if u.get("update_type") in ["minor", "patch"]]
        elif update_type == "patch":
            return [u for u in updates if u.get("update_type") == "patch"]
        else:
            return [u for u in updates if u.get("type") == update_type]
    
    def _apply_update(self, update: Dict) -> bool:
        """Apply a single dependency update."""
        try:
            if update["type"] == "python" or update["type"] == "python_security":
                return self._update_python_dependency(update)
            elif update["type"] == "docker":
                return self._update_docker_dependency(update)
            elif update["type"] == "github_action":
                return self._update_github_action(update)
        except Exception as e:
            print(f"Error applying update for {update['name']}: {e}")
        
        return False
    
    def _update_python_dependency(self, update: Dict) -> bool:
        """Update a Python dependency."""
        try:
            # Update requirements.txt
            requirements_file = self.repo_path / "requirements.txt"
            if requirements_file.exists():
                with open(requirements_file, 'r') as f:
                    lines = f.readlines()
                
                updated = False
                for i, line in enumerate(lines):
                    if line.strip().startswith(update["name"]):
                        if update["type"] == "python_security":
                            # For security updates, use the safe version
                            new_line = f"{update['name']}>={update['latest_version']}\n"
                        else:
                            new_line = f"{update['name']}=={update['latest_version']}\n"
                        lines[i] = new_line
                        updated = True
                        break
                
                if updated:
                    with open(requirements_file, 'w') as f:
                        f.writelines(lines)
                    print(f"Updated {update['name']} in requirements.txt")
                    return True
                    
        except Exception as e:
            print(f"Error updating Python dependency {update['name']}: {e}")
        
        return False
    
    def _update_docker_dependency(self, update: Dict) -> bool:
        """Update a Docker dependency."""
        try:
            dockerfile_path = self.repo_path / "Dockerfile"
            
            with open(dockerfile_path, 'r') as f:
                content = f.read()
            
            # Replace the FROM statement
            old_ref = f"{update['name']}:{update['current_version']}"
            new_ref = f"{update['name']}:{update['latest_version']}"
            
            updated_content = content.replace(old_ref, new_ref)
            
            if updated_content != content:
                with open(dockerfile_path, 'w') as f:
                    f.write(updated_content)
                print(f"Updated {update['name']} in Dockerfile")
                return True
                
        except Exception as e:
            print(f"Error updating Docker dependency {update['name']}: {e}")
        
        return False
    
    def _update_github_action(self, update: Dict) -> bool:
        """Update a GitHub Action dependency."""
        try:
            workflow_path = self.repo_path / ".github" / "workflows" / update["workflow"]
            
            with open(workflow_path, 'r') as f:
                content = f.read()
            
            # Replace the action reference
            old_ref = f"{update['name']}@{update['current_version']}"
            new_ref = f"{update['name']}@{update['latest_version']}"
            
            updated_content = content.replace(old_ref, new_ref)
            
            if updated_content != content:
                with open(workflow_path, 'w') as f:
                    f.write(updated_content)
                print(f"Updated {update['name']} in {update['workflow']}")
                return True
                
        except Exception as e:
            print(f"Error updating GitHub Action {update['name']}: {e}")
        
        return False
    
    def _generate_commit_message(self, updates: List[Dict]) -> str:
        """Generate a commit message for the updates."""
        update_types = set(u.get("update_type", "unknown") for u in updates)
        
        if "security" in update_types:
            prefix = "security"
            description = "Security vulnerability fixes"
        elif len(update_types) == 1 and "patch" in update_types:
            prefix = "deps"
            description = "Patch version updates"
        elif len(update_types) == 1 and "minor" in update_types:
            prefix = "deps"
            description = "Minor version updates"
        else:
            prefix = "deps"
            description = "Dependency updates"
        
        update_summary = []
        for update in updates[:5]:  # Limit to first 5 updates
            update_summary.append(f"- {update['name']}: {update['current_version']} → {update['latest_version']}")
        
        if len(updates) > 5:
            update_summary.append(f"- ... and {len(updates) - 5} more")
        
        commit_message = f"{prefix}: {description}\n\n"
        commit_message += "\n".join(update_summary)
        commit_message += "\n\n🤖 Automated dependency update"
        
        return commit_message
    
    def create_pull_request(self, branch_name: str, updates: List[Dict]) -> bool:
        """Create a pull request for the updates."""
        if not self.github_token:
            print("No GitHub token available, cannot create PR")
            return False
        
        try:
            headers = {
                "Authorization": f"token {self.github_token}",
                "Content-Type": "application/json"
            }
            
            pr_title = self._generate_pr_title(updates)
            pr_body = self._generate_pr_body(updates)
            
            pr_data = {
                "title": pr_title,
                "body": pr_body,
                "head": branch_name,
                "base": "main"
            }
            
            url = f"https://api.github.com/repos/{self.repo_name}/pulls"
            response = requests.post(url, headers=headers, json=pr_data)
            
            if response.status_code == 201:
                pr_url = response.json()["html_url"]
                print(f"Created pull request: {pr_url}")
                return True
            else:
                print(f"Failed to create PR: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"Error creating pull request: {e}")
            return False
    
    def _generate_pr_title(self, updates: List[Dict]) -> str:
        """Generate a title for the pull request."""
        security_updates = [u for u in updates if u.get("update_type") == "security"]
        
        if security_updates:
            return f"🔒 Security updates ({len(security_updates)} packages)"
        else:
            return f"📦 Dependency updates ({len(updates)} packages)"
    
    def _generate_pr_body(self, updates: List[Dict]) -> str:
        """Generate a body for the pull request."""
        body = "## Dependency Updates\n\n"
        body += "This PR contains automated dependency updates.\n\n"
        
        # Group updates by type
        python_updates = [u for u in updates if u["type"].startswith("python")]
        docker_updates = [u for u in updates if u["type"] == "docker"]
        action_updates = [u for u in updates if u["type"] == "github_action"]
        security_updates = [u for u in updates if u.get("update_type") == "security"]
        
        if security_updates:
            body += "### 🔒 Security Updates\n\n"
            for update in security_updates:
                body += f"- **{update['name']}**: {update.get('current_version', 'N/A')} → {update.get('latest_version', 'N/A')}\n"
            body += "\n"
        
        if python_updates:
            body += "### 🐍 Python Dependencies\n\n"
            for update in python_updates:
                if update not in security_updates:
                    body += f"- **{update['name']}**: {update['current_version']} → {update['latest_version']}\n"
            body += "\n"
        
        if docker_updates:
            body += "### 🐳 Docker Dependencies\n\n"
            for update in docker_updates:
                body += f"- **{update['name']}**: {update['current_version']} → {update['latest_version']}\n"
            body += "\n"
        
        if action_updates:
            body += "### ⚡ GitHub Actions\n\n"
            for update in action_updates:
                body += f"- **{update['name']}**: {update['current_version']} → {update['latest_version']}\n"
            body += "\n"
        
        body += "### Testing\n\n"
        body += "- [ ] All tests pass\n"
        body += "- [ ] No breaking changes detected\n"
        body += "- [ ] Security scans clean\n\n"
        
        body += "---\n"
        body += "*This PR was automatically created by the dependency update system.*"
        
        return body


def main():
    """Main function for dependency updater."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Update project dependencies")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check for updates, don't apply them")
    parser.add_argument("--type", choices=["all", "security", "minor", "patch", "python", "docker", "github_actions"],
                       default="all", help="Type of updates to process")
    parser.add_argument("--create-pr", action="store_true",
                       help="Create pull request for updates")
    parser.add_argument("--repo-path", default=".",
                       help="Path to repository")
    
    args = parser.parse_args()
    
    # Initialize updater
    updater = DependencyUpdater(args.repo_path)
    
    # Check for updates
    all_updates = updater.check_all_dependencies()
    
    # Flatten updates list
    updates = []
    for update_type, update_list in all_updates.items():
        updates.extend(update_list)
    
    if not updates:
        print("No dependency updates available")
        return
    
    print(f"Found {len(updates)} potential updates")
    
    if args.check_only:
        # Just report what was found
        for update in updates:
            print(f"- {update['name']}: {update.get('current_version', 'N/A')} → {update.get('latest_version', 'N/A')} ({update.get('update_type', 'unknown')})")
        return
    
    # Create update branch
    branch_created = updater.create_update_branch(updates, args.type)
    
    if branch_created and args.create_pr:
        # Create pull request
        branch_name = f"dependency-updates/{args.type}-{datetime.now().strftime('%Y%m%d')}"
        filtered_updates = updater._filter_updates(updates, args.type)
        updater.create_pull_request(branch_name, filtered_updates)
    
    print("Dependency update process completed!")


if __name__ == "__main__":
    main()