#!/usr/bin/env python3
"""
Security check for secrets in the repository
"""

import os
import re
import subprocess
from pathlib import Path

def check_git_files():
    """Check all files tracked by git for potential secrets."""
    
    print("🔒 Security Check for Open Source Repository")
    print("=" * 50)
    
    # Get all files tracked by git
    try:
        result = subprocess.run(['git', 'ls-files'], 
                              capture_output=True, text=True, check=True)
        git_files = result.stdout.strip().split('\n')
    except subprocess.CalledProcessError:
        print("❌ Not a git repository")
        return False
    
    # Patterns that might indicate secrets
    secret_patterns = [
        (r'[a-zA-Z0-9]{64}', 'API Key (64 chars)'),
        (r'[a-zA-Z0-9]{32}', 'Secret (32 chars)'),
        (r'sk-[a-zA-Z0-9]{48}', 'OpenAI API Key'),
        (r'ghp_[a-zA-Z0-9]{36}', 'GitHub Token'),
        (r'AKIA[0-9A-Z]{16}', 'AWS Access Key'),
        (r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', 'Email Address'),
        (r'password\s*[:=]\s*["\']?[^"\'\\s]+', 'Password'),
        (r'secret\s*[:=]\s*["\']?[^"\'\\s]+', 'Secret'),
        (r'api[_-]?key\s*[:=]\s*["\']?[^"\'\\s]+', 'API Key'),
        (r'token\s*[:=]\s*["\']?[^"\'\\s]+', 'Token'),
    ]
    
    issues_found = []
    
    # Check each file
    for file_path in git_files:
        if not os.path.exists(file_path):
            continue
            
        # Skip binary files and certain extensions
        if file_path.endswith(('.png', '.jpg', '.jpeg', '.gif', '.ico', '.woff', '.woff2')):
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Check for patterns
            for pattern, description in secret_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    # Skip obvious examples and placeholders
                    matched_text = match.group()
                    if any(placeholder in matched_text.lower() for placeholder in [
                        'your_', 'example', 'placeholder', 'xxx', 'yyy', 'zzz', 
                        'redacted', 'masked', 'hidden', '_here', 'fake', 'test'
                    ]):
                        continue
                    
                    # Skip documentation and comments
                    line_num = content[:match.start()].count('\n') + 1
                    line = content.split('\n')[line_num - 1]
                    if line.strip().startswith(('#', '//', '/*', '*', '--')):
                        continue
                    
                    issues_found.append({
                        'file': file_path,
                        'line': line_num,
                        'pattern': description,
                        'match': matched_text[:20] + '...' if len(matched_text) > 20 else matched_text
                    })
                    
        except Exception as e:
            print(f"⚠️  Could not read {file_path}: {e}")
    
    # Report findings
    if issues_found:
        print("🚨 POTENTIAL SECRETS FOUND:")
        for issue in issues_found:
            print(f"   {issue['file']}:{issue['line']} - {issue['pattern']} - {issue['match']}")
        print(f"\n❌ Found {len(issues_found)} potential security issues!")
        return False
    else:
        print("✅ No secrets found in git-tracked files")
        return True

def check_gitignore():
    """Check if .gitignore properly excludes sensitive files."""
    
    print("\n🛡️  Checking .gitignore...")
    
    required_ignores = ['.env', '*.log', '__pycache__', '.vscode', '.idea']
    gitignore_path = '.gitignore'
    
    if not os.path.exists(gitignore_path):
        print("❌ No .gitignore file found!")
        return False
    
    with open(gitignore_path, 'r') as f:
        gitignore_content = f.read()
    
    missing = []
    for item in required_ignores:
        if item not in gitignore_content:
            missing.append(item)
    
    if missing:
        print(f"⚠️  Missing in .gitignore: {missing}")
        return False
    else:
        print("✅ .gitignore properly configured")
        return True

def check_env_files():
    """Check if .env files are properly excluded."""
    
    print("\n🔐 Checking environment files...")
    
    env_files = ['.env', '.env.local', '.env.production']
    issues = []
    
    for env_file in env_files:
        if os.path.exists(env_file):
            # Check if tracked by git
            try:
                result = subprocess.run(['git', 'ls-files', env_file], 
                                      capture_output=True, text=True)
                if result.stdout.strip():
                    issues.append(f"{env_file} is tracked by git!")
            except:
                pass
    
    if issues:
        print("🚨 CRITICAL ISSUES:")
        for issue in issues:
            print(f"   {issue}")
        return False
    else:
        print("✅ Environment files properly excluded")
        return True

def main():
    """Run all security checks."""
    
    checks = [
        check_git_files(),
        check_gitignore(), 
        check_env_files()
    ]
    
    print(f"\n📊 Security Check Results:")
    print(f"   Passed: {sum(checks)}/{len(checks)} checks")
    
    if all(checks):
        print("🎉 SECURITY CHECK PASSED - Safe for open source!")
        return True
    else:
        print("❌ SECURITY ISSUES FOUND - Fix before publishing!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
