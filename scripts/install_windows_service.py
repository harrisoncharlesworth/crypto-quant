#!/usr/bin/env python3
"""
Windows Service Installer for Crypto Quant Bot
Installs the bot as a Windows service for 24/7 background operation
"""

import os
import sys
import winreg
import subprocess
from pathlib import Path

class WindowsServiceInstaller:
    """Installs the crypto quant bot as a Windows service."""
    
    def __init__(self):
        self.service_name = "CryptoQuantBot"
        self.service_display_name = "Crypto Quant Bot 24/7 Trading Service"
        self.service_description = "Automated crypto trading bot with daily email reports"
        self.script_path = Path(__file__).parent / "run_live_bot.py"
        self.python_path = sys.executable
        
    def check_admin_privileges(self):
        """Check if running with administrator privileges."""
        try:
            return subprocess.check_output(['net', 'session'], stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            return False
            
    def install_service(self):
        """Install the Windows service."""
        print("🔧 Installing Crypto Quant Bot as Windows Service...")
        
        if not self.check_admin_privileges():
            print("❌ Administrator privileges required!")
            print("   Please run this script as Administrator")
            return False
            
        try:
            # Create service using sc command
            cmd = [
                'sc', 'create', self.service_name,
                'binPath=', f'"{self.python_path}" "{self.script_path}"',
                'DisplayName=', self.service_display_name,
                'start=', 'auto'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Service created successfully!")
                
                # Set service description
                desc_cmd = [
                    'sc', 'description', self.service_name,
                    self.service_description
                ]
                subprocess.run(desc_cmd, capture_output=True)
                
                # Start the service
                start_cmd = ['sc', 'start', self.service_name]
                start_result = subprocess.run(start_cmd, capture_output=True, text=True)
                
                if start_result.returncode == 0:
                    print("✅ Service started successfully!")
                    print(f"📧 Daily reports will be sent to: ebullemor@gmail.com")
                    print("🔄 Service will auto-start on boot")
                    return True
                else:
                    print(f"⚠️ Service created but failed to start: {start_result.stderr}")
                    return False
            else:
                print(f"❌ Failed to create service: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Service installation failed: {e}")
            return False
            
    def uninstall_service(self):
        """Uninstall the Windows service."""
        print("🗑️ Uninstalling Crypto Quant Bot Service...")
        
        if not self.check_admin_privileges():
            print("❌ Administrator privileges required!")
            return False
            
        try:
            # Stop the service first
            stop_cmd = ['sc', 'stop', self.service_name]
            subprocess.run(stop_cmd, capture_output=True)
            
            # Delete the service
            delete_cmd = ['sc', 'delete', self.service_name]
            result = subprocess.run(delete_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Service uninstalled successfully!")
                return True
            else:
                print(f"❌ Failed to uninstall service: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Service uninstallation failed: {e}")
            return False
            
    def check_service_status(self):
        """Check the status of the service."""
        try:
            cmd = ['sc', 'query', self.service_name]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("📊 Service Status:")
                print(result.stdout)
                return True
            else:
                print("❌ Service not found or error checking status")
                return False
                
        except Exception as e:
            print(f"❌ Failed to check service status: {e}")
            return False
            
    def start_service(self):
        """Start the service."""
        try:
            cmd = ['sc', 'start', self.service_name]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Service started successfully!")
                return True
            else:
                print(f"❌ Failed to start service: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start service: {e}")
            return False
            
    def stop_service(self):
        """Stop the service."""
        try:
            cmd = ['sc', 'stop', self.service_name]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Service stopped successfully!")
                return True
            else:
                print(f"❌ Failed to stop service: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to stop service: {e}")
            return False

def main():
    """Main installation function."""
    installer = WindowsServiceInstaller()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "install":
            installer.install_service()
        elif command == "uninstall":
            installer.uninstall_service()
        elif command == "status":
            installer.check_service_status()
        elif command == "start":
            installer.start_service()
        elif command == "stop":
            installer.stop_service()
        else:
            print("❌ Unknown command. Use: install, uninstall, status, start, or stop")
    else:
        print("🚀 Crypto Quant Bot Windows Service Installer")
        print("=" * 50)
        print("Commands:")
        print("  install   - Install and start the service")
        print("  uninstall - Remove the service")
        print("  status    - Check service status")
        print("  start     - Start the service")
        print("  stop      - Stop the service")
        print()
        print("Example: python install_windows_service.py install")

if __name__ == "__main__":
    main()
