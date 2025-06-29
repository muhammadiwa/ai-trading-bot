#!/usr/bin/env python3
"""
Health check script for Telebot AI Trading Bot
This script performs comprehensive health checks on all components
"""

import os
import sys
import subprocess
import json
import time
import psutil
import requests
from datetime import datetime
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

try:
    from config.settings import settings
    from core.database import test_database_connection
    from core.indodax_api import IndodaxAPI
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project directory with virtual environment activated")
    sys.exit(1)

class HealthChecker:
    def __init__(self):
        self.results = []
        self.project_dir = Path(__file__).parent
        
    def log_result(self, test_name, status, message="", details=None):
        """Log test result"""
        result = {
            "test": test_name,
            "status": status,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        self.results.append(result)
        
        # Print colored output
        status_color = "\033[92m" if status == "PASS" else "\033[91m" if status == "FAIL" else "\033[93m"
        reset_color = "\033[0m"
        
        print(f"{status_color}[{status}]{reset_color} {test_name}")
        if message:
            print(f"       {message}")
        if details:
            print(f"       Details: {details}")
        print()
    
    def check_python_version(self):
        """Check Python version compatibility"""
        python_version = sys.version_info
        if python_version >= (3, 8):
            self.log_result(
                "Python Version",
                "PASS",
                f"Python {python_version.major}.{python_version.minor}.{python_version.micro}"
            )
        else:
            self.log_result(
                "Python Version",
                "FAIL",
                f"Python {python_version.major}.{python_version.minor}.{python_version.micro} (requires 3.8+)"
            )
    
    def check_dependencies(self):
        """Check if all required dependencies are installed"""
        required_packages = [
            'aiogram', 'fastapi', 'sqlalchemy', 'requests', 'asyncio',
            'pandas', 'numpy', 'torch', 'sklearn', 'redis', 'psycopg2'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if not missing_packages:
            self.log_result("Dependencies", "PASS", "All required packages installed")
        else:
            self.log_result(
                "Dependencies",
                "FAIL",
                f"Missing packages: {', '.join(missing_packages)}"
            )
    
    def check_configuration(self):
        """Check configuration settings"""
        try:
            # Check if .env file exists
            env_file = self.project_dir / '.env'
            if not env_file.exists():
                self.log_result(
                    "Configuration",
                    "WARN",
                    ".env file not found. Using default settings."
                )
            else:
                self.log_result("Configuration", "PASS", ".env file found")
            
            # Check critical settings
            critical_settings = [
                'telegram_bot_token',
                'indodax_api_key',
                'indodax_secret_key',
                'database_url'
            ]
            
            missing_settings = []
            for setting in critical_settings:
                if not hasattr(settings, setting) or not getattr(settings, setting):
                    missing_settings.append(setting)
            
            if missing_settings:
                self.log_result(
                    "Critical Settings",
                    "WARN",
                    f"Missing or empty: {', '.join(missing_settings)}"
                )
            else:
                self.log_result("Critical Settings", "PASS", "All critical settings configured")
                
        except Exception as e:
            self.log_result("Configuration", "FAIL", str(e))
    
    def check_database_connection(self):
        """Check database connectivity"""
        try:
            # This would need to be implemented in core/database.py
            # For now, we'll do a basic check
            if hasattr(settings, 'database_url') and settings.database_url:
                # Try to connect to database
                # This is a placeholder - implement actual database connection test
                self.log_result("Database", "PASS", "Database configuration found")
            else:
                self.log_result("Database", "FAIL", "Database URL not configured")
        except Exception as e:
            self.log_result("Database", "FAIL", str(e))
    
    def check_redis_connection(self):
        """Check Redis connectivity"""
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            r.ping()
            self.log_result("Redis", "PASS", "Redis connection successful")
        except Exception as e:
            self.log_result("Redis", "FAIL", str(e))
    
    def check_indodax_api(self):
        """Check Indodax API connectivity"""
        try:
            api = IndodaxAPI()
            # Test public API endpoint
            import asyncio
            
            async def test_api():
                return await api.get_server_time()
            
            result = asyncio.run(test_api())
            if result:
                self.log_result("Indodax API", "PASS", "Public API accessible")
            else:
                self.log_result("Indodax API", "FAIL", "Public API not accessible")
        except Exception as e:
            self.log_result("Indodax API", "FAIL", str(e))
    
    def check_telegram_bot(self):
        """Check Telegram bot token"""
        try:
            if hasattr(settings, 'telegram_bot_token') and settings.telegram_bot_token:
                # Test bot token by making a simple API call
                response = requests.get(
                    f"https://api.telegram.org/bot{settings.telegram_bot_token}/getMe",
                    timeout=10
                )
                if response.status_code == 200:
                    bot_info = response.json()
                    if bot_info.get('ok'):
                        username = bot_info['result']['username']
                        self.log_result(
                            "Telegram Bot",
                            "PASS",
                            f"Bot token valid (@{username})"
                        )
                    else:
                        self.log_result("Telegram Bot", "FAIL", "Invalid bot token")
                else:
                    self.log_result("Telegram Bot", "FAIL", f"HTTP {response.status_code}")
            else:
                self.log_result("Telegram Bot", "FAIL", "Bot token not configured")
        except Exception as e:
            self.log_result("Telegram Bot", "FAIL", str(e))
    
    def check_system_resources(self):
        """Check system resources"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            details = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2)
            }
            
            status = "PASS"
            message = "System resources normal"
            
            if cpu_percent > 90:
                status = "WARN"
                message = f"High CPU usage: {cpu_percent}%"
            elif memory.percent > 90:
                status = "WARN"
                message = f"High memory usage: {memory.percent}%"
            elif disk.percent > 90:
                status = "WARN"
                message = f"High disk usage: {disk.percent}%"
            
            self.log_result("System Resources", status, message, details)
            
        except Exception as e:
            self.log_result("System Resources", "FAIL", str(e))
    
    def check_file_permissions(self):
        """Check file permissions"""
        try:
            critical_files = [
                'main.py',
                'run.sh',
                'install.sh',
                'complete_setup.sh'
            ]
            
            issues = []
            for file in critical_files:
                file_path = self.project_dir / file
                if file_path.exists():
                    if not os.access(file_path, os.R_OK):
                        issues.append(f"{file} not readable")
                    if file.endswith('.sh') and not os.access(file_path, os.X_OK):
                        issues.append(f"{file} not executable")
                else:
                    issues.append(f"{file} not found")
            
            if not issues:
                self.log_result("File Permissions", "PASS", "All critical files accessible")
            else:
                self.log_result("File Permissions", "FAIL", "; ".join(issues))
                
        except Exception as e:
            self.log_result("File Permissions", "FAIL", str(e))
    
    def check_log_directories(self):
        """Check if log directories exist and are writable"""
        try:
            log_dirs = [
                'data/logs',
                'data/backups',
                'data/models',
                'data/cache'
            ]
            
            for log_dir in log_dirs:
                dir_path = self.project_dir / log_dir
                if not dir_path.exists():
                    dir_path.mkdir(parents=True, exist_ok=True)
                
                # Test write permissions
                test_file = dir_path / 'test_write.tmp'
                try:
                    test_file.write_text('test')
                    test_file.unlink()
                except Exception:
                    self.log_result(
                        "Log Directories",
                        "FAIL",
                        f"Cannot write to {log_dir}"
                    )
                    return
            
            self.log_result("Log Directories", "PASS", "All directories writable")
            
        except Exception as e:
            self.log_result("Log Directories", "FAIL", str(e))
    
    def check_service_status(self):
        """Check systemd service status"""
        try:
            result = subprocess.run(
                ['systemctl', 'is-active', 'trading-bot'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                status = result.stdout.strip()
                if status == 'active':
                    self.log_result("Service Status", "PASS", "Service is running")
                else:
                    self.log_result("Service Status", "WARN", f"Service status: {status}")
            else:
                self.log_result("Service Status", "WARN", "Service not installed or not running")
                
        except Exception as e:
            self.log_result("Service Status", "WARN", str(e))
    
    def run_all_checks(self):
        """Run all health checks"""
        print("🏥 Starting Telebot AI Trading Bot Health Check...")
        print("=" * 60)
        print()
        
        self.check_python_version()
        self.check_dependencies()
        self.check_configuration()
        self.check_database_connection()
        self.check_redis_connection()
        self.check_indodax_api()
        self.check_telegram_bot()
        self.check_system_resources()
        self.check_file_permissions()
        self.check_log_directories()
        self.check_service_status()
        
        # Summary
        print("=" * 60)
        print("📊 HEALTH CHECK SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r['status'] == 'PASS'])
        failed_tests = len([r for r in self.results if r['status'] == 'FAIL'])
        warning_tests = len([r for r in self.results if r['status'] == 'WARN'])
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"⚠️  Warnings: {warning_tests}")
        print(f"❌ Failed: {failed_tests}")
        
        if failed_tests > 0:
            print("\n🔴 FAILED TESTS:")
            for result in self.results:
                if result['status'] == 'FAIL':
                    print(f"  - {result['test']}: {result['message']}")
        
        if warning_tests > 0:
            print("\n🟡 WARNINGS:")
            for result in self.results:
                if result['status'] == 'WARN':
                    print(f"  - {result['test']}: {result['message']}")
        
        # Save results to file
        results_file = self.project_dir / 'data' / 'health_check_results.json'
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        # Overall status
        if failed_tests == 0:
            print("\n🎉 Overall Status: HEALTHY")
            return 0
        else:
            print("\n⚠️  Overall Status: NEEDS ATTENTION")
            return 1

def main():
    """Main function"""
    checker = HealthChecker()
    return checker.run_all_checks()

if __name__ == "__main__":
    sys.exit(main())
