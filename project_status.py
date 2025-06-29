#!/usr/bin/env python3
"""
Comprehensive project status and completion checker for Telebot AI Trading Bot
This script provides a complete overview of project status and completion
"""

import os
import sys
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

class ProjectStatusChecker:
    """Comprehensive project status checker"""
    
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "project_structure": {},
            "code_completeness": {},
            "configuration": {},
            "dependencies": {},
            "documentation": {},
            "deployment": {},
            "testing": {},
            "overall_score": 0
        }
    
    def check_project_structure(self):
        """Check project directory structure"""
        print("📁 Checking project structure...")
        
        required_dirs = [
            "ai", "bot", "config", "core", "data", "tests", "web",
            "data/logs", "data/backups", "data/models", "data/cache"
        ]
        
        required_files = [
            "main.py", "requirements.txt", "requirements-dev.txt",
            "README.md", "DEVELOPMENT.md", "CHANGELOG.md",
            "docker-compose.yml", "Dockerfile", ".gitignore",
            "run.sh", "install.sh", "complete_setup.sh",
            "health_check.py", "migrate_db.py", "validate_config.py"
        ]
        
        structure_score = 0
        total_items = len(required_dirs) + len(required_files)
        
        # Check directories
        missing_dirs = []
        for dir_path in required_dirs:
            full_path = self.project_dir / dir_path
            if full_path.exists() and full_path.is_dir():
                structure_score += 1
            else:
                missing_dirs.append(dir_path)
        
        # Check files
        missing_files = []
        for file_path in required_files:
            full_path = self.project_dir / file_path
            if full_path.exists() and full_path.is_file():
                structure_score += 1
            else:
                missing_files.append(file_path)
        
        self.results["project_structure"] = {
            "score": round((structure_score / total_items) * 100, 2),
            "missing_directories": missing_dirs,
            "missing_files": missing_files,
            "total_items": total_items,
            "found_items": structure_score
        }
        
        print(f"   Structure Score: {self.results['project_structure']['score']}%")
        if missing_dirs:
            print(f"   Missing directories: {', '.join(missing_dirs)}")
        if missing_files:
            print(f"   Missing files: {', '.join(missing_files)}")
    
    def check_code_completeness(self):
        """Check code file completeness"""
        print("💻 Checking code completeness...")
        
        core_modules = {
            "main.py": ["main", "asyncio", "logging"],
            "config/settings.py": ["Settings", "validate_settings"],
            "core/database.py": ["User", "Trade", "Portfolio", "AISignal"],
            "core/indodax_api.py": ["IndodaxAPI", "get_ticker", "place_order"],
            "core/scheduler.py": ["init_scheduler", "schedule"],
            "core/risk_manager.py": ["RiskManager", "calculate_position_size"],
            "core/portfolio_manager.py": ["PortfolioManager", "calculate_pnl"],
            "ai/signal_generator.py": ["SignalGenerator", "generate_signal"],
            "ai/sentiment_analyzer.py": ["SentimentAnalyzer", "analyze_sentiment"],
            "bot/telegram_bot.py": ["TelegramBot", "start", "handle_message"],
            "bot/keyboards.py": ["create_main_keyboard", "create_trading_keyboard"],
            "bot/messages.py": ["Messages", "get_message"],
            "bot/utils.py": ["format_currency", "encrypt_api_key"],
            "web/dashboard.py": ["FastAPI", "health_check", "get_stats"]
        }
        
        completeness_score = 0
        total_modules = len(core_modules)
        implementation_details = {}
        
        for module_path, required_items in core_modules.items():
            full_path = self.project_dir / module_path
            
            if full_path.exists():
                try:
                    content = full_path.read_text()
                    found_items = []
                    missing_items = []
                    
                    for item in required_items:
                        if item in content:
                            found_items.append(item)
                        else:
                            missing_items.append(item)
                    
                    module_score = len(found_items) / len(required_items)
                    completeness_score += module_score
                    
                    implementation_details[module_path] = {
                        "score": round(module_score * 100, 2),
                        "found": found_items,
                        "missing": missing_items,
                        "file_size": len(content),
                        "line_count": len(content.splitlines())
                    }
                    
                except Exception as e:
                    implementation_details[module_path] = {
                        "score": 0,
                        "error": str(e)
                    }
            else:
                implementation_details[module_path] = {
                    "score": 0,
                    "error": "File not found"
                }
        
        self.results["code_completeness"] = {
            "score": round((completeness_score / total_modules) * 100, 2),
            "modules": implementation_details,
            "total_modules": total_modules
        }
        
        print(f"   Code Completeness Score: {self.results['code_completeness']['score']}%")
    
    def check_configuration(self):
        """Check configuration completeness"""
        print("⚙️  Checking configuration...")
        
        config_score = 0
        total_checks = 4
        
        # Check .env.example exists
        env_example = self.project_dir / "config" / ".env.example"
        if env_example.exists():
            config_score += 1
        
        # Check settings.py exists and has key classes
        settings_file = self.project_dir / "config" / "settings.py"
        if settings_file.exists():
            content = settings_file.read_text()
            if "class Settings" in content and "validate_settings" in content:
                config_score += 1
        
        # Check if validation script exists
        validation_script = self.project_dir / "validate_config.py"
        if validation_script.exists():
            config_score += 1
        
        # Check migration script exists
        migration_script = self.project_dir / "migrate_db.py"
        if migration_script.exists():
            config_score += 1
        
        self.results["configuration"] = {
            "score": round((config_score / total_checks) * 100, 2),
            "env_example_exists": env_example.exists(),
            "settings_complete": settings_file.exists(),
            "validation_script": validation_script.exists(),
            "migration_script": migration_script.exists()
        }
        
        print(f"   Configuration Score: {self.results['configuration']['score']}%")
    
    def check_dependencies(self):
        """Check dependency management"""
        print("📦 Checking dependencies...")
        
        dep_score = 0
        total_checks = 3
        
        # Check requirements.txt
        req_file = self.project_dir / "requirements.txt"
        if req_file.exists():
            content = req_file.read_text()
            if len(content.splitlines()) > 10:  # Should have many dependencies
                dep_score += 1
        
        # Check requirements-dev.txt
        req_dev_file = self.project_dir / "requirements-dev.txt"
        if req_dev_file.exists():
            dep_score += 1
        
        # Check if critical dependencies are listed
        if req_file.exists():
            content = req_file.read_text()
            critical_deps = ["aiogram", "fastapi", "sqlalchemy", "torch", "pandas"]
            if all(dep in content for dep in critical_deps):
                dep_score += 1
        
        self.results["dependencies"] = {
            "score": round((dep_score / total_checks) * 100, 2),
            "requirements_exists": req_file.exists(),
            "requirements_dev_exists": req_dev_file.exists(),
            "critical_deps_present": dep_score == total_checks
        }
        
        print(f"   Dependencies Score: {self.results['dependencies']['score']}%")
    
    def check_documentation(self):
        """Check documentation completeness"""
        print("📚 Checking documentation...")
        
        doc_score = 0
        total_checks = 4
        
        docs_to_check = [
            ("README.md", 1000),  # Should be substantial
            ("DEVELOPMENT.md", 500),
            ("CHANGELOG.md", 100),
            ("indodax_trading_bot_blueprint.md", 500)
        ]
        
        doc_details = {}
        
        for doc_file, min_size in docs_to_check:
            file_path = self.project_dir / doc_file
            if file_path.exists():
                content = file_path.read_text()
                if len(content) >= min_size:
                    doc_score += 1
                doc_details[doc_file] = {
                    "exists": True,
                    "size": len(content),
                    "adequate": len(content) >= min_size
                }
            else:
                doc_details[doc_file] = {
                    "exists": False,
                    "size": 0,
                    "adequate": False
                }
        
        self.results["documentation"] = {
            "score": round((doc_score / total_checks) * 100, 2),
            "documents": doc_details
        }
        
        print(f"   Documentation Score: {self.results['documentation']['score']}%")
    
    def check_deployment(self):
        """Check deployment readiness"""
        print("🚀 Checking deployment readiness...")
        
        deploy_score = 0
        total_checks = 6
        
        deployment_files = [
            "Dockerfile",
            "docker-compose.yml",
            "trading-bot.service",
            "run.sh",
            "install.sh",
            "complete_setup.sh"
        ]
        
        deploy_details = {}
        
        for file_name in deployment_files:
            file_path = self.project_dir / file_name
            if file_path.exists():
                deploy_score += 1
                deploy_details[file_name] = {
                    "exists": True,
                    "executable": os.access(file_path, os.X_OK) if file_name.endswith('.sh') else None
                }
            else:
                deploy_details[file_name] = {"exists": False}
        
        self.results["deployment"] = {
            "score": round((deploy_score / total_checks) * 100, 2),
            "files": deploy_details
        }
        
        print(f"   Deployment Score: {self.results['deployment']['score']}%")
    
    def check_testing(self):
        """Check testing setup"""
        print("🧪 Checking testing setup...")
        
        test_score = 0
        total_checks = 4
        
        # Check test directory exists
        test_dir = self.project_dir / "tests"
        if test_dir.exists():
            test_score += 1
        
        # Check conftest.py exists
        conftest = test_dir / "conftest.py"
        if conftest.exists():
            test_score += 1
        
        # Check test files exist
        test_files = list(test_dir.glob("test_*.py"))
        if test_files:
            test_score += 1
        
        # Check utility scripts exist
        utility_scripts = ["health_check.py", "migrate_db.py", "validate_config.py"]
        if all((self.project_dir / script).exists() for script in utility_scripts):
            test_score += 1
        
        self.results["testing"] = {
            "score": round((test_score / total_checks) * 100, 2),
            "test_directory": test_dir.exists(),
            "conftest_exists": conftest.exists(),
            "test_files_count": len(test_files),
            "utility_scripts": test_score == total_checks
        }
        
        print(f"   Testing Score: {self.results['testing']['score']}%")
    
    def calculate_overall_score(self):
        """Calculate overall project completion score"""
        categories = [
            "project_structure", "code_completeness", "configuration",
            "dependencies", "documentation", "deployment", "testing"
        ]
        
        total_score = sum(self.results[category]["score"] for category in categories)
        overall_score = total_score / len(categories)
        
        self.results["overall_score"] = round(overall_score, 2)
        
        return overall_score
    
    def get_completion_status(self, score):
        """Get completion status based on score"""
        if score >= 90:
            return "🎉 EXCELLENT", "Project is production-ready!"
        elif score >= 80:
            return "✅ VERY GOOD", "Project is nearly complete with minor items to address"
        elif score >= 70:
            return "👍 GOOD", "Project is well-developed with some areas for improvement"
        elif score >= 60:
            return "⚠️  FAIR", "Project has good foundation but needs more work"
        elif score >= 50:
            return "🔧 NEEDS WORK", "Project is functional but requires significant improvements"
        else:
            return "❌ INCOMPLETE", "Project needs substantial development"
    
    def generate_recommendations(self):
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Structure recommendations
        if self.results["project_structure"]["score"] < 100:
            missing_dirs = self.results["project_structure"]["missing_directories"]
            missing_files = self.results["project_structure"]["missing_files"]
            if missing_dirs:
                recommendations.append(f"Create missing directories: {', '.join(missing_dirs)}")
            if missing_files:
                recommendations.append(f"Create missing files: {', '.join(missing_files)}")
        
        # Code recommendations
        if self.results["code_completeness"]["score"] < 90:
            recommendations.append("Complete implementation of core modules")
            recommendations.append("Add error handling and logging to all modules")
            recommendations.append("Implement comprehensive unit tests")
        
        # Configuration recommendations
        if self.results["configuration"]["score"] < 100:
            recommendations.append("Set up proper environment configuration")
            recommendations.append("Create comprehensive validation scripts")
        
        # Documentation recommendations
        if self.results["documentation"]["score"] < 90:
            recommendations.append("Improve documentation completeness")
            recommendations.append("Add API documentation")
            recommendations.append("Create user guides and tutorials")
        
        # Deployment recommendations
        if self.results["deployment"]["score"] < 100:
            recommendations.append("Complete deployment configuration")
            recommendations.append("Test Docker and systemd deployment")
            recommendations.append("Set up monitoring and alerting")
        
        return recommendations
    
    def print_detailed_report(self):
        """Print detailed completion report"""
        print("\n" + "="*80)
        print("🤖 TELEBOT AI TRADING BOT - PROJECT COMPLETION REPORT")
        print("="*80)
        
        overall_score = self.calculate_overall_score()
        status_emoji, status_message = self.get_completion_status(overall_score)
        
        print(f"\n📊 OVERALL SCORE: {overall_score}% - {status_emoji}")
        print(f"    {status_message}")
        
        print(f"\n📅 Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n📋 CATEGORY BREAKDOWN:")
        print("-" * 40)
        
        categories = [
            ("Project Structure", "project_structure"),
            ("Code Completeness", "code_completeness"),
            ("Configuration", "configuration"),
            ("Dependencies", "dependencies"),
            ("Documentation", "documentation"),
            ("Deployment", "deployment"),
            ("Testing", "testing")
        ]
        
        for name, key in categories:
            score = self.results[key]["score"]
            status = "✅" if score >= 90 else "⚠️ " if score >= 70 else "❌"
            print(f"  {status} {name:<20}: {score:>6.1f}%")
        
        # Recommendations
        recommendations = self.generate_recommendations()
        if recommendations:
            print("\n💡 RECOMMENDATIONS:")
            print("-" * 40)
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        # Next steps
        print("\n🚀 NEXT STEPS:")
        print("-" * 40)
        if overall_score >= 90:
            print("  1. Final testing and validation")
            print("  2. Production deployment")
            print("  3. User onboarding and documentation")
            print("  4. Monitoring and maintenance setup")
        elif overall_score >= 80:
            print("  1. Address remaining implementation gaps")
            print("  2. Complete comprehensive testing")
            print("  3. Finalize documentation")
            print("  4. Prepare for production deployment")
        else:
            print("  1. Complete core module implementation")
            print("  2. Set up proper configuration management")
            print("  3. Implement comprehensive testing")
            print("  4. Improve documentation and deployment setup")
        
        print("\n🛠️  USEFUL COMMANDS:")
        print("-" * 40)
        print("  ./complete_setup.sh          - Complete project setup")
        print("  python3 health_check.py      - Run health checks")
        print("  python3 validate_config.py   - Validate configuration")
        print("  python3 migrate_db.py init   - Initialize database")
        print("  ./manage.sh start            - Start the bot")
        print("  ./manage.sh status           - Check bot status")
        
        # Save detailed report
        report_file = self.project_dir / "data" / "project_status_report.json"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: {report_file}")
        print("="*80)
    
    def run_complete_analysis(self):
        """Run complete project analysis"""
        print("🔍 Starting comprehensive project analysis...")
        print()
        
        self.check_project_structure()
        self.check_code_completeness()
        self.check_configuration()
        self.check_dependencies()
        self.check_documentation()
        self.check_deployment()
        self.check_testing()
        
        self.print_detailed_report()
        
        return self.results["overall_score"]

def main():
    """Main function"""
    checker = ProjectStatusChecker()
    score = checker.run_complete_analysis()
    
    # Exit with appropriate code
    if score >= 80:
        sys.exit(0)  # Success
    elif score >= 60:
        sys.exit(1)  # Warning
    else:
        sys.exit(2)  # Error

if __name__ == "__main__":
    main()
