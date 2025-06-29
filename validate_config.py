#!/usr/bin/env python3
"""
Configuration validation and setup assistant for Telebot AI Trading Bot
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

class ConfigValidator:
    """Configuration validation and setup assistant"""
    
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.env_file = self.project_dir / '.env'
        self.env_example_file = self.project_dir / 'config' / '.env.example'
        self.errors = []
        self.warnings = []
        self.suggestions = []
    
    def log_error(self, message: str):
        """Log an error"""
        self.errors.append(message)
        print(f"❌ ERROR: {message}")
    
    def log_warning(self, message: str):
        """Log a warning"""
        self.warnings.append(message)
        print(f"⚠️  WARNING: {message}")
    
    def log_suggestion(self, message: str):
        """Log a suggestion"""
        self.suggestions.append(message)
        print(f"💡 SUGGESTION: {message}")
    
    def log_info(self, message: str):
        """Log information"""
        print(f"ℹ️  INFO: {message}")
    
    def check_env_file_exists(self) -> bool:
        """Check if .env file exists"""
        if not self.env_file.exists():
            self.log_error(".env file not found")
            if self.env_example_file.exists():
                self.log_suggestion(f"Copy {self.env_example_file} to {self.env_file} to get started")
            return False
        return True
    
    def load_env_variables(self) -> Dict[str, str]:
        """Load environment variables from .env file"""
        env_vars = {}
        if not self.env_file.exists():
            return env_vars
        
        try:
            with open(self.env_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"\'')
                            env_vars[key] = value
                        else:
                            self.log_warning(f"Invalid line format in .env at line {line_num}: {line}")
        except Exception as e:
            self.log_error(f"Failed to read .env file: {e}")
        
        return env_vars
    
    def validate_telegram_config(self, env_vars: Dict[str, str]) -> bool:
        """Validate Telegram bot configuration"""
        self.log_info("Validating Telegram configuration...")
        
        valid = True
        
        # Check bot token
        bot_token = env_vars.get('TELEGRAM_BOT_TOKEN', '')
        if not bot_token:
            self.log_error("TELEGRAM_BOT_TOKEN is not set")
            valid = False
        elif not re.match(r'^\d+:[A-Za-z0-9_-]+$', bot_token):
            self.log_error("TELEGRAM_BOT_TOKEN format is invalid")
            valid = False
        
        # Check admin user ID
        admin_user_id = env_vars.get('TELEGRAM_ADMIN_USER_ID', '')
        if not admin_user_id:
            self.log_warning("TELEGRAM_ADMIN_USER_ID is not set")
        elif not admin_user_id.isdigit():
            self.log_error("TELEGRAM_ADMIN_USER_ID must be a number")
            valid = False
        
        return valid
    
    def validate_indodax_config(self, env_vars: Dict[str, str]) -> bool:
        """Validate Indodax API configuration"""
        self.log_info("Validating Indodax API configuration...")
        
        valid = True
        
        # Check API key
        api_key = env_vars.get('INDODAX_API_KEY', '')
        if not api_key:
            self.log_error("INDODAX_API_KEY is not set")
            valid = False
        elif len(api_key) < 32:
            self.log_warning("INDODAX_API_KEY seems too short")
        
        # Check secret key
        secret_key = env_vars.get('INDODAX_SECRET_KEY', '')
        if not secret_key:
            self.log_error("INDODAX_SECRET_KEY is not set")
            valid = False
        elif len(secret_key) < 32:
            self.log_warning("INDODAX_SECRET_KEY seems too short")
        
        # Check base URL
        base_url = env_vars.get('INDODAX_BASE_URL', 'https://indodax.com')
        if not base_url.startswith('https://'):
            self.log_warning("INDODAX_BASE_URL should use HTTPS")
        
        return valid
    
    def validate_database_config(self, env_vars: Dict[str, str]) -> bool:
        """Validate database configuration"""
        self.log_info("Validating database configuration...")
        
        valid = True
        
        # Check database URL
        db_url = env_vars.get('DATABASE_URL', '')
        if not db_url:
            self.log_error("DATABASE_URL is not set")
            valid = False
        elif not db_url.startswith('postgresql'):
            self.log_error("DATABASE_URL must be a PostgreSQL connection string")
            valid = False
        
        # Check Redis URL
        redis_url = env_vars.get('REDIS_URL', '')
        if not redis_url:
            self.log_warning("REDIS_URL is not set, using default redis://localhost:6379")
        elif not redis_url.startswith('redis://'):
            self.log_warning("REDIS_URL should start with redis://")
        
        return valid
    
    def validate_ai_config(self, env_vars: Dict[str, str]) -> bool:
        """Validate AI/ML configuration"""
        self.log_info("Validating AI/ML configuration...")
        
        valid = True
        
        # Check OpenAI API key
        openai_key = env_vars.get('OPENAI_API_KEY', '')
        if not openai_key:
            self.log_warning("OPENAI_API_KEY is not set - AI features may be limited")
        elif not openai_key.startswith('sk-'):
            self.log_error("OPENAI_API_KEY format is invalid")
            valid = False
        
        return valid
    
    def validate_security_config(self, env_vars: Dict[str, str]) -> bool:
        """Validate security configuration"""
        self.log_info("Validating security configuration...")
        
        valid = True
        
        # Check secret key
        secret_key = env_vars.get('SECRET_KEY', '')
        if not secret_key:
            self.log_error("SECRET_KEY is not set")
            valid = False
        elif len(secret_key) < 32:
            self.log_warning("SECRET_KEY should be at least 32 characters long")
        
        # Check encryption key
        encryption_key = env_vars.get('ENCRYPTION_KEY', '')
        if not encryption_key:
            self.log_error("ENCRYPTION_KEY is not set")
            valid = False
        elif len(encryption_key) != 44:  # Fernet key length
            self.log_warning("ENCRYPTION_KEY should be 44 characters long (Fernet key)")
        
        return valid
    
    def validate_logging_config(self, env_vars: Dict[str, str]) -> bool:
        """Validate logging configuration"""
        self.log_info("Validating logging configuration...")
        
        valid = True
        
        # Check log level
        log_level = env_vars.get('LOG_LEVEL', 'INFO')
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if log_level.upper() not in valid_levels:
            self.log_error(f"LOG_LEVEL must be one of: {', '.join(valid_levels)}")
            valid = False
        
        return valid
    
    def validate_trading_config(self, env_vars: Dict[str, str]) -> bool:
        """Validate trading configuration"""
        self.log_info("Validating trading configuration...")
        
        valid = True
        
        # Check default stop loss
        stop_loss = env_vars.get('DEFAULT_STOP_LOSS', '5.0')
        try:
            stop_loss_value = float(stop_loss)
            if stop_loss_value <= 0 or stop_loss_value > 50:
                self.log_warning("DEFAULT_STOP_LOSS should be between 0 and 50 percent")
        except ValueError:
            self.log_error("DEFAULT_STOP_LOSS must be a number")
            valid = False
        
        # Check default take profit
        take_profit = env_vars.get('DEFAULT_TAKE_PROFIT', '10.0')
        try:
            take_profit_value = float(take_profit)
            if take_profit_value <= 0 or take_profit_value > 100:
                self.log_warning("DEFAULT_TAKE_PROFIT should be between 0 and 100 percent")
        except ValueError:
            self.log_error("DEFAULT_TAKE_PROFIT must be a number")
            valid = False
        
        # Check max trade amount
        max_trade = env_vars.get('MAX_TRADE_AMOUNT', '1000000')
        try:
            max_trade_value = float(max_trade)
            if max_trade_value <= 0:
                self.log_error("MAX_TRADE_AMOUNT must be positive")
                valid = False
        except ValueError:
            self.log_error("MAX_TRADE_AMOUNT must be a number")
            valid = False
        
        return valid
    
    def generate_missing_keys(self) -> Dict[str, str]:
        """Generate missing configuration keys"""
        import secrets
        from cryptography.fernet import Fernet
        
        missing_keys = {}
        
        # Generate secret key
        missing_keys['SECRET_KEY'] = secrets.token_urlsafe(32)
        
        # Generate encryption key
        missing_keys['ENCRYPTION_KEY'] = Fernet.generate_key().decode()
        
        return missing_keys
    
    def create_env_file_interactive(self):
        """Create .env file interactively"""
        print("\n🛠️  Interactive .env file creation")
        print("=" * 40)
        
        config = {}
        
        # Telegram configuration
        print("\n📱 Telegram Bot Configuration:")
        config['TELEGRAM_BOT_TOKEN'] = input("Telegram Bot Token: ").strip()
        config['TELEGRAM_ADMIN_USER_ID'] = input("Admin User ID (optional): ").strip()
        
        # Indodax configuration
        print("\n💰 Indodax API Configuration:")
        config['INDODAX_API_KEY'] = input("Indodax API Key: ").strip()
        config['INDODAX_SECRET_KEY'] = input("Indodax Secret Key: ").strip()
        config['INDODAX_BASE_URL'] = input("Indodax Base URL (default: https://indodax.com): ").strip() or "https://indodax.com"
        
        # Database configuration
        print("\n🗄️  Database Configuration:")
        config['DATABASE_URL'] = input("Database URL: ").strip()
        config['REDIS_URL'] = input("Redis URL (default: redis://localhost:6379): ").strip() or "redis://localhost:6379"
        
        # AI configuration
        print("\n🤖 AI Configuration (optional):")
        config['OPENAI_API_KEY'] = input("OpenAI API Key (optional): ").strip()
        
        # Generate security keys
        print("\n🔐 Generating security keys...")
        missing_keys = self.generate_missing_keys()
        config.update(missing_keys)
        
        # Trading configuration
        print("\n📈 Trading Configuration:")
        config['DEFAULT_STOP_LOSS'] = input("Default Stop Loss % (default: 5.0): ").strip() or "5.0"
        config['DEFAULT_TAKE_PROFIT'] = input("Default Take Profit % (default: 10.0): ").strip() or "10.0"
        config['MAX_TRADE_AMOUNT'] = input("Max Trade Amount (default: 1000000): ").strip() or "1000000"
        
        # Other configuration
        config['LOG_LEVEL'] = input("Log Level (default: INFO): ").strip() or "INFO"
        config['DEBUG'] = input("Debug Mode (true/false, default: false): ").strip() or "false"
        
        # Write .env file
        try:
            with open(self.env_file, 'w') as f:
                f.write("# Telebot AI Trading Bot Configuration\n")
                f.write("# Generated on: " + str(Path(__file__).stat().st_mtime) + "\n\n")
                
                for key, value in config.items():
                    if value:
                        f.write(f"{key}={value}\n")
                    else:
                        f.write(f"# {key}=\n")
            
            print(f"\n✅ .env file created successfully: {self.env_file}")
            return True
            
        except Exception as e:
            self.log_error(f"Failed to create .env file: {e}")
            return False
    
    def run_validation(self) -> bool:
        """Run complete configuration validation"""
        print("🔍 Telebot AI Trading Bot - Configuration Validation")
        print("=" * 60)
        
        # Check if .env file exists
        if not self.check_env_file_exists():
            print("\n❓ Would you like to create a .env file interactively? (y/n): ", end="")
            if input().lower().startswith('y'):
                return self.create_env_file_interactive()
            else:
                return False
        
        # Load environment variables
        env_vars = self.load_env_variables()
        if not env_vars:
            self.log_error("No environment variables found in .env file")
            return False
        
        print(f"\n📋 Found {len(env_vars)} environment variables")
        
        # Run all validations
        validations = [
            self.validate_telegram_config(env_vars),
            self.validate_indodax_config(env_vars),
            self.validate_database_config(env_vars),
            self.validate_ai_config(env_vars),
            self.validate_security_config(env_vars),
            self.validate_logging_config(env_vars),
            self.validate_trading_config(env_vars)
        ]
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        
        print(f"Total Errors: {len(self.errors)}")
        print(f"Total Warnings: {len(self.warnings)}")
        print(f"Total Suggestions: {len(self.suggestions)}")
        
        if self.errors:
            print("\n🔴 ERRORS (must be fixed):")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print("\n🟡 WARNINGS (should be addressed):")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if self.suggestions:
            print("\n💡 SUGGESTIONS:")
            for suggestion in self.suggestions:
                print(f"  - {suggestion}")
        
        # Overall result
        all_valid = all(validations) and len(self.errors) == 0
        
        if all_valid:
            print("\n🎉 Configuration validation PASSED!")
            print("Your bot is ready to run!")
        else:
            print("\n⚠️  Configuration validation FAILED!")
            print("Please fix the errors above before running the bot.")
        
        return all_valid

def main():
    """Main function"""
    validator = ConfigValidator()
    
    if len(sys.argv) > 1 and sys.argv[1] == "create":
        # Interactive creation mode
        success = validator.create_env_file_interactive()
        sys.exit(0 if success else 1)
    else:
        # Validation mode
        success = validator.run_validation()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
