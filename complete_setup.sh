#!/bin/bash

# Telebot AI Trading Bot - Complete Setup and Installation Script
# This script completes the setup and installation of the Telegram Trading Bot AI

set -e

echo "🚀 Starting Telebot AI Trading Bot Complete Setup..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Check if script is run as root for system-wide installation
if [[ $EUID -eq 0 ]]; then
   print_warning "This script should not be run as root. Please run as regular user."
   exit 1
fi

PROJECT_DIR="/var/www/html/telebot-ai"
VENV_DIR="$PROJECT_DIR/venv"

print_header "=== Telebot AI Trading Bot Complete Setup ==="

# Step 1: Check system requirements
print_status "Checking system requirements..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
REQUIRED_VERSION="3.8"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    print_error "Python 3.8 or higher is required. Current version: $PYTHON_VERSION"
    exit 1
fi

print_status "Python version $PYTHON_VERSION is compatible ✓"

# Check if required system packages are installed
REQUIRED_PACKAGES=("git" "curl" "wget" "build-essential" "python3-dev" "python3-pip" "python3-venv")

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! dpkg -l | grep -q "^ii  $package "; then
        print_warning "$package is not installed. Installing..."
        sudo apt-get update
        sudo apt-get install -y "$package"
    fi
done

print_status "System requirements check completed ✓"

# Step 2: Set up project directory permissions
print_status "Setting up project directory permissions..."

if [ ! -d "$PROJECT_DIR" ]; then
    print_error "Project directory $PROJECT_DIR does not exist."
    exit 1
fi

# Ensure proper ownership
sudo chown -R $USER:$USER "$PROJECT_DIR"
chmod +x "$PROJECT_DIR/run.sh"
chmod +x "$PROJECT_DIR/install.sh"

print_status "Directory permissions set ✓"

# Step 3: Create Python virtual environment
print_status "Creating Python virtual environment..."

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    print_status "Virtual environment created ✓"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip

print_status "Virtual environment ready ✓"

# Step 4: Install Python dependencies
print_status "Installing Python dependencies..."

# Install main requirements
pip install -r "$PROJECT_DIR/requirements.txt"

# Install development requirements
pip install -r "$PROJECT_DIR/requirements-dev.txt"

print_status "Python dependencies installed ✓"

# Step 5: Set up database
print_status "Setting up database..."

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
    print_warning "PostgreSQL is not installed. Installing..."
    sudo apt-get update
    sudo apt-get install -y postgresql postgresql-contrib
    sudo systemctl start postgresql
    sudo systemctl enable postgresql
fi

# Check if Redis is installed
if ! command -v redis-server &> /dev/null; then
    print_warning "Redis is not installed. Installing..."
    sudo apt-get update
    sudo apt-get install -y redis-server
    sudo systemctl start redis-server
    sudo systemctl enable redis-server
fi

print_status "Database setup completed ✓"

# Step 6: Create environment configuration
print_status "Setting up environment configuration..."

if [ ! -f "$PROJECT_DIR/.env" ]; then
    cp "$PROJECT_DIR/config/.env.example" "$PROJECT_DIR/.env"
    print_warning "Created .env file from template. Please configure it with your API keys!"
    print_warning "Edit $PROJECT_DIR/.env with your actual configuration"
else
    print_status "Environment file already exists"
fi

# Step 7: Create log directories
print_status "Creating log directories..."
mkdir -p "$PROJECT_DIR/data/logs"
mkdir -p "$PROJECT_DIR/data/backups"
mkdir -p "$PROJECT_DIR/data/models"
mkdir -p "$PROJECT_DIR/data/cache"

print_status "Log directories created ✓"

# Step 8: Set up systemd service
print_status "Installing systemd service..."

# Copy service file to system directory
sudo cp "$PROJECT_DIR/trading-bot.service" "/etc/systemd/system/"

# Update service file with correct paths
sudo sed -i "s|/path/to/your/project|$PROJECT_DIR|g" "/etc/systemd/system/trading-bot.service"
sudo sed -i "s|User=your-username|User=$USER|g" "/etc/systemd/system/trading-bot.service"

# Reload systemd
sudo systemctl daemon-reload

print_status "Systemd service installed ✓"

# Step 9: Create startup script
print_status "Setting up startup script..."

cat > "$PROJECT_DIR/start.sh" << EOF
#!/bin/bash
# Start script for Telebot AI Trading Bot

cd "$PROJECT_DIR"
source venv/bin/activate

# Check if .env exists
if [ ! -f .env ]; then
    echo "Error: .env file not found. Please copy .env.example to .env and configure it."
    exit 1
fi

# Start the bot
python main.py
EOF

chmod +x "$PROJECT_DIR/start.sh"

print_status "Startup script created ✓"

# Step 10: Run initial tests
print_status "Running initial tests..."

cd "$PROJECT_DIR"

# Test import of main modules
python3 -c "
try:
    from config.settings import settings
    from bot.telegram_bot import TelegramBot
    from core.indodax_api import IndodaxAPI
    from ai.signal_generator import SignalGenerator
    print('✓ All main modules can be imported successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
    exit(1)
"

# Run unit tests if possible
if [ -f "requirements-dev.txt" ]; then
    print_status "Running unit tests..."
    python -m pytest tests/ -v || print_warning "Some tests failed. Check test output."
fi

print_status "Initial tests completed ✓"

# Step 11: Create management scripts
print_status "Creating management scripts..."

# Create bot management script
cat > "$PROJECT_DIR/manage.sh" << 'EOF'
#!/bin/bash

# Management script for Telebot AI Trading Bot

PROJECT_DIR="/var/www/html/telebot-ai"
SERVICE_NAME="trading-bot"

case "$1" in
    start)
        echo "Starting Telebot AI Trading Bot..."
        sudo systemctl start $SERVICE_NAME
        ;;
    stop)
        echo "Stopping Telebot AI Trading Bot..."
        sudo systemctl stop $SERVICE_NAME
        ;;
    restart)
        echo "Restarting Telebot AI Trading Bot..."
        sudo systemctl restart $SERVICE_NAME
        ;;
    status)
        sudo systemctl status $SERVICE_NAME
        ;;
    logs)
        journalctl -u $SERVICE_NAME -f
        ;;
    enable)
        echo "Enabling Telebot AI Trading Bot to start on boot..."
        sudo systemctl enable $SERVICE_NAME
        ;;
    disable)
        echo "Disabling Telebot AI Trading Bot from starting on boot..."
        sudo systemctl disable $SERVICE_NAME
        ;;
    backup)
        echo "Creating backup..."
        tar -czf "/tmp/telebot-ai-backup-$(date +%Y%m%d-%H%M%S).tar.gz" \
            --exclude="venv" \
            --exclude="__pycache__" \
            --exclude="*.pyc" \
            --exclude="data/logs" \
            "$PROJECT_DIR"
        echo "Backup created in /tmp/"
        ;;
    update)
        echo "Updating dependencies..."
        cd "$PROJECT_DIR"
        source venv/bin/activate
        pip install --upgrade -r requirements.txt
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|enable|disable|backup|update}"
        exit 1
        ;;
esac
EOF

chmod +x "$PROJECT_DIR/manage.sh"

print_status "Management scripts created ✓"

# Step 12: Create monitoring script
cat > "$PROJECT_DIR/monitor.py" << 'EOF'
#!/usr/bin/env python3
"""
Simple monitoring script for Telebot AI Trading Bot
"""
import subprocess
import time
import logging
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import psutil
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/www/html/telebot-ai/data/logs/monitor.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def check_service_status():
    """Check if the trading bot service is running"""
    try:
        result = subprocess.run(
            ['systemctl', 'is-active', 'trading-bot'],
            capture_output=True,
            text=True
        )
        return result.stdout.strip() == 'active'
    except Exception as e:
        logger.error(f"Error checking service status: {e}")
        return False

def check_system_resources():
    """Check system resource usage"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    disk_percent = psutil.disk_usage('/').percent
    
    return {
        'cpu': cpu_percent,
        'memory': memory_percent,
        'disk': disk_percent
    }

def send_alert(subject, message):
    """Send alert email (configure SMTP settings)"""
    # Configure your SMTP settings here
    pass

def main():
    """Main monitoring loop"""
    logger.info("Starting Telebot AI Trading Bot Monitor")
    
    while True:
        try:
            # Check service status
            if not check_service_status():
                logger.warning("Trading bot service is not running!")
                # Attempt to restart
                subprocess.run(['sudo', 'systemctl', 'start', 'trading-bot'])
                time.sleep(10)
                
                if not check_service_status():
                    logger.error("Failed to restart trading bot service!")
                    send_alert("Trading Bot Alert", "Failed to restart trading bot service")
            
            # Check system resources
            resources = check_system_resources()
            if resources['cpu'] > 90:
                logger.warning(f"High CPU usage: {resources['cpu']}%")
            if resources['memory'] > 90:
                logger.warning(f"High memory usage: {resources['memory']}%")
            if resources['disk'] > 90:
                logger.warning(f"High disk usage: {resources['disk']}%")
            
            logger.info(f"System OK - CPU: {resources['cpu']}%, Memory: {resources['memory']}%, Disk: {resources['disk']}%")
            
        except Exception as e:
            logger.error(f"Monitor error: {e}")
        
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main()
EOF

chmod +x "$PROJECT_DIR/monitor.py"

print_status "Monitoring script created ✓"

# Final steps
print_header "=== Setup Complete! ==="
print_status "Telebot AI Trading Bot has been successfully set up!"

echo ""
print_header "📋 Next Steps:"
echo "1. Configure your API keys in $PROJECT_DIR/.env"
echo "2. Set up your database connection in .env"
echo "3. Configure your Telegram bot token in .env"
echo "4. Test the configuration: cd $PROJECT_DIR && ./start.sh"
echo "5. Enable the service: sudo systemctl enable trading-bot"
echo "6. Start the service: sudo systemctl start trading-bot"

echo ""
print_header "🛠️  Useful Commands:"
echo "- Start bot: $PROJECT_DIR/manage.sh start"
echo "- Stop bot: $PROJECT_DIR/manage.sh stop"
echo "- Check status: $PROJECT_DIR/manage.sh status"
echo "- View logs: $PROJECT_DIR/manage.sh logs"
echo "- Backup: $PROJECT_DIR/manage.sh backup"

echo ""
print_header "📁 Important Paths:"
echo "- Project: $PROJECT_DIR"
echo "- Virtual env: $VENV_DIR"
echo "- Config: $PROJECT_DIR/.env"
echo "- Logs: $PROJECT_DIR/data/logs/"
echo "- Service: /etc/systemd/system/trading-bot.service"

echo ""
print_warning "Don't forget to:"
echo "- Configure your .env file with real API keys"
echo "- Test the bot in a safe environment first"
echo "- Set up proper monitoring and alerting"
echo "- Regular backups of your configuration and data"

echo ""
print_status "Setup completed successfully! Happy trading! 🚀"

deactivate
