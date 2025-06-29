#!/bin/bash

# Installation script untuk Telegram Trading Bot AI
# Untuk Ubuntu/Debian systems

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please do not run this script as root"
    exit 1
fi

print_status "Starting installation of Telegram Trading Bot AI..."

# Update system packages
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
print_status "Installing system dependencies..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    postgresql \
    postgresql-contrib \
    redis-server \
    nginx \
    git \
    curl \
    build-essential \
    libpq-dev \
    pkg-config

# Install Node.js (for potential dashboard)
print_status "Installing Node.js..."
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install PM2 for process management
print_status "Installing PM2..."
sudo npm install -g pm2

# Setup PostgreSQL
print_status "Setting up PostgreSQL..."
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
print_status "Creating database..."
sudo -u postgres psql << EOF
CREATE DATABASE trading_bot_db;
CREATE USER botuser WITH PASSWORD 'changeme123!';
GRANT ALL PRIVILEGES ON DATABASE trading_bot_db TO botuser;
ALTER USER botuser CREATEDB;
\q
EOF

# Setup Redis
print_status "Setting up Redis..."
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Create application directory
APP_DIR="/var/www/html/telebot-ai"
print_status "Creating application directory at $APP_DIR..."

if [ ! -d "$APP_DIR" ]; then
    sudo mkdir -p "$APP_DIR"
    sudo chown $USER:$USER "$APP_DIR"
fi

# Clone or copy application if not exists
if [ ! -f "$APP_DIR/main.py" ]; then
    print_warning "Application files not found in $APP_DIR"
    print_status "Please copy your application files to $APP_DIR"
fi

cd "$APP_DIR"

# Create virtual environment
print_status "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
if [ -f "requirements.txt" ]; then
    print_status "Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    print_warning "requirements.txt not found, skipping Python dependencies"
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p data logs ai/models config

# Setup configuration
print_status "Setting up configuration..."
if [ ! -f "config/.env" ]; then
    if [ -f "config/.env.example" ]; then
        cp config/.env.example config/.env
        print_warning "Please edit config/.env with your configuration"
    else
        print_warning "config/.env.example not found"
    fi
fi

# Generate secret key
SECRET_KEY=$(openssl rand -hex 32)
print_status "Generated secret key: $SECRET_KEY"

# Update .env file with generated values
if [ -f "config/.env" ]; then
    sed -i "s/your_super_secret_key_for_encryption/$SECRET_KEY/g" config/.env
    sed -i "s/username:password@localhost:5432/botuser:changeme123!@localhost:5432/g" config/.env
fi

# Setup systemd service
print_status "Setting up systemd service..."
if [ -f "trading-bot.service" ]; then
    sudo cp trading-bot.service /etc/systemd/system/
    sudo systemctl daemon-reload
    print_success "Systemd service installed"
else
    print_warning "trading-bot.service not found"
fi

# Setup nginx (optional)
print_status "Setting up Nginx configuration..."
sudo tee /etc/nginx/sites-available/trading-bot << EOF
server {
    listen 80;
    server_name localhost;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Enable nginx site
sudo ln -sf /etc/nginx/sites-available/trading-bot /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# Set proper permissions
print_status "Setting file permissions..."
sudo chown -R $USER:www-data "$APP_DIR"
chmod -R 755 "$APP_DIR"
chmod +x run.sh

# Initialize database
print_status "Initializing database..."
if [ -f "main.py" ]; then
    python -c "
import asyncio
import sys
sys.path.append('.')
from core.database import init_database

async def main():
    try:
        await init_database()
        print('Database initialized successfully')
    except Exception as e:
        print(f'Database initialization failed: {e}')

asyncio.run(main())
" || print_warning "Database initialization failed"
else
    print_warning "main.py not found, skipping database initialization"
fi

# Setup firewall
print_status "Configuring firewall..."
sudo ufw allow ssh
sudo ufw allow 80
sudo ufw allow 443
sudo ufw --force enable

print_success "Installation completed!"
print_status "Next steps:"
echo "1. Edit config/.env with your configuration:"
echo "   - TELEGRAM_BOT_TOKEN"
echo "   - INDODAX_API_KEY (optional)"
echo "   - INDODAX_SECRET_KEY (optional)"
echo "   - Other settings as needed"
echo ""
echo "2. Start the bot:"
echo "   ./run.sh start"
echo ""
echo "3. Or enable systemd service:"
echo "   sudo systemctl enable trading-bot"
echo "   sudo systemctl start trading-bot"
echo ""
echo "4. Check logs:"
echo "   ./run.sh logs"
echo "   # or"
echo "   sudo journalctl -u trading-bot -f"
echo ""
print_warning "Remember to:"
echo "- Change default database password"
echo "- Configure SSL certificates for production"
echo "- Set up monitoring and backups"
echo "- Review security settings"
