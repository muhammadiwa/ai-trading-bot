#!/bin/bash

# Trading Bot Startup Script
# Usage: ./run.sh [development|production|test]

set -e

# Default environment
ENV=${1:-development}

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    if ! command_exists python3; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    if ! command_exists pip; then
        print_error "pip is not installed"
        exit 1
    fi
    
    print_success "Dependencies check passed"
}

# Function to setup virtual environment
setup_venv() {
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    print_status "Activating virtual environment..."
    source venv/bin/activate
    
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt
}

# Function to check environment variables
check_env() {
    print_status "Checking environment configuration..."
    
    if [ ! -f "config/.env" ]; then
        print_warning "config/.env not found, copying from example..."
        cp config/.env.example config/.env
        print_warning "Please edit config/.env with your configuration"
        return 1
    fi
    
    # Load environment variables
    export $(cat config/.env | grep -v '^#' | xargs)
    
    # Check required variables
    required_vars=("TELEGRAM_BOT_TOKEN" "SECRET_KEY")
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            print_error "Required environment variable $var is not set"
            return 1
        fi
    done
    
    print_success "Environment configuration check passed"
}

# Function to setup database
setup_database() {
    print_status "Setting up database..."
    
    if [ "$ENV" = "development" ]; then
        # For development, use SQLite if PostgreSQL is not available
        if ! command_exists psql; then
            print_warning "PostgreSQL not found, using SQLite for development"
            export DATABASE_URL="sqlite:///./trading_bot.db"
        fi
    fi
    
    # Initialize database
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
        exit(1)

asyncio.run(main())
"
    
    print_success "Database setup completed"
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    
    if command_exists pytest; then
        pytest tests/ -v --cov=./ --cov-report=term-missing
    else
        print_warning "pytest not installed, skipping tests"
    fi
}

# Function to start the application
start_app() {
    print_status "Starting Trading Bot in $ENV mode..."
    
    case $ENV in
        "development")
            export DEBUG=true
            export LOG_LEVEL=DEBUG
            python main.py
            ;;
        "production")
            export DEBUG=false
            export LOG_LEVEL=INFO
            
            if command_exists pm2; then
                pm2 start main.py --name "trading-bot" --interpreter python3
                print_success "Trading Bot started with PM2"
            else
                print_warning "PM2 not found, starting in foreground"
                python main.py
            fi
            ;;
        "test")
            run_tests
            ;;
        *)
            print_error "Unknown environment: $ENV"
            print_status "Usage: $0 [development|production|test]"
            exit 1
            ;;
    esac
}

# Function to stop the application
stop_app() {
    print_status "Stopping Trading Bot..."
    
    if command_exists pm2; then
        pm2 stop trading-bot
        pm2 delete trading-bot
        print_success "Trading Bot stopped"
    else
        print_warning "PM2 not found, please stop manually with Ctrl+C"
    fi
}

# Function to show logs
show_logs() {
    if command_exists pm2; then
        pm2 logs trading-bot
    else
        print_warning "PM2 not found, logs not available"
    fi
}

# Function to show status
show_status() {
    if command_exists pm2; then
        pm2 status trading-bot
    else
        print_warning "PM2 not found, status not available"
    fi
}

# Main execution
main() {
    case $1 in
        "start")
            check_dependencies
            setup_venv
            check_env || exit 1
            setup_database
            start_app
            ;;
        "stop")
            stop_app
            ;;
        "restart")
            stop_app
            sleep 2
            main start
            ;;
        "logs")
            show_logs
            ;;
        "status")
            show_status
            ;;
        "test")
            check_dependencies
            setup_venv
            ENV="test"
            run_tests
            ;;
        *)
            # Default behavior - start with environment
            check_dependencies
            setup_venv
            check_env || exit 1
            setup_database
            start_app
            ;;
    esac
}

# Handle script arguments
if [ $# -eq 0 ]; then
    main "start"
else
    main "$@"
fi
