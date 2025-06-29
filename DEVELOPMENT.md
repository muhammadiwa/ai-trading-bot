# Development Guide - Telebot AI Trading Bot

## Table of Contents
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Debugging](#debugging)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Contributing](#contributing)

## Getting Started

### Prerequisites
- Python 3.8+
- PostgreSQL 12+
- Redis 5+
- Git
- Node.js (for web dashboard, optional)

### Quick Setup

1. **Clone and Setup**
   ```bash
   cd /var/www/html/telebot-ai
   ./complete_setup.sh
   ```

2. **Configure Environment**
   ```bash
   python3 validate_config.py create
   ```

3. **Initialize Database**
   ```bash
   python3 migrate_db.py init
   ```

4. **Health Check**
   ```bash
   python3 health_check.py
   ```

5. **Start Development**
   ```bash
   ./start.sh
   ```

## Development Environment

### Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Environment Variables
Copy `.env.example` to `.env` and configure:

```bash
cp config/.env.example .env
# Edit .env with your configuration
```

### Database Setup
```bash
# Initialize database
python3 migrate_db.py init

# Check migration status
python3 migrate_db.py status

# Create backup
python3 migrate_db.py backup
```

## Project Structure

```
telebot-ai/
├── ai/                     # AI/ML modules
│   ├── __init__.py
│   ├── signal_generator.py # AI signal generation
│   └── sentiment_analyzer.py # Sentiment analysis
├── bot/                    # Telegram bot interface
│   ├── __init__.py
│   ├── telegram_bot.py    # Main bot handler
│   ├── keyboards.py       # Inline keyboards
│   ├── messages.py        # Message templates
│   └── utils.py           # Bot utilities
├── config/                 # Configuration
│   ├── __init__.py
│   ├── settings.py        # Settings management
│   └── .env.example       # Environment template
├── core/                   # Core functionality
│   ├── __init__.py
│   ├── database.py        # Database models
│   ├── indodax_api.py     # Indodax API client
│   ├── scheduler.py       # Background tasks
│   ├── risk_manager.py    # Risk management
│   └── portfolio_manager.py # Portfolio management
├── data/                   # Data storage
│   ├── logs/              # Log files
│   ├── backups/           # Database backups
│   ├── models/            # AI models
│   └── cache/             # Cache files
├── tests/                  # Test files
│   ├── conftest.py        # Test configuration
│   └── test_trading.py    # Trading tests
├── main.py                 # Application entry point
├── requirements.txt        # Production dependencies
├── requirements-dev.txt    # Development dependencies
└── utilities/
    ├── complete_setup.sh   # Complete setup script
    ├── health_check.py     # Health monitoring
    ├── migrate_db.py       # Database migration
    ├── validate_config.py  # Configuration validation
    └── manage.sh           # Management script
```

## Development Workflow

### 1. Feature Development
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes
# Run tests
python -m pytest tests/ -v

# Check code quality
flake8 .
black .
isort .

# Commit changes
git add .
git commit -m "feat: add your feature description"
```

### 2. Database Changes
```bash
# Create migration
python3 migrate_db.py create

# Apply migration
python3 migrate_db.py migrate

# Check status
python3 migrate_db.py status
```

### 3. Testing
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_trading.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### 4. Configuration Management
```bash
# Validate configuration
python3 validate_config.py

# Create new configuration
python3 validate_config.py create
```

## Testing

### Unit Tests
```bash
# Run all unit tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_trading.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### Integration Tests
```bash
# Test API connections
python3 health_check.py

# Test database
python3 migrate_db.py test

# Test configuration
python3 validate_config.py
```

### Manual Testing
```bash
# Start bot in development mode
DEBUG=true python main.py

# Test specific commands
python -c "from bot.telegram_bot import TelegramBot; bot = TelegramBot(); print('Bot initialized')"
```

## Debugging

### Logging
```bash
# View live logs
tail -f data/logs/app.log

# View error logs
tail -f data/logs/error.log

# View trading logs
tail -f data/logs/trading.log
```

### Debug Mode
```bash
# Enable debug mode
export DEBUG=true
python main.py

# Or set in .env
DEBUG=true
```

### Database Debugging
```bash
# Connect to database
psql $DATABASE_URL

# Check tables
python3 migrate_db.py status

# View migration history
python3 migrate_db.py history
```

## API Documentation

### Indodax API Integration
The bot integrates with Indodax API for:
- Market data retrieval
- Order placement
- Portfolio management
- Balance checking

### Internal API Endpoints
If FastAPI is enabled, internal endpoints are available at:
- `GET /health` - Health check
- `GET /status` - Bot status
- `GET /metrics` - Performance metrics

### Telegram Bot Commands
- `/start` - Initialize bot
- `/help` - Show help
- `/balance` - Check balance
- `/portfolio` - View portfolio
- `/signals` - Get AI signals
- `/settings` - Configure settings

## Deployment

### Production Deployment
```bash
# Build production image
docker build -t telebot-ai .

# Run with docker-compose
docker-compose up -d

# Check status
docker-compose ps
```

### Systemd Service
```bash
# Install service
sudo systemctl enable trading-bot

# Start service
sudo systemctl start trading-bot

# Check status
sudo systemctl status trading-bot

# View logs
journalctl -u trading-bot -f
```

### Monitoring
```bash
# Health check
python3 health_check.py

# Monitor resources
python3 monitor.py

# Check logs
./manage.sh logs
```

## Contributing

### Code Style
- Use Black for code formatting
- Use isort for import sorting
- Follow PEP 8 guidelines
- Use type hints where possible

### Commit Messages
Follow conventional commits:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `style:` - Code style
- `refactor:` - Code refactoring
- `test:` - Testing
- `chore:` - Maintenance

### Pull Request Process
1. Fork the repository
2. Create feature branch
3. Make changes
4. Run tests
5. Submit pull request

### Development Environment
```bash
# Install pre-commit hooks
pre-commit install

# Run code quality checks
flake8 .
black .
isort .
mypy .
```

## Troubleshooting

### Common Issues

1. **Database Connection Issues**
   ```bash
   # Check database status
   python3 migrate_db.py test
   
   # Verify connection string
   python3 validate_config.py
   ```

2. **API Authentication Issues**
   ```bash
   # Test API keys
   python3 health_check.py
   
   # Verify configuration
   python3 validate_config.py
   ```

3. **Bot Not Responding**
   ```bash
   # Check bot status
   ./manage.sh status
   
   # View logs
   ./manage.sh logs
   ```

4. **Permission Issues**
   ```bash
   # Fix permissions
   sudo chown -R $USER:$USER /var/www/html/telebot-ai
   chmod +x *.sh
   ```

### Getting Help
- Check the logs: `./manage.sh logs`
- Run health check: `python3 health_check.py`
- Validate configuration: `python3 validate_config.py`
- Check system status: `./manage.sh status`

---

For more information, see the main [README.md](README.md) file.
