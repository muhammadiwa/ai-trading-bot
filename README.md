# 📈 Telegram AI-Powered Trading Bot for Indodax

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-aiogram-green)
![Exchange](https://img.shields.io/badge/Exchange-Indodax-orange)
![AI](https://img.shields.io/badge/AI-Powered-purple)
![License](https://img.shields.io/badge/License-MIT-red)

**Bot trading cryptocurrency canggih dengan AI untuk Indodax Exchange**

[Features](#-features) • [Installation](#-installation) • [Configuration](#-configuration) • [Usage](#-usage) • [API](#-api-documentation)

</div>

---

## 🎯 Overview

Bot Telegram AI-powered untuk trading cryptocurrency di Indodax Exchange. Bot ini menggunakan kecerdasan buatan untuk:

- 📊 **Analisis Sinyal Trading** - AI menganalisis indikator teknis dan sentiment pasar
- 💹 **Trading Otomatis** - Eksekusi order otomatis berdasarkan sinyal AI
- 📈 **Manajemen Portfolio** - Tracking dan analisis portfolio real-time
- 🔔 **Notifikasi Real-time** - Alert untuk sinyal, eksekusi trade, dan perubahan pasar
- 🛡️ **Risk Management** - Stop loss, take profit, dan manajemen risiko otomatis

## ✨ Features

### 🤖 AI Trading Engine
- **Signal Generation**: AI menghasilkan sinyal buy/sell berdasarkan multiple indicators
- **Sentiment Analysis**: Analisis sentiment dari news dan social media  
- **Price Prediction**: Model LSTM untuk prediksi harga
- **Technical Analysis**: RSI, MACD, Bollinger Bands, Moving Averages

### 💼 Portfolio Management
- **Real-time Tracking**: Monitor portfolio value dan performance
- **Asset Allocation**: Analisis diversifikasi dan rebalancing
- **P&L Analysis**: Profit/Loss tracking dengan visualisasi
- **Risk Assessment**: Analisis risiko portfolio

### 🔄 Trading Features
- **Manual Trading**: Execute buy/sell via Telegram commands
- **Auto Trading**: Trading otomatis berdasarkan AI signals
- **Order Management**: View dan cancel active orders
- **DCA (Dollar Cost Averaging)**: Investasi berkala otomatis

### 🛡️ Risk Management
- **Stop Loss/Take Profit**: Automatic position management
- **Position Sizing**: Kontrol ukuran posisi maksimal
- **Daily Trade Limits**: Batasi jumlah trade per hari
- **Balance Validation**: Validasi saldo sebelum trade

### 🌐 Multi-language Support
- 🇮🇩 **Bahasa Indonesia** (Default)
- 🇺🇸 **English**

## 🚀 Installation

### Prerequisites

- Python 3.8 atau lebih tinggi
- PostgreSQL database
- Redis server
- Telegram Bot Token
- Indodax API credentials

### Quick Install

```bash
# Clone repository
git clone https://github.com/yourusername/telebot-ai.git
cd telebot-ai

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp config/.env.example config/.env

# Edit configuration
nano config/.env

# Initialize database
python -c "
import asyncio
from core.database import init_database
asyncio.run(init_database())
"

# Run the bot
python main.py
```

### Docker Installation

```bash
# Build and run with Docker Compose
docker-compose up -d
```

## ⚙️ Configuration

### Environment Variables

Copy `config/.env.example` to `config/.env` dan sesuaikan:

```bash
# Telegram Configuration
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_ADMIN_IDS=123456789,987654321

# Indodax API (Optional - users can add via bot)
INDODAX_API_KEY=your_indodax_api_key
INDODAX_SECRET_KEY=your_indodax_secret_key

# Database
DATABASE_URL=postgresql://username:password@localhost:5432/trading_bot_db
REDIS_URL=redis://localhost:6379/0

# AI/ML
OPENAI_API_KEY=your_openai_api_key

# Security
SECRET_KEY=your_super_secret_key_for_encryption

# Trading Settings
DEFAULT_TRADING_AMOUNT=100000
MAX_DAILY_TRADES=10
STOP_LOSS_PERCENTAGE=5.0
TAKE_PROFIT_PERCENTAGE=10.0
```

### Database Setup

```bash
# PostgreSQL
sudo -u postgres createdb trading_bot_db
sudo -u postgres psql -c "CREATE USER botuser WITH PASSWORD 'yourpassword';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE trading_bot_db TO botuser;"

# Redis
sudo systemctl start redis-server
```

## 📱 Usage

### Basic Commands

| Command | Description |
|---------|-------------|
| `/start` | 🚀 Mulai menggunakan bot |
| `/help` | ❓ Panduan lengkap |
| `/daftar` | 📝 Daftar akun trading |
| `/portfolio` | 💼 Lihat portfolio |
| `/balance` | 💰 Cek saldo |
| `/signal` | 📊 Dapatkan sinyal AI |
| `/buy` | 🟢 Beli cryptocurrency |
| `/sell` | 🔴 Jual cryptocurrency |
| `/orders` | 📋 Lihat order aktif |
| `/settings` | ⚙️ Pengaturan akun |

### Trading Examples

```bash
# Get AI signal for Bitcoin
/signal BTC

# Buy Bitcoin worth 1 million IDR
/buy BTC 1000000

# Sell 0.001 Bitcoin
/sell BTC 0.001

# Check portfolio
/portfolio

# View active orders
/orders
```

### Admin Commands

```bash
# Admin panel
/admin

# Broadcast message
/broadcast Hello semua users!

# Bot statistics
/stats

# Trading statistics
/trading_stats
```

## 🔧 API Documentation

### Indodax API Integration

Bot menggunakan Indodax REST API untuk:

#### Public Endpoints
- **Server Time**: `/api/server_time`
- **Trading Pairs**: `/api/pairs`
- **Ticker Data**: `/api/ticker/{pair_id}`
- **Order Book**: `/api/depth/{pair_id}`
- **Trade History**: `/api/trades/{pair_id}`
- **OHLC Data**: `/tradingview/history_v2`

#### Private Endpoints
- **Account Info**: `POST /tapi` (method: getInfo)
- **Balance**: `POST /tapi` (method: getInfo)
- **Place Order**: `POST /tapi` (method: trade)
- **Cancel Order**: `POST /tapi` (method: cancelOrder)
- **Open Orders**: `POST /tapi` (method: openOrders)
- **Order History**: `POST /tapi` (method: orderHistory)

### AI Models

#### Signal Generator
```python
from ai.signal_generator import SignalGenerator

generator = SignalGenerator()
signal = await generator.generate_signal("btc_idr")

# Signal output:
{
    "signal_type": "buy",  # buy, sell, hold
    "confidence": 0.85,    # 0.0 to 1.0
    "price_prediction": 650000000,
    "indicators": {
        "rsi": 35.2,
        "macd": "bullish",
        "bb_position": "lower"
    }
}
```

#### Sentiment Analyzer
```python
from ai.sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()
sentiment = await analyzer.analyze_sentiment("btc_idr")

# Sentiment output:
{
    "score": 0.25,         # -1.0 to 1.0
    "label": "positive",   # positive, negative, neutral
    "confidence": 0.8,
    "news_sentiment": {...},
    "social_sentiment": {...}
}
```

## 📊 Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Telegram Bot  │◄──►│   Backend API    │◄──►│   Indodax API   │
│   (aiogram)     │    │   (FastAPI)      │    │   (REST)        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Interface│    │   AI Engine      │    │   Database      │
│   Commands      │    │   LSTM/Prophet   │    │   PostgreSQL    │
│   Keyboards     │    │   Technical Analysis │  │   Redis Cache   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Tech Stack

| Component | Technology |
|-----------|------------|
| **Bot Framework** | aiogram 3.x |
| **Backend** | FastAPI |
| **Database** | PostgreSQL + Redis |
| **AI/ML** | PyTorch, TensorFlow, Prophet |
| **Exchange API** | Indodax REST API |
| **Deployment** | Docker, PM2 |
| **Monitoring** | Structlog, Sentry |

## 🔒 Security

### API Key Encryption
- API keys dienkripsi menggunakan Fernet (symmetric encryption)
- Secret keys tidak pernah disimpan dalam plain text
- Enkripsi berdasarkan SECRET_KEY environment variable

### Access Control
- Role-based access (admin, premium, regular users)
- Rate limiting untuk mencegah spam
- Input validation dan sanitization

### Trading Security
- Balance validation sebelum setiap trade
- Position size limits
- Daily trade limits
- Stop loss enforcement

## 📈 Performance

### Optimization Features
- **Redis Caching**: Cache price data dan signals
- **Connection Pooling**: Efficient database connections
- **Async Operations**: Non-blocking I/O operations
- **Background Tasks**: Scheduler untuk automated tasks

### Monitoring
- **Structured Logging**: JSON logs dengan Structlog
- **Error Tracking**: Sentry integration
- **Performance Metrics**: Response time monitoring
- **Health Checks**: Database dan API connectivity

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=./ --cov-report=html

# Run specific test
python -m pytest tests/test_trading.py -v
```

## 🚀 Deployment

### Production Deployment

```bash
# Using PM2
pm2 start main.py --name "trading-bot" --interpreter python3

# Using Docker
docker-compose -f docker-compose.prod.yml up -d

# Using systemd
sudo cp trading-bot.service /etc/systemd/system/
sudo systemctl enable trading-bot
sudo systemctl start trading-bot
```

### Environment Setup

```bash
# Production environment
export ENVIRONMENT=production
export DEBUG=false
export LOG_LEVEL=INFO

# Security
export SECRET_KEY=$(openssl rand -hex 32)
export DATABASE_URL="postgresql://user:pass@db:5432/trading_bot"
```

## 🤝 Contributing

1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install

# Run linting
black . && flake8 . && mypy .
```

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

## ⚠️ Disclaimer

**PERINGATAN PENTING:**

- Bot ini adalah tools untuk membantu trading, bukan financial advisor
- Trading cryptocurrency memiliki risiko tinggi
- Selalu lakukan riset sendiri sebelum trading
- Gunakan hanya dana yang siap hilang
- Developer tidak bertanggung jawab atas kerugian trading

## 📞 Support

- 🐛 **Bug Reports**: [Create Issue](https://github.com/yourusername/telebot-ai/issues)
- 💡 **Feature Requests**: [Create Issue](https://github.com/yourusername/telebot-ai/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/telebot-ai/discussions)
- 📧 **Email**: support@yourdomain.com

## 🙏 Acknowledgments

- [Indodax](https://indodax.com) untuk API documentation
- [aiogram](https://aiogram.dev) untuk Telegram Bot framework
- [FastAPI](https://fastapi.tiangolo.com) untuk backend framework
- [PyTorch](https://pytorch.org) untuk machine learning capabilities

---

<div align="center">

**Made with ❤️ in Indonesia**

⭐ **Star this repo if you find it helpful!** ⭐

</div>
