# 🤖 AI Trading Bot

A sophisticated AI-powered cryptocurrency trading bot with real-time market analysis, automated trading strategies, risk management, and a beautiful web interface built with Streamlit.

![AI Trading Bot Dashboard](https://img.shields.io/badge/Status-Active-green) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)

## ✨ Features

### 🎯 Core Trading Features
- **Automated Trading**: AI-powered buy/sell decisions based on multiple indicators
- **Real-time Market Data**: Live price feeds from Indodax exchange
- **Technical Analysis**: RSI, MACD, Moving Averages, Bollinger Bands
- **AI Predictions**: LSTM neural network for price forecasting
- **Risk Management**: Stop-loss, take-profit, position sizing
- **Backtesting**: Test strategies on historical data

### 📊 Dashboard Features
- **Modern Web Interface**: Beautiful, responsive Streamlit dashboard
- **Live Charts**: Real-time candlestick charts with technical indicators
- **Portfolio Tracking**: Monitor assets, P&L, and performance
- **Trading Signals**: Visual buy/sell/hold recommendations
- **Market Analysis**: Comprehensive technical and AI analysis
- **Trade History**: Complete log of all trading activities

### 🔔 Communication Features
- **Telegram Notifications**: Real-time trade alerts and reports
- **WhatsApp Integration**: SMS and WhatsApp notifications via Twilio
- **Email Alerts**: Important notifications via email
- **Daily Reports**: Automated portfolio performance summaries

### 🛡️ Security & Risk Management
- **API Security**: Encrypted API key storage
- **Risk Controls**: Maximum daily trades, position limits
- **Stop Loss/Take Profit**: Automatic risk management
- **Portfolio Limits**: Maximum exposure controls

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Linux/Ubuntu server (recommended)
- Indodax API account
- Telegram Bot (optional)
- Twilio account (optional)

### 1. Installation

```bash
# Navigate to project directory
cd /var/www/html/ai-trading-bot

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration file
nano .env
```

Fill in your API credentials:
```env
# Indodax API (Required)
INDODAX_API_KEY=your_api_key_here
INDODAX_SECRET_KEY=your_secret_key_here

# Telegram Bot (Optional)
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Twilio for SMS/WhatsApp (Optional)
TWILIO_ACCOUNT_SID=your_twilio_sid_here
TWILIO_AUTH_TOKEN=your_twilio_token_here
```

### 3. Run the Dashboard

```bash
# Make startup script executable
chmod +x run_dashboard.sh

# Start the dashboard
./run_dashboard.sh
```

The dashboard will be available at `http://localhost:8501`

### 4. Alternative: Command Line Mode

```bash
# Run the bot in command line mode
python main.py
```

## 📱 Web Dashboard

### Dashboard Pages

1. **🏠 Dashboard**: Overview, portfolio metrics, recent activity
2. **📈 Live Trading**: Real-time trading interface with charts
3. **🎯 Market Analysis**: Technical indicators and AI predictions
4. **💼 Portfolio**: Asset allocation, performance tracking
5. **🔄 Backtesting**: Strategy testing on historical data
6. **⚙️ Settings**: Configuration and parameters
7. **📱 Notifications**: Test and manage alerts
8. **📝 Trading Log**: Complete trade history

### Key Features

- **Real-time Data**: Live price updates and market data
- **Interactive Charts**: Candlestick charts with technical indicators
- **Trading Controls**: Manual buy/sell with order management
- **AI Insights**: Machine learning predictions and confidence levels
- **Risk Monitoring**: Real-time portfolio risk assessment

## 🧠 AI & Machine Learning

### LSTM Neural Network
- **Architecture**: Multi-layer LSTM for price prediction
- **Features**: Price, volume, technical indicators
- **Training**: Automatic model training on historical data
- **Prediction**: 1-hour to 7-day price forecasts

### Technical Analysis
- **RSI**: Relative Strength Index for momentum
- **MACD**: Moving Average Convergence Divergence
- **SMA/EMA**: Simple and Exponential Moving Averages
- **Bollinger Bands**: Volatility and trend analysis

## ⚙️ Configuration

### Trading Parameters
```env
STOP_LOSS_PERCENT=5          # Stop loss percentage
TAKE_PROFIT_PERCENT=15       # Take profit percentage
MAX_DAILY_TRADES=20          # Maximum trades per day
RISK_PER_TRADE=0.02         # Risk 2% per trade
```

### AI Model Settings
```env
LSTM_EPOCHS=50              # Training epochs
BATCH_SIZE=32               # Training batch size
PREDICTION_DAYS=7           # Prediction horizon
SEQUENCE_LENGTH=60          # Input sequence length
```

## 📊 API Integration

### Supported Exchanges
- **Indodax**: Indonesian cryptocurrency exchange
- **Extensible**: Easy to add other exchanges

### API Endpoints
- Market data and ticker information
- Account balance and portfolio
- Order placement and management
- Trade history and transactions

## 🔔 Notifications

### Telegram Setup
1. Create a bot with @BotFather
2. Get bot token and chat ID
3. Add credentials to .env file

### Twilio Setup
1. Create Twilio account
2. Get Account SID and Auth Token
3. Configure phone numbers for SMS/WhatsApp

### Notification Types
- Trade execution alerts
- Price target notifications
- Daily portfolio reports
- System status updates
- Error and warning messages

## 🛡️ Security Best Practices

### API Security
- Store API keys in environment variables
- Use read-only API keys when possible
- Enable IP whitelisting on exchange
- Regular key rotation

### System Security
- Run with limited user privileges
- Use HTTPS for web interface
- Regular security updates
- Monitor system logs

## 📈 Performance & Monitoring

### Metrics Tracking
- Portfolio performance
- Trade success rates
- Risk-adjusted returns
- Maximum drawdown
- Sharpe ratio

### Logging
- All trades logged with timestamps
- Error tracking and debugging
- Performance metrics
- System health monitoring

## 🔧 Advanced Configuration

### Production Deployment

```bash
# Install as systemd service
sudo cp ai-trading-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ai-trading-bot
sudo systemctl start ai-trading-bot
```

### Nginx Reverse Proxy
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Docker Deployment
```dockerfile
# Build Docker image
docker build -t ai-trading-bot .

# Run container
docker run -d -p 8501:8501 --env-file .env ai-trading-bot
```

## 🧪 Testing

### Backtesting
- Test strategies on historical data
- Multiple timeframes and parameters
- Risk-adjusted performance metrics
- Visual results and analysis

### Paper Trading
- Test with simulated funds
- Real market conditions
- No financial risk
- Strategy validation

## 📝 Logging & Debugging

### Log Files
- `logs/trading.log`: Trade execution log
- `logs/errors.log`: Error and exception log
- `logs/market_data.log`: Market data log
- `logs/notifications.log`: Notification log

### Debug Mode
```bash
# Enable debug mode
export DEBUG=True
python main.py
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make your changes
4. Add tests if applicable
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

**Important**: This software is for educational and research purposes only. Cryptocurrency trading carries significant financial risk. The developers are not responsible for any financial losses. Always do your own research and never invest more than you can afford to lose.

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check documentation and FAQ
- Review log files for errors
- Test with paper trading first

## 🔄 Updates

### Version 1.0
- Initial release
- Basic trading functionality
- Streamlit dashboard
- Telegram notifications

### Roadmap
- [ ] Additional exchanges (Binance, Coinbase)
- [ ] More AI models (Transformer, ARIMA)
- [ ] Mobile app
- [ ] Advanced risk management
- [ ] Social trading features
- [ ] Multi-timeframe analysis

---

**Made with ❤️ for the crypto trading community**
