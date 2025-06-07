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

Fill in your API credentials and trading parameters in the `.env` file.

### 3. Run the Dashboard

```bash
# Start the web dashboard
./run_dashboard.sh
```

The dashboard will be available at `http://localhost:8501`

## 📋 Configuration

### Environment Variables

#### Required Settings
```bash
# Indodax API (Required for trading)
INDODAX_API_KEY=your_api_key_here
INDODAX_SECRET_KEY=your_secret_key_here

# Trading Parameters
STOP_LOSS_PERCENT=5
TAKE_PROFIT_PERCENT=15
MAX_DAILY_TRADES=20
```

#### Optional Notifications
```bash
# Telegram Bot
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Twilio (SMS/WhatsApp)
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
```

## 🖥️ Dashboard Overview

### Pages Available:
1. **📊 Dashboard**: Portfolio overview, performance metrics, live charts
2. **🚀 Live Trading**: Real-time trading controls and monitoring
3. **📈 Market Analysis**: Technical indicators and AI predictions
4. **💼 Portfolio**: Asset allocation and performance tracking
5. **🔬 Backtesting**: Strategy testing on historical data
6. **⚙️ Settings**: Configuration and system controls
7. **🔔 Notifications**: Alert management and communication
8. **📋 Trading Log**: Complete trade history and analysis

## 🤖 AI Features

### Technical Analysis
- **RSI (Relative Strength Index)**: Momentum oscillator
- **MACD**: Moving Average Convergence Divergence
- **Moving Averages**: SMA, EMA trend analysis
- **Bollinger Bands**: Volatility and price level indicators
- **Volume Analysis**: Trading volume patterns

### AI Predictions
- **LSTM Neural Network**: Deep learning price prediction
- **Multi-timeframe Analysis**: 1m, 5m, 15m, 1h, 4h, 1d
- **Sentiment Analysis**: Market sentiment indicators
- **Pattern Recognition**: Chart pattern detection

## 🔄 Trading Strategies

### Available Strategies:
1. **Scalping**: Quick profits on small price movements
2. **Swing Trading**: Medium-term trend following
3. **DCA (Dollar Cost Averaging)**: Regular investment strategy
4. **Grid Trading**: Buy low, sell high in ranges
5. **Arbitrage**: Price difference exploitation

## 📊 Risk Management

### Built-in Protections:
- **Stop Loss**: Automatic loss limitation
- **Take Profit**: Profit taking automation
- **Position Sizing**: Risk-based trade sizing
- **Daily Limits**: Maximum trades and losses per day
- **Drawdown Protection**: Portfolio drawdown limits

## 🔔 Notifications

### Supported Channels:
- **Telegram**: Instant messages and reports
- **WhatsApp**: Via Twilio integration
- **SMS**: Text message alerts
- **Email**: Important notifications
- **Dashboard**: Real-time web notifications

## 📈 Performance Tracking

### Metrics Available:
- **Total Portfolio Value**: Real-time portfolio worth
- **P&L (Profit & Loss)**: Detailed profit/loss tracking
- **Win Rate**: Percentage of profitable trades
- **Average Return**: Mean return per trade
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest portfolio decline

## 🛠️ Technical Stack

### Core Technologies:
- **Python 3.8+**: Main programming language
- **Streamlit**: Web dashboard framework
- **TensorFlow/Keras**: AI/ML model development
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive charts and visualization

### APIs & Integrations:
- **Indodax API**: Cryptocurrency exchange
- **Yahoo Finance**: Market data backup
- **Telegram Bot API**: Messaging
- **Twilio API**: SMS/WhatsApp notifications

## 🚀 Deployment

### Production Deployment:

```bash
# Copy systemd service file
sudo cp ai-trading-bot.service /etc/systemd/system/

# Enable and start service
sudo systemctl enable ai-trading-bot
sudo systemctl start ai-trading-bot

# Check status
sudo systemctl status ai-trading-bot
```

### Docker Deployment (Optional):

```bash
# Build Docker image
docker build -t ai-trading-bot .

# Run container
docker run -d -p 8501:8501 --env-file .env ai-trading-bot
```

## 📝 Usage Examples

### Manual Trading:
```python
from main import AITradingBot

# Initialize bot
bot = AITradingBot()

# Get market analysis
analysis = bot.analyze_market('btcidr')

# Manual trade execution
bot.execute_trade('btcidr', 'buy', 0.001)
```

### Automated Trading:
```bash
# Run continuous trading
python main.py --symbol btcidr --strategy scalping --auto
```

## 🔧 Troubleshooting

### Common Issues:

1. **API Connection Error**:
   - Check API credentials in `.env`
   - Verify network connectivity
   - Check API rate limits

2. **Dashboard Not Loading**:
   - Ensure port 8501 is available
   - Check Python dependencies
   - Verify Streamlit installation

3. **Trading Not Working**:
   - Check account balance
   - Verify trading permissions
   - Check risk management settings

## 📊 Monitoring

### Log Files:
- `logs/trades.csv`: All trading activities
- `logs/portfolio.csv`: Portfolio snapshots
- `logs/signals.csv`: Trading signals
- `logs/errors.csv`: Error tracking

### Dashboard Monitoring:
- Real-time performance metrics
- Live trading activity
- System health indicators
- API connectivity status

## 🔒 Security

### Best Practices:
- Keep API keys secure in `.env`
- Use read-only API keys when possible
- Enable IP whitelisting on exchange
- Monitor trading activity regularly
- Set appropriate risk limits

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

**IMPORTANT**: This trading bot is for educational and research purposes. Cryptocurrency trading involves substantial risk of loss. Never trade with money you cannot afford to lose. The authors are not responsible for any financial losses incurred through the use of this software.

## 📞 Support

For support, questions, or feature requests:
- Create an issue on GitHub
- Join our Telegram group
- Email: support@example.com

---

**Happy Trading! 🚀📈**