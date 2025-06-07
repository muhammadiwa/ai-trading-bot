# AI Trading Bot - README

Aplikasi Bot Trading Otomatis Berbasis AI untuk cryptocurrency trading di platform Indodax.

## 🚀 Fitur Utama

### 🔍 Data Historis
- ✅ Ambil harga dari Indodax via API
- ✅ Penyimpanan data historis OHLCV
- ✅ Real-time market data

### 📈 Analisis Teknikal
- ✅ MACD (Moving Average Convergence Divergence)
- ✅ RSI (Relative Strength Index)
- ✅ Moving Average (SMA, EMA)
- ✅ Bollinger Bands
- ✅ Stochastic Oscillator
- ✅ Volume indicators

### 🤖 Prediksi Harga AI
- ✅ LSTM Deep Learning model
- ✅ Multi-feature prediction
- ✅ Confidence scoring
- ✅ Model training & evaluation

### 💸 Trading Otomatis
- ✅ Beli/Jual via Indodax API
- ✅ Signal-based trading
- ✅ Manual trading control

### 📲 Notifikasi Real-time
- ✅ WhatsApp (via Twilio)
- ✅ Telegram Bot
- ✅ Trading signals
- ✅ Portfolio updates

### 📁 Logging Transaksi
- ✅ CSV file logging
- ✅ Trade history
- ✅ Signal logging
- ✅ Error tracking

### ⚙️ Risk Management
- ✅ Stop Loss & Take Profit
- ✅ Position sizing
- ✅ Daily loss limits
- ✅ Max trades per day

### 📊 Backtesting
- ✅ Strategy testing
- ✅ Performance metrics
- ✅ Multiple strategies
- ✅ Risk analysis

### 🖥️ GUI Dashboard
- ✅ Streamlit web interface
- ✅ Real-time charts
- ✅ Portfolio overview
- ✅ Manual trading

## 📦 Instalasi

### 1. Clone Repository
```bash
git clone <repository-url>
cd ai-trading-bot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Konfigurasi Environment
Edit file `.env` dengan konfigurasi Anda:

```env
# Indodax API
INDODAX_API_KEY=your_indodax_api_key
INDODAX_SECRET_KEY=your_indodax_secret_key

# Telegram
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# WhatsApp (Twilio)
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
TWILIO_WHATSAPP_TO=whatsapp:+your_phone_number

# Trading Parameters
DEFAULT_TRADE_AMOUNT=100000
MAX_DAILY_TRADES=10
STOP_LOSS_PERCENT=5
TAKE_PROFIT_PERCENT=10
```

## 🚀 Penggunaan

### 1. Jalankan Dashboard Web
```bash
streamlit run dashboard.py
```

### 2. Jalankan Bot via Command Line
```bash
python main.py
```

### 3. Training Model AI
```python
from main import AITradingBot

bot = AITradingBot()
bot.train_ai_model('btcidr', retrain=True)
```

### 4. Backtesting Strategy
```python
results = bot.run_backtest('btcidr', 'ma_cross', days=30)
print(results['summary'])
```

## 📊 Dashboard Features

### 🏠 Dashboard
- Portfolio overview
- Performance metrics
- Recent activity
- Asset allocation

### 📈 Market Analysis
- Technical indicators
- AI predictions
- Market data
- Price charts

### 🎯 Trading Signals
- Signal breakdown
- Manual trading
- Signal strength
- Risk analysis

### 💼 Portfolio
- Holdings overview
- Performance tracking
- Trade history
- P&L analysis

### 🧪 Backtesting
- Strategy testing
- Performance metrics
- Trade simulation
- Risk analysis

### ⚙️ Settings
- API configuration
- Trading parameters
- Notifications
- AI model settings

## 🔧 Konfigurasi API

### Indodax API
1. Login ke akun Indodax
2. Buat API key di pengaturan
3. Masukkan API key dan secret ke `.env`

### Telegram Bot
1. Chat dengan @BotFather di Telegram
2. Buat bot baru dengan `/newbot`
3. Dapatkan token dan chat ID
4. Masukkan ke `.env`

### WhatsApp (Twilio)
1. Daftar akun Twilio
2. Dapatkan Account SID dan Auth Token
3. Setup WhatsApp sandbox
4. Masukkan konfigurasi ke `.env`

## 📊 Monitoring & Logging

### Log Files
- `logs/trades.csv` - Riwayat trading
- `logs/signals.csv` - Sinyal trading
- `logs/portfolio.csv` - Status portfolio
- `logs/errors.csv` - Error log

### Model Files
- `models/best_lstm_model.h5` - Trained LSTM model
- `models/scaler.pkl` - Data scaler

## ⚠️ Risk Management

### Built-in Risk Controls
- Stop loss otomatis
- Take profit targets
- Position sizing
- Daily loss limits
- Maximum trades per day

### Best Practices
- Mulai dengan amount kecil
- Test di paper trading dulu
- Monitor performance secara berkala
- Set risk tolerance yang sesuai

## 📱 Notifikasi

### Signal Notifications
- Buy/Sell signals
- AI confidence levels
- Technical indicators
- Risk warnings

### Trade Notifications
- Order execution
- Fill confirmations
- P&L updates
- Portfolio changes

### Alert Notifications
- Error alerts
- Risk limit warnings
- System status
- Daily reports

## 🧪 Testing

### Test Notifications
```python
import asyncio
from main import AITradingBot

bot = AITradingBot()
results = asyncio.run(bot.test_notifications())
print(results)
```

### Backtest Strategy
```python
results = bot.run_backtest(
    pair='btcidr',
    strategy='ma_cross',
    days=30
)
```

## 📈 Performance Metrics

### Trading Metrics
- Total return
- Win rate
- Maximum drawdown
- Sharpe ratio
- Average trade size

### AI Model Metrics
- Prediction accuracy
- Direction accuracy
- Confidence levels
- Model validation loss

## 🔄 Maintenance

### Daily Tasks
- Check bot status
- Review trading performance
- Monitor error logs
- Verify notifications

### Weekly Tasks
- Retrain AI model
- Analyze strategy performance
- Update risk parameters
- Backup data

## 🆘 Troubleshooting

### Common Issues

1. **API Connection Failed**
   - Periksa API key dan secret
   - Pastikan koneksi internet stabil
   - Cek status Indodax API

2. **Model Training Error**
   - Pastikan data historis mencukupi
   - Check memory usage
   - Verify TensorFlow installation

3. **Notification Failed**
   - Periksa token Telegram/Twilio
   - Verify chat ID/phone number
   - Test connection manually

### Error Codes
- `DATA_FETCH_ERROR` - Gagal ambil data market
- `ANALYSIS_ERROR` - Error dalam analisis teknikal
- `TRADE_ERROR` - Gagal eksekusi trade
- `AI_TRAINING_ERROR` - Error training model

## 📞 Support

Untuk bantuan teknis atau pertanyaan:
- Baca dokumentasi lengkap
- Check error logs
- Review configuration
- Test components individually

## ⚖️ Disclaimer

**PERINGATAN**: Trading cryptocurrency memiliki risiko tinggi. Bot ini adalah tool bantuan dan tidak menjamin profit. Gunakan dengan risiko sendiri dan pastikan untuk:

- Test dengan amount kecil dulu
- Pahami cara kerja bot
- Monitor performance secara berkala
- Set risk management yang tepat
- Jangan invest lebih dari yang sanggup hilang

Pengembang tidak bertanggung jawab atas kerugian trading yang terjadi.

## 📄 License

MIT License - Silakan gunakan dan modifikasi sesuai kebutuhan.

---

**Happy Trading! 🚀📈**
