"""
Message templates for the Telegram bot in multiple languages
"""
from datetime import datetime
from typing import Dict, Any, List
from core.database import AISignal, Trade

class Messages:
    """Message templates for the bot in multiple languages"""
    
    def __init__(self):
        self.messages = {
            "id": {
                "welcome": """
🤖 <b>Selamat datang di AI Trading Bot Indodax!</b>

Bot ini membantu Anda trading cryptocurrency di Indodax dengan dukungan AI untuk:
• 📊 Analisis sinyal trading
• 💹 Otomasi trading
• 📈 Manajemen portfolio
• 🔔 Notifikasi real-time

<b>Mulai dengan mengetik /daftar untuk menghubungkan akun Indodax Anda</b>

⚠️ <i>Disclaimer: Trading cryptocurrency memiliki risiko tinggi. Gunakan bot ini dengan bijak dan sesuai kemampuan Anda.</i>
                """,
                
                "help": """
📖 <b>Panduan Penggunaan Bot</b>

<b>📝 Pendaftaran:</b>
/daftar - Daftarkan akun dan hubungkan dengan Indodax

<b>💼 Portfolio & Saldo:</b>
/portfolio - Lihat portfolio lengkap
/balance - Cek saldo akun
/orders - Lihat order aktif

<b>💹 Trading:</b>
/buy - Beli cryptocurrency
/sell - Jual cryptocurrency
/signal - Dapatkan sinyal AI

<b>⚙️ Pengaturan:</b>
/settings - Pengaturan akun
/bahasa - Ubah bahasa

<b>🔧 Admin (khusus admin):</b>
/admin - Panel admin
/broadcast - Kirim pesan broadcast
/stats - Statistik bot

<b>❓ Butuh bantuan?</b>
Hubungi admin untuk dukungan teknis.
                """,
                
                "registration": """
📝 <b>Pendaftaran Akun Trading</b>

Untuk menggunakan fitur trading, Anda perlu menghubungkan akun Indodax:

1️⃣ Login ke <a href="https://indodax.com">Indodax.com</a>
2️⃣ Buka menu API Management
3️⃣ Buat API Key baru dengan permission:
   • ✅ View (untuk melihat saldo)
   • ✅ Trade (untuk trading)
   • ❌ Withdraw (jangan aktifkan untuk keamanan)

4️⃣ Copy API Key Anda dan kirim ke chat ini

⚠️ <b>Keamanan:</b>
• API Key akan dienkripsi dan disimpan aman
• Bot tidak dapat withdraw dana Anda
• Anda bisa cabut akses kapan saja

<b>Masukkan API Key Indodax Anda:</b>
                """,
                
                "trade_selection": """
💹 <b>Pilih Cryptocurrency untuk {trade_type}</b>

Pilih pasangan trading yang ingin Anda {trade_type}:
                """,
                
                "enter_amount": """
💰 <b>Masukkan Jumlah</b>

Masukkan jumlah dalam IDR yang ingin Anda {trade_type}:

Contoh: 100000 (untuk Rp 100.000)
                """,
                
                "settings_menu": """
⚙️ <b>Pengaturan Akun</b>

Kelola preferensi trading dan pengaturan akun Anda:

• 🛑 Stop Loss: Batas kerugian otomatis
• 🎯 Take Profit: Target keuntungan otomatis  
• 💰 Batas Trade: Maksimal amount per trade
• 🔄 Auto Trading: Trading otomatis dengan AI
• 🔔 Notifikasi: Pengaturan alert
• 🌐 Bahasa: Ubah bahasa interface
                """
            },
            
            "en": {
                "welcome": """
🤖 <b>Welcome to AI Trading Bot for Indodax!</b>

This bot helps you trade cryptocurrency on Indodax with AI support for:
• 📊 Trading signal analysis
• 💹 Trading automation
• 📈 Portfolio management
• 🔔 Real-time notifications

<b>Start by typing /register to connect your Indodax account</b>

⚠️ <i>Disclaimer: Cryptocurrency trading has high risks. Use this bot wisely and according to your capacity.</i>
                """,
                
                "help": """
📖 <b>Bot Usage Guide</b>

<b>📝 Registration:</b>
/register - Register account and connect to Indodax

<b>💼 Portfolio & Balance:</b>
/portfolio - View complete portfolio
/balance - Check account balance
/orders - View active orders

<b>💹 Trading:</b>
/buy - Buy cryptocurrency
/sell - Sell cryptocurrency
/signal - Get AI signals

<b>⚙️ Settings:</b>
/settings - Account settings
/language - Change language

<b>🔧 Admin (admin only):</b>
/admin - Admin panel
/broadcast - Send broadcast message
/stats - Bot statistics

<b>❓ Need help?</b>
Contact admin for technical support.
                """,
                
                "registration": """
📝 <b>Trading Account Registration</b>

To use trading features, you need to connect your Indodax account:

1️⃣ Login to <a href="https://indodax.com">Indodax.com</a>
2️⃣ Open API Management menu
3️⃣ Create new API Key with permissions:
   • ✅ View (to see balance)
   • ✅ Trade (for trading)
   • ❌ Withdraw (don't enable for security)

4️⃣ Copy your API Key and send it to this chat

⚠️ <b>Security:</b>
• API Key will be encrypted and stored securely
• Bot cannot withdraw your funds
• You can revoke access anytime

<b>Enter your Indodax API Key:</b>
                """,
                
                "trade_selection": """
💹 <b>Select Cryptocurrency to {trade_type}</b>

Choose the trading pair you want to {trade_type}:
                """,
                
                "enter_amount": """
💰 <b>Enter Amount</b>

Enter the amount in IDR you want to {trade_type}:

Example: 100000 (for Rp 100,000)
                """,
                
                "settings_menu": """
⚙️ <b>Account Settings</b>

Manage your trading preferences and account settings:

• 🛑 Stop Loss: Automatic loss limit
• 🎯 Take Profit: Automatic profit target
• 💰 Trade Limit: Maximum amount per trade
• 🔄 Auto Trading: Automatic trading with AI
• 🔔 Notifications: Alert settings
• 🌐 Language: Change interface language
                """
            }
        }
    
    def get_message(self, key: str, language: str = "id", **kwargs) -> str:
        """Get message template by key and language"""
        try:
            message = self.messages[language][key]
            return message.format(**kwargs) if kwargs else message
        except KeyError:
            # Fallback to Indonesian if key not found
            try:
                message = self.messages["id"][key]
                return message.format(**kwargs) if kwargs else message
            except KeyError:
                return "❌ Pesan tidak ditemukan / Message not found"
    
    def get_welcome_message(self, language: str = "id") -> str:
        """Get welcome message"""
        return self.get_message("welcome", language)
    
    def get_help_message(self, language: str = "id") -> str:
        """Get help message"""
        return self.get_message("help", language)
    
    def get_registration_message(self, language: str = "id") -> str:
        """Get registration message"""
        return self.get_message("registration", language)
    
    def get_trade_selection_message(self, trade_type: str, language: str = "id") -> str:
        """Get trade selection message"""
        action_text = "beli" if trade_type == "buy" else "jual"
        if language == "en":
            action_text = trade_type
        
        return self.get_message("trade_selection", language, trade_type=action_text)
    
    def get_settings_message(self, language: str = "id") -> str:
        """Get settings menu message"""
        return self.get_message("settings_menu", language)
    
    def format_portfolio(self, portfolio_data: Dict[str, Any], language: str = "id") -> str:
        """Format portfolio data for display"""
        if not portfolio_data:
            return "❌ Tidak dapat mengambil data portfolio / Cannot retrieve portfolio data"
        
        if language == "en":
            title = "💼 <b>Your Portfolio</b>\n\n"
            total_text = "💰 <b>Total Balance:</b>"
            balance_text = "<b>Asset Balances:</b>"
        else:
            title = "💼 <b>Portfolio Anda</b>\n\n"
            total_text = "💰 <b>Total Saldo:</b>"
            balance_text = "<b>Saldo Aset:</b>"
        
        message = title
        
        # Calculate total value in IDR
        balance = portfolio_data.get('balance', {})
        total_idr = 0
        
        # Add IDR balance
        idr_balance = float(balance.get('idr', 0))
        total_idr += idr_balance
        
        # Asset details
        message += f"{balance_text}\n"
        message += f"💵 IDR: {self._format_idr(idr_balance)}\n"
        
        # Add other crypto balances
        for currency, amount in balance.items():
            if currency != 'idr' and float(amount) > 0:
                message += f"₿ {currency.upper()}: {float(amount):.8f}\n"
        
        message += f"\n{total_text} {self._format_idr(total_idr)}"
        
        return message
    
    def format_balance(self, balance_data: Dict[str, Any], language: str = "id") -> str:
        """Format balance data for display"""
        if not balance_data:
            return "❌ Tidak dapat mengambil data saldo / Cannot retrieve balance data"
        
        if language == "en":
            title = "💰 <b>Account Balance</b>\n\n"
            available_text = "Available"
            locked_text = "Locked"
        else:
            title = "💰 <b>Saldo Akun</b>\n\n"
            available_text = "Tersedia"
            locked_text = "Terkunci"
        
        message = title
        
        # Main balances
        for currency, amount in balance_data.items():
            if float(amount) > 0:
                if currency == 'idr':
                    message += f"💵 IDR: {self._format_idr(float(amount))}\n"
                else:
                    message += f"₿ {currency.upper()}: {float(amount):.8f}\n"
        
        return message
    
    def format_signal(self, signal: AISignal, language: str = "id") -> str:
        """Format AI signal for display"""
        
        if language == "en":
            title = "📊 <b>AI Trading Signal</b>\n\n"
            pair_text = "Pair"
            signal_text = "Signal"
            confidence_text = "Confidence"
            prediction_text = "Price Prediction"
            time_text = "Generated"
            expires_text = "Expires"
        else:
            title = "📊 <b>Sinyal Trading AI</b>\n\n"
            pair_text = "Pasangan"
            signal_text = "Sinyal"
            confidence_text = "Keyakinan"
            prediction_text = "Prediksi Harga"
            time_text = "Dibuat"
            expires_text = "Kadaluarsa"
        
        # Signal emoji
        signal_emoji = {
            "buy": "🟢",
            "sell": "🔴", 
            "hold": "🟡"
        }
        
        confidence_percent = signal.confidence * 100
        
        message = title
        message += f"📈 {pair_text}: <b>{signal.pair_id.upper().replace('_', '/')}</b>\n"
        message += f"{signal_emoji.get(signal.signal_type, '🔷')} {signal_text}: <b>{signal.signal_type.upper()}</b>\n"
        message += f"🎯 {confidence_text}: <b>{confidence_percent:.1f}%</b>\n"
        
        if signal.price_prediction and signal.price_prediction > 0:
            message += f"💰 {prediction_text}: <b>{self._format_price(signal.price_prediction)}</b>\n"
        
        message += f"\n⏰ {time_text}: {signal.created_at.strftime('%d/%m/%Y %H:%M')}\n"
        message += f"⏳ {expires_text}: {signal.expires_at.strftime('%d/%m/%Y %H:%M')}\n"
        
        # Add indicators summary
        if signal.indicators:
            message += f"\n📋 <b>Indikator Teknis:</b>\n"
            indicators = signal.indicators
            
            if 'rsi' in indicators:
                rsi = indicators['rsi']
                rsi_status = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Normal"
                message += f"• RSI: {rsi:.1f} ({rsi_status})\n"
            
            if 'macd' in indicators and 'macd_signal' in indicators:
                macd_trend = "Bullish" if indicators['macd'] > indicators['macd_signal'] else "Bearish"
                message += f"• MACD: {macd_trend}\n"
        
        return message
    
    def format_orders(self, orders_data: Dict[str, Any], language: str = "id") -> str:
        """Format orders data for display"""
        if not orders_data or not orders_data.get('return', {}).get('orders'):
            if language == "en":
                return "📋 <b>Active Orders</b>\n\n❌ No active orders found"
            else:
                return "📋 <b>Order Aktif</b>\n\n❌ Tidak ada order aktif"
        
        if language == "en":
            title = "📋 <b>Active Orders</b>\n\n"
            type_text = "Type"
            amount_text = "Amount"
            price_text = "Price"
        else:
            title = "📋 <b>Order Aktif</b>\n\n"
            type_text = "Tipe"
            amount_text = "Jumlah"
            price_text = "Harga"
        
        message = title
        orders = orders_data.get('return', {}).get('orders', {})
        
        for pair, pair_orders in orders.items():
            if pair_orders:
                message += f"💹 <b>{pair.upper().replace('_', '/')}</b>\n"
                
                for order in pair_orders:
                    order_type = order.get('type', '')
                    order_emoji = "🟢" if order_type == "buy" else "🔴"
                    
                    message += f"{order_emoji} {type_text}: {order_type.upper()}\n"
                    message += f"💰 {amount_text}: {order.get('amount', 0)}\n"
                    message += f"💵 {price_text}: {self._format_price(float(order.get('price', 0)))}\n"
                    message += "─────────────\n"
        
        return message
    
    def format_trade_success(self, trade_type: str, pair_id: str, amount: float, price: float, language: str = "id") -> str:
        """Format successful trade message"""
        
        if language == "en":
            success_text = "✅ <b>Trade Executed Successfully!</b>\n\n"
            type_text = "Type"
            pair_text = "Pair"
            amount_text = "Amount"
            price_text = "Price"
            total_text = "Total"
        else:
            success_text = "✅ <b>Trade Berhasil Dieksekusi!</b>\n\n"
            type_text = "Tipe"
            pair_text = "Pasangan"
            amount_text = "Jumlah"
            price_text = "Harga"
            total_text = "Total"
        
        trade_emoji = "🟢" if trade_type == "buy" else "🔴"
        total_value = amount * price
        
        message = success_text
        message += f"{trade_emoji} {type_text}: <b>{trade_type.upper()}</b>\n"
        message += f"📈 {pair_text}: <b>{pair_id.upper().replace('_', '/')}</b>\n"
        message += f"💰 {amount_text}: <b>{amount:.8f}</b>\n"
        message += f"💵 {price_text}: <b>{self._format_price(price)}</b>\n"
        message += f"💸 {total_text}: <b>{self._format_idr(total_value)}</b>\n"
        
        return message
    
    def format_admin_stats(self, stats: Dict[str, Any]) -> str:
        """Format admin statistics"""
        message = "📊 <b>Bot Statistics</b>\n\n"
        message += f"👥 Total Users: <b>{stats.get('total_users', 0)}</b>\n"
        message += f"✅ Registered Users: <b>{stats.get('registered_users', 0)}</b>\n"
        message += f"⭐ Premium Users: <b>{stats.get('premium_users', 0)}</b>\n"
        message += f"💹 Total Trades: <b>{stats.get('total_trades', 0)}</b>\n"
        
        return message
    
    def _format_idr(self, amount: float) -> str:
        """Format IDR amount with thousand separators"""
        return f"Rp {amount:,.0f}".replace(',', '.')
    
    def _format_price(self, price: float) -> str:
        """Format price with appropriate decimal places"""
        if price >= 1000:
            return f"{price:,.0f}".replace(',', '.')
        elif price >= 1:
            return f"{price:,.2f}".replace(',', '.')
        else:
            return f"{price:.8f}"
