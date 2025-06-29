"""
Message templates for the Telegram bot in multiple languages
"""
from datetime import datetime
from typing import Dict, Any, List
from core.database import AISignal, Trade
from bot.utils import format_currency
import structlog

logger = structlog.get_logger(__name__)

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
                """,
                
                "trading_menu": """
💹 <b>Menu Trading</b>

Pilih aksi trading Anda:

• 🟢 <b>Beli</b> - Beli cryptocurrency
• 🔴 <b>Jual</b> - Jual kepemilikan Anda
• 📊 <b>Pasar</b> - Lihat data pasar
• 📈 <b>Chart</b> - Analisis teknikal

<i>Selalu trading dengan bijak dan jangan investasi lebih dari yang mampu Anda tanggung.</i>
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
                """,
                
                "trading_menu": """
💹 <b>Trading Menu</b>

Choose your trading action:

• 🟢 <b>Buy</b> - Purchase cryptocurrency
• 🔴 <b>Sell</b> - Sell your holdings
• 📊 <b>Market</b> - View market data
• 📈 <b>Charts</b> - Technical analysis

<i>Always trade responsibly and never invest more than you can afford to lose.</i>
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
    
    def get_trading_menu_message(self, language: str = "id") -> str:
        """Get trading menu message"""
        if language == "en":
            return """
💹 <b>Trading Menu</b>

Choose your trading action:

• 🟢 <b>Buy</b> - Purchase cryptocurrency
• 🔴 <b>Sell</b> - Sell your holdings
• 📊 <b>Market</b> - View market data
• 📈 <b>Charts</b> - Technical analysis

<i>Always trade responsibly and never invest more than you can afford to lose.</i>
            """
        else:
            return """
💹 <b>Menu Trading</b>

Pilih aksi trading Anda:

• 🟢 <b>Beli</b> - Beli cryptocurrency
• 🔴 <b>Jual</b> - Jual kepemilikan Anda
• 📊 <b>Pasar</b> - Lihat data pasar
• 📈 <b>Chart</b> - Analisis teknikal

<i>Selalu trading dengan bijak dan jangan investasi lebih dari yang mampu Anda tanggung.</i>
            """
    
    def format_portfolio(self, portfolio_data: Dict[str, Any], language: str = "id") -> str:
        """Format portfolio data for display"""
        if "error" in portfolio_data:
            return f"❌ Error: {portfolio_data['error']}"
        
        if language == "en":
            text = "💼 <b>Your Portfolio</b>\n\n"
        else:
            text = "💼 <b>Portfolio Anda</b>\n\n"
        
        if portfolio_data.get("assets"):
            total_value = portfolio_data.get("total_idr_value", 0)
            text += f"💰 Total Value: {format_currency(total_value)} IDR\n\n"
            
            for asset in portfolio_data["assets"]:
                currency = asset["currency"]
                balance = asset["balance"]
                idr_value = asset["idr_value"]
                
                if currency == "IDR":
                    text += f"💵 {currency}: {format_currency(balance)}\n"
                else:
                    current_price = asset.get("current_price", 0)
                    text += f"🪙 {currency}: {balance:.8f}\n"
                    text += f"   💰 Value: {format_currency(idr_value)} IDR\n"
                    if current_price > 0:
                        text += f"   📊 Price: {format_currency(current_price)} IDR\n"
                text += "\n"
        else:
            if language == "en":
                text += "📭 No assets found in your portfolio."
            else:
                text += "📭 Tidak ada aset ditemukan di portfolio Anda."
        
        return text
    
    def format_balance(self, balance_data: Dict[str, Any], language: str = "id") -> str:
        """Format balance data for display"""
        if "error" in balance_data:
            return f"❌ Error: {balance_data['error']}"
        
        if language == "en":
            text = "💰 <b>Account Balance</b>\n\n"
        else:
            text = "💰 <b>Saldo Akun</b>\n\n"
        
        # Check if balance_data is directly the balance dict
        if isinstance(balance_data, dict):
            has_balance = False
            
            for currency, balance in balance_data.items():
                try:
                    balance_float = float(balance)
                    if balance_float > 0:
                        has_balance = True
                        if currency.lower() == "idr":
                            text += f"💵 {currency.upper()}: {format_currency(balance_float)}\n"
                        else:
                            text += f"🪙 {currency.upper()}: {balance_float:.8f}\n"
                except (ValueError, TypeError):
                    continue
            
            if not has_balance:
                if language == "en":
                    text += "📭 No balance available or all balances are zero."
                else:
                    text += "📭 Tidak ada saldo tersedia atau semua saldo kosong."
        else:
            if language == "en":
                text += "📭 No balance data available."
            else:
                text += "📭 Data saldo tidak tersedia."
        
        return text
    
    def format_orders(self, orders_data: Dict[str, Any], language: str = "id") -> str:
        """Format orders data for display"""
        if "error" in orders_data:
            return f"❌ Error: {orders_data['error']}"
        
        if language == "en":
            text = "📋 <b>Your Open Orders</b>\n\n"
        else:
            text = "📋 <b>Order Terbuka Anda</b>\n\n"
        
        try:
            # Check if we have orders data
            if "return" in orders_data and "orders" in orders_data["return"]:
                orders = orders_data["return"]["orders"]
                
                # orders is a dict with pair_id as key and list of orders as value
                if isinstance(orders, dict) and orders:
                    order_count = 0
                    for pair_id, pair_orders in orders.items():
                        if isinstance(pair_orders, list):
                            for order in pair_orders:
                                order_type = order.get("type", "")
                                price = float(order.get("price", 0))
                                order_id = order.get("order_id", "")
                                
                                # Get remaining amount based on order type
                                if order_type == "buy":
                                    remain_amount = float(order.get("remain_idr", 0))
                                    amount_text = f"{format_currency(remain_amount)} IDR"
                                else:  # sell
                                    # Try different field names for remaining amount
                                    remain_amount = 0
                                    for field in ["remain_btc", "remain_eth", f"remain_{pair_id.split('_')[0]}"]:
                                        if field in order:
                                            remain_amount = float(order.get(field, 0))
                                            break
                                    amount_text = f"{remain_amount:.8f} {pair_id.split('_')[0].upper()}"
                                
                                text += f"📌 #{order_id} - {order_type.upper()} {pair_id.upper()}\n"
                                text += f"   💰 Price: {format_currency(price)} IDR\n"
                                text += f"   📊 Amount: {amount_text}\n\n"
                                order_count += 1
                    
                    if order_count == 0:
                        if language == "en":
                            text += "📭 No open orders."
                        else:
                            text += "📭 Tidak ada order terbuka."
                else:
                    if language == "en":
                        text += "📭 No open orders."
                    else:
                        text += "📭 Tidak ada order terbuka."
            else:
                if language == "en":
                    text += "📭 No orders data available."
                else:
                    text += "📭 Data order tidak tersedia."
        except Exception as e:
            logger.error("Error formatting orders", error=str(e))
            if language == "en":
                text += "❌ Error formatting orders data."
            else:
                text += "❌ Error memformat data order."
        
        return text
    
    def format_signal(self, signal: Any, language: str = "id") -> str:
        """Format AI signal for display"""
        if not signal:
            if language == "en":
                return "❌ No signal available"
            else:
                return "❌ Tidak ada sinyal tersedia"
        
        if language == "en":
            text = f"📊 <b>AI Trading Signal</b>\n\n"
        else:
            text = f"📊 <b>Sinyal Trading AI</b>\n\n"
        
        try:
            pair_id = getattr(signal, 'pair_id', 'Unknown')
            signal_type = getattr(signal, 'signal_type', 'HOLD')
            confidence = getattr(signal, 'confidence', 0.0)
            price_prediction = getattr(signal, 'price_prediction', 0.0)
            
            text += f"🪙 Pair: {pair_id.upper()}\n"
            text += f"📈 Signal: {signal_type.upper()}\n"
            text += f"🎯 Confidence: {confidence:.1%}\n"
            
            if price_prediction > 0:
                text += f"💰 Price Target: {format_currency(price_prediction)} IDR\n"
            
            if signal_type.upper() == "BUY":
                text += "\n🟢 <b>Recommendation: BUY</b>"
            elif signal_type.upper() == "SELL":
                text += "\n🔴 <b>Recommendation: SELL</b>"
            else:
                text += "\n⚪ <b>Recommendation: HOLD</b>"
                
        except Exception as e:
            logger.error("Error formatting signal", error=str(e))
            text += "❌ Error formatting signal data"
        
        return text
    
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
