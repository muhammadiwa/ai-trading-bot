import requests
import json
from typing import Dict, Optional
from twilio.rest import Client
import asyncio
from telegram import Bot
from datetime import datetime

class NotificationManager:
    """
    Kelas untuk mengirim notifikasi ke WhatsApp dan Telegram
    """
    
    def __init__(self, telegram_token: str = None, telegram_chat_id: str = None,
                 twilio_sid: str = None, twilio_token: str = None,
                 whatsapp_from: str = None, whatsapp_to: str = None):
        
        # Telegram configuration
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.telegram_bot = None
        
        if telegram_token:
            self.telegram_bot = Bot(token=telegram_token)
        
        # WhatsApp/Twilio configuration
        self.twilio_client = None
        self.whatsapp_from = whatsapp_from
        self.whatsapp_to = whatsapp_to
        
        if twilio_sid and twilio_token:
            self.twilio_client = Client(twilio_sid, twilio_token)
    
    async def send_telegram_message(self, message: str) -> bool:
        """
        Mengirim pesan ke Telegram
        """
        try:
            if self.telegram_bot and self.telegram_chat_id:
                await self.telegram_bot.send_message(
                    chat_id=self.telegram_chat_id,
                    text=message,
                    parse_mode='HTML'
                )
                print("Telegram message sent successfully")
                return True
            else:
                print("Telegram configuration not complete")
                return False
        except Exception as e:
            print(f"Error sending Telegram message: {e}")
            return False
    
    def send_whatsapp_message(self, message: str) -> bool:
        """
        Mengirim pesan ke WhatsApp via Twilio
        """
        try:
            if self.twilio_client and self.whatsapp_from and self.whatsapp_to:
                message_obj = self.twilio_client.messages.create(
                    body=message,
                    from_=self.whatsapp_from,
                    to=self.whatsapp_to
                )
                print(f"WhatsApp message sent successfully: {message_obj.sid}")
                return True
            else:
                print("WhatsApp configuration not complete")
                return False
        except Exception as e:
            print(f"Error sending WhatsApp message: {e}")
            return False
    
    def format_trading_signal(self, signal_data: Dict) -> str:
        """
        Format pesan untuk sinyal trading
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"""
🤖 <b>AI Trading Bot Signal</b>
📅 Time: {timestamp}
💰 Pair: {signal_data.get('pair', 'Unknown')}
💹 Current Price: Rp {signal_data.get('current_price', 0):,.0f}

📊 <b>Technical Analysis:</b>
• RSI: {signal_data.get('rsi_signal', 'N/A')} ({signal_data.get('rsi_value', 0):.2f})
• MACD: {signal_data.get('macd_signal', 'N/A')}
• SMA: {signal_data.get('sma_signal', 'N/A')}
• Bollinger: {signal_data.get('bb_signal', 'N/A')}

🎯 <b>AI Prediction:</b>
• Next Price: Rp {signal_data.get('predicted_price', 0):,.0f}
• Direction: {signal_data.get('predicted_direction', 'N/A')}
• Confidence: {signal_data.get('confidence', 0):.1f}%

📈 <b>Trading Signal: {signal_data.get('overall_signal', 'HOLD')}</b>

⚠️ <b>Risk Management:</b>
• Stop Loss: {signal_data.get('stop_loss', 'N/A')}%
• Take Profit: {signal_data.get('take_profit', 'N/A')}%
"""
        return message
    
    def format_trade_execution(self, trade_data: Dict) -> str:
        """
        Format pesan untuk eksekusi trading
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"""
🚀 <b>Trade Executed</b>
📅 Time: {timestamp}
💰 Pair: {trade_data.get('pair', 'Unknown')}

📋 <b>Order Details:</b>
• Type: {trade_data.get('order_type', 'Unknown')}
• Amount: {trade_data.get('amount', 0):.6f}
• Price: Rp {trade_data.get('price', 0):,.0f}
• Total: Rp {trade_data.get('total', 0):,.0f}

📊 <b>Order Status:</b>
• Status: {trade_data.get('status', 'Unknown')}
• Order ID: {trade_data.get('order_id', 'N/A')}

💡 <b>Reason:</b> {trade_data.get('reason', 'N/A')}
"""
        return message
    
    def format_portfolio_update(self, portfolio_data: Dict) -> str:
        """
        Format pesan untuk update portfolio
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"""
💼 <b>Portfolio Update</b>
📅 Time: {timestamp}

💰 <b>Balance:</b>
• IDR: Rp {portfolio_data.get('idr_balance', 0):,.0f}
• BTC: {portfolio_data.get('btc_balance', 0):.8f}
• ETH: {portfolio_data.get('eth_balance', 0):.6f}

📈 <b>Performance:</b>
• Total Value: Rp {portfolio_data.get('total_value', 0):,.0f}
• Daily P&L: Rp {portfolio_data.get('daily_pnl', 0):,.0f} ({portfolio_data.get('daily_pnl_percent', 0):.2f}%)
• Total P&L: Rp {portfolio_data.get('total_pnl', 0):,.0f} ({portfolio_data.get('total_pnl_percent', 0):.2f}%)

📊 <b>Today's Stats:</b>
• Trades: {portfolio_data.get('trades_today', 0)}
• Win Rate: {portfolio_data.get('win_rate', 0):.1f}%
"""
        return message
    
    def format_alert_message(self, alert_data: Dict) -> str:
        """
        Format pesan untuk alert/warning
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        alert_type = alert_data.get('type', 'INFO')
        emoji = "⚠️" if alert_type == "WARNING" else "🚨" if alert_type == "ERROR" else "ℹ️"
        
        message = f"""
{emoji} <b>Alert: {alert_type}</b>
📅 Time: {timestamp}

📋 <b>Message:</b>
{alert_data.get('message', 'No message provided')}

💰 Pair: {alert_data.get('pair', 'N/A')}
💹 Current Price: Rp {alert_data.get('current_price', 0):,.0f}
"""
        return message
    
    async def send_notification(self, message: str, notification_type: str = "both") -> Dict[str, bool]:
        """
        Mengirim notifikasi ke platform yang dipilih
        """
        results = {
            'telegram': False,
            'whatsapp': False
        }
        
        if notification_type in ["telegram", "both"]:
            results['telegram'] = await self.send_telegram_message(message)
        
        if notification_type in ["whatsapp", "both"]:
            results['whatsapp'] = self.send_whatsapp_message(message)
        
        return results
    
    async def send_trading_signal_notification(self, signal_data: Dict) -> Dict[str, bool]:
        """
        Mengirim notifikasi sinyal trading
        """
        message = self.format_trading_signal(signal_data)
        return await self.send_notification(message)
    
    async def send_trade_execution_notification(self, trade_data: Dict) -> Dict[str, bool]:
        """
        Mengirim notifikasi eksekusi trading
        """
        message = self.format_trade_execution(trade_data)
        return await self.send_notification(message)
    
    async def send_portfolio_update_notification(self, portfolio_data: Dict) -> Dict[str, bool]:
        """
        Mengirim notifikasi update portfolio
        """
        message = self.format_portfolio_update(portfolio_data)
        return await self.send_notification(message)
    
    async def send_alert_notification(self, alert_data: Dict) -> Dict[str, bool]:
        """
        Mengirim notifikasi alert
        """
        message = self.format_alert_message(alert_data)
        return await self.send_notification(message)
    
    def test_notifications(self) -> Dict[str, bool]:
        """
        Test notifikasi untuk memastikan konfigurasi benar
        """
        test_message = """
🧪 <b>Test Notification</b>
✅ AI Trading Bot is working properly!
📅 Time: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        results = {}
        
        # Test Telegram
        if self.telegram_bot and self.telegram_chat_id:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results['telegram'] = loop.run_until_complete(
                    self.send_telegram_message(test_message)
                )
                loop.close()
            except Exception as e:
                print(f"Error testing Telegram: {e}")
                results['telegram'] = False
        else:
            results['telegram'] = False
        
        # Test WhatsApp
        results['whatsapp'] = self.send_whatsapp_message(test_message.replace('<b>', '*').replace('</b>', '*'))
        
        return results
