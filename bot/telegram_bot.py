"""
Telegram bot interface for the AI Trading Bot
"""
import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

# Import untuk backtesting visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton, FSInputFile
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.enums import ParseMode
import structlog

from config.settings import settings
from core.database import get_db, User, Trade, Portfolio, AISignal, UserSettings
from core.indodax_api import IndodaxAPI
from ai.signal_generator import SignalGenerator
from core.backtester import Backtester
from bot.enhanced_features import EnhancedTradingFeatures
from bot.keyboards import create_main_keyboard, create_trading_keyboard, create_settings_keyboard
from bot.messages import Messages
from bot.utils import format_currency, is_admin_user, encrypt_api_key, decrypt_api_key

logger = structlog.get_logger(__name__)

# FSM States
class RegistrationStates(StatesGroup):
    waiting_for_api_key = State()
    waiting_for_secret_key = State()
    confirming_registration = State()

class TradingStates(StatesGroup):
    selecting_pair = State()
    entering_amount = State()
    confirming_trade = State()

class SettingsStates(StatesGroup):
    editing_stop_loss = State()
    editing_take_profit = State()
    editing_max_trade_amount = State()

class TelegramBot:
    """Main Telegram bot class"""
    
    def __init__(self):
        self.bot = Bot(token=settings.telegram_bot_token, parse_mode=ParseMode.HTML)
        self.dp = Dispatcher(storage=MemoryStorage())
        self.router = Router()
        self.signal_generator = SignalGenerator()
        self.backtester = Backtester()
        self.enhanced_features = EnhancedTradingFeatures(self, self.signal_generator)
        self.messages = Messages()
        
        # Register handlers
        self._register_handlers()
        
        # Include router in dispatcher
        self.dp.include_router(self.router)
    
    def _register_handlers(self):
        """Register all bot command and callback handlers"""
        
        # Basic commands
        self.router.message(Command("start"))(self.cmd_start)
        self.router.message(Command("help"))(self.cmd_help)
        self.router.message(Command("daftar", "register"))(self.cmd_register)
        
        # Trading commands
        self.router.message(Command("portfolio"))(self.cmd_portfolio)
        self.router.message(Command("balance", "saldo"))(self.cmd_balance)
        self.router.message(Command("buy", "beli"))(self.cmd_buy)
        self.router.message(Command("sell", "jual"))(self.cmd_sell)
        self.router.message(Command("signal"))(self.cmd_signal)
        self.router.message(Command("orders"))(self.cmd_orders)
        
        # Settings commands
        self.router.message(Command("settings", "pengaturan"))(self.cmd_settings)
        self.router.message(Command("language", "bahasa"))(self.cmd_language)
        
        # Admin commands
        self.router.message(Command("admin"))(self.cmd_admin)
        self.router.message(Command("broadcast"))(self.cmd_broadcast)
        self.router.message(Command("stats"))(self.cmd_stats)
        
        # New AI and advanced features commands
        self.router.message(Command("ask"))(self.cmd_ask)
        self.router.message(Command("autotrading"))(self.cmd_autotrading)
        self.router.message(Command("dca"))(self.cmd_dca)
        self.router.message(Command("backtest"))(self.cmd_backtest)
        
        # Callback query handlers
        self.router.callback_query(F.data.startswith("trade_"))(self.callback_trade)
        self.router.callback_query(F.data.startswith("signal_"))(self.callback_signal)
        self.router.callback_query(F.data.startswith("backtest_"))(self.callback_backtest)
        self.router.callback_query(F.data.startswith("settings_"))(self.callback_settings)
        self.router.callback_query(F.data.startswith("confirm_"))(self.callback_confirm)
        self.router.callback_query(F.data.startswith("lang_"))(self.callback_language)
        self.router.callback_query(F.data.startswith("admin_"))(self.callback_admin)
        
        # Main menu callback handlers
        self.router.callback_query(F.data == "portfolio")(self.callback_portfolio)
        self.router.callback_query(F.data == "balance")(self.callback_balance)
        self.router.callback_query(F.data == "signals")(self.callback_signals)
        self.router.callback_query(F.data == "trading")(self.callback_trading)
        self.router.callback_query(F.data == "orders")(self.callback_orders)
        self.router.callback_query(F.data == "settings")(self.callback_settings_menu)
        self.router.callback_query(F.data == "help")(self.callback_help)
        
        # FSM handlers
        self.router.message(RegistrationStates.waiting_for_api_key)(self.process_api_key)
        self.router.message(RegistrationStates.waiting_for_secret_key)(self.process_secret_key)
        self.router.message(TradingStates.entering_amount)(self.process_trade_amount)
        self.router.message(SettingsStates.editing_stop_loss)(self.process_stop_loss)
        self.router.message(SettingsStates.editing_take_profit)(self.process_take_profit)
        self.router.message(SettingsStates.editing_max_trade_amount)(self.process_max_trade_amount)
        
        # Amount selection callback handlers
        self.router.callback_query(F.data.startswith("amount_"))(self.callback_amount_selection)
        
    async def start(self):
        """Start the Telegram bot"""
        try:
            logger.info("Starting Telegram bot...")
            
            # Set bot commands
            await self._set_bot_commands()
            
            # Start polling
            await self.dp.start_polling(self.bot)
            
        except Exception as e:
            logger.error("Failed to start Telegram bot", error=str(e))
            raise
    
    async def stop(self):
        """Stop the Telegram bot"""
        try:
            await self.bot.session.close()
            logger.info("Telegram bot stopped")
        except Exception as e:
            logger.error("Failed to stop Telegram bot", error=str(e))
    
    async def _set_bot_commands(self):
        """Set bot commands menu"""
        from aiogram.types import BotCommand
        
        commands = [
            BotCommand(command="start", description="🚀 Mulai menggunakan bot"),
            BotCommand(command="help", description="❓ Bantuan dan panduan"),
            BotCommand(command="daftar", description="📝 Daftar akun trading"),
            BotCommand(command="portfolio", description="💼 Lihat portfolio"),
            BotCommand(command="balance", description="💰 Cek saldo"),
            BotCommand(command="signal", description="📊 Dapatkan sinyal AI"),
            BotCommand(command="buy", description="🟢 Beli cryptocurrency"),
            BotCommand(command="sell", description="🔴 Jual cryptocurrency"),
            BotCommand(command="orders", description="📋 Lihat order aktif"),
            BotCommand(command="settings", description="⚙️ Pengaturan akun"),
            BotCommand(command="ask", description="🤖 AI assistant"),
            BotCommand(command="autotrading", description="🔄 Auto trading"),
            BotCommand(command="dca", description="💰 Dollar Cost Averaging"),
            BotCommand(command="backtest", description="📊 Backtest strategi"),
        ]
        
        await self.bot.set_my_commands(commands)
    
    async def _get_or_create_user(self, telegram_user) -> User:
        """Get existing user or create new one"""
        try:
            db = get_db()
            try:
                # Try to get existing user
                user = db.query(User).filter(User.telegram_id == telegram_user.id).first()
                
                if user:
                    # Update user info if changed
                    if (getattr(user, 'username') != telegram_user.username or 
                        getattr(user, 'first_name') != telegram_user.first_name or
                        getattr(user, 'last_name') != telegram_user.last_name):
                        
                        setattr(user, 'username', telegram_user.username)
                        setattr(user, 'first_name', telegram_user.first_name)
                        setattr(user, 'last_name', telegram_user.last_name)
                        setattr(user, 'updated_at', datetime.utcnow())
                        db.commit()
                    
                    return user
                else:
                    # Create new user
                    new_user = User(
                        telegram_id=telegram_user.id,
                        username=telegram_user.username,
                        first_name=telegram_user.first_name,
                        last_name=telegram_user.last_name,
                        language="id",  # Default to Indonesian
                        is_active=True,
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                    
                    db.add(new_user)
                    db.commit()
                    db.refresh(new_user)
                    
                    logger.info("New user created", telegram_id=telegram_user.id, username=telegram_user.username)
                    return new_user
                    
            finally:
                db.close()
                
        except Exception as e:
            logger.error("Failed to get or create user", telegram_id=telegram_user.id, error=str(e))
            raise
    
    # Command Handlers
    
    async def cmd_start(self, message: Message):
        """Handle /start command"""
        try:
            user = await self._get_or_create_user(message.from_user)
            
            welcome_text = self.messages.get_welcome_message(getattr(user, 'language', 'id'))
            keyboard = create_main_keyboard(getattr(user, 'language', 'id'))
            
            await message.answer(welcome_text, reply_markup=keyboard)
            
        except Exception as e:
            logger.error("Failed to handle start command", error=str(e))
            await message.answer("❌ Terjadi kesalahan. Silakan coba lagi.")
    
    async def cmd_help(self, message: Message):
        """Handle /help command"""
        try:
            user = await self._get_or_create_user(message.from_user)
            help_text = self.messages.get_help_message(getattr(user, 'language', 'id'))
            
            await message.answer(help_text)
            
        except Exception as e:
            logger.error("Failed to handle help command", error=str(e))
            await message.answer("❌ Terjadi kesalahan. Silakan coba lagi.")
    
    async def cmd_register(self, message: Message, state: FSMContext):
        """Handle /register command"""
        try:
            user = await self._get_or_create_user(message.from_user)
            
            if getattr(user, 'indodax_api_key', None):
                await message.answer("✅ Anda sudah terdaftar dan terkoneksi dengan Indodax!")
                return
            
            register_text = self.messages.get_registration_message(getattr(user, 'language', 'id'))
            await message.answer(register_text)
            
            await state.set_state(RegistrationStates.waiting_for_api_key)
            
        except Exception as e:
            logger.error("Failed to handle register command", error=str(e))
            await message.answer("❌ Terjadi kesalahan. Silakan coba lagi.")
    
    async def cmd_portfolio(self, message: Message):
        """Handle /portfolio command"""
        try:
            user = await self._get_or_create_user(message.from_user)
            
            if not getattr(user, 'indodax_api_key', None):
                await message.answer("❌ Anda belum terdaftar. Gunakan /daftar untuk mendaftar.")
                return
            
            # Get portfolio data
            portfolio_data = await self._get_portfolio_data(user)
            portfolio_text = self.messages.format_portfolio(portfolio_data, getattr(user, 'language', 'id'))
            
            await message.answer(portfolio_text)
            
        except Exception as e:
            logger.error("Failed to handle portfolio command", error=str(e))
            await message.answer("❌ Terjadi kesalahan saat mengambil data portfolio.")
    
    async def cmd_balance(self, message: Message):
        """Handle /balance command"""
        try:
            user = await self._get_or_create_user(message.from_user)
            
            if not getattr(user, 'indodax_api_key', None):
                await message.answer("❌ Anda belum terdaftar. Gunakan /daftar untuk mendaftar.")
                return
            
            # Get balance from Indodax
            api_key = decrypt_api_key(str(getattr(user, 'indodax_api_key', '')))
            secret_key = decrypt_api_key(str(getattr(user, 'indodax_secret_key', '')))
            
            user_api = IndodaxAPI(api_key, secret_key)
            balance_data = await user_api.get_balance()
            
            logger.info("Balance data received", balance_data=balance_data, data_type=type(balance_data))
            
            balance_text = self.messages.format_balance(balance_data, getattr(user, 'language', 'id'))
            await message.answer(balance_text)
            
        except Exception as e:
            logger.error("Failed to handle balance command", error=str(e))
            await message.answer("❌ Terjadi kesalahan saat mengambil data saldo.")
    
    async def cmd_signal(self, message: Message):
        """Handle /signal command with enhanced AI analysis"""
        try:
            user = await self._get_or_create_user(message.from_user)
            
            # Parse command for specific pair
            if not message.text:
                await message.answer("❌ Pesan tidak valid.")
                return
                
            command_parts = message.text.split()
            pair_id = None
            
            if len(command_parts) > 1:
                requested_pair = command_parts[1].lower()
                pair_id = f"{requested_pair}_idr"
            
            # Use enhanced signal generation
            await self.enhanced_features.enhanced_signal_generation(message, pair_id)
            
        except Exception as e:
            logger.error("Failed to handle signal command", error=str(e))
            await message.answer("❌ Terjadi kesalahan saat menghasilkan sinyal.")
    
    async def cmd_buy(self, message: Message, state: FSMContext):
        """Handle /buy command"""
        await self._handle_trade_command(message, state, "buy")
    
    async def cmd_sell(self, message: Message, state: FSMContext):
        """Handle /sell command"""
        await self._handle_trade_command(message, state, "sell")
    
    async def _handle_trade_command(self, message: Message, state: FSMContext, trade_type: str):
        """Handle buy/sell commands"""
        try:
            user = await self._get_or_create_user(message.from_user)
            
            if not getattr(user, 'indodax_api_key', None):
                await message.answer("❌ Anda belum terdaftar. Gunakan /daftar untuk mendaftar.")
                return
            
            # Check if message.text is available
            if not message.text:
                await message.answer("❌ Pesan tidak valid.")
                return
            
            # Parse command arguments
            command_parts = message.text.split()
            
            if len(command_parts) >= 3:
                # Direct trade: /buy BTC 100000
                pair_symbol = command_parts[1].upper()
                amount = float(command_parts[2])
                
                pair_id = f"{pair_symbol.lower()}_idr"
                await self._execute_trade(message, user, pair_id, trade_type, amount)
            else:
                # Interactive trade selection
                keyboard = create_trading_keyboard(trade_type, getattr(user, 'language', 'id'))
                trade_text = self.messages.get_trade_selection_message(trade_type, getattr(user, 'language', 'id'))
                
                await message.answer(trade_text, reply_markup=keyboard)
                await state.set_state(TradingStates.selecting_pair)
                await state.update_data(trade_type=trade_type)
            
        except ValueError:
            await message.answer("❌ Format salah. Gunakan: /buy SYMBOL AMOUNT")
        except Exception as e:
            logger.error("Failed to handle trade command", trade_type=trade_type, error=str(e))
            await message.answer("❌ Terjadi kesalahan saat memproses order.")
    
    async def cmd_orders(self, message: Message):
        """Handle /orders command"""
        try:
            user = await self._get_or_create_user(message.from_user)
            
            if not getattr(user, 'indodax_api_key', None):
                await message.answer("❌ Anda belum terdaftar. Gunakan /daftar untuk mendaftar.")
                return
            
            # Get open orders
            api_key = decrypt_api_key(str(getattr(user, 'indodax_api_key', '')))
            secret_key = decrypt_api_key(str(getattr(user, 'indodax_secret_key', '')))
            
            user_api = IndodaxAPI(api_key, secret_key)
            orders_data = await user_api.get_open_orders()
            
            orders_text = self.messages.format_orders(orders_data, getattr(user, 'language', 'id'))
            await message.answer(orders_text)
            
        except Exception as e:
            logger.error("Failed to handle orders command", error=str(e))
            await message.answer("❌ Terjadi kesalahan saat mengambil data order.")
    
    async def cmd_settings(self, message: Message):
        """Handle /settings command"""
        try:
            user = await self._get_or_create_user(message.from_user)
            settings_keyboard = create_settings_keyboard(getattr(user, 'language', 'id'))
            settings_text = self.messages.get_settings_message(getattr(user, 'language', 'id'))
            
            await message.answer(settings_text, reply_markup=settings_keyboard)
            
        except Exception as e:
            logger.error("Failed to handle settings command", error=str(e))
            await message.answer("❌ Terjadi kesalahan.")
    
    async def cmd_language(self, message: Message):
        """Handle /language command"""
        try:
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="🇮🇩 Bahasa Indonesia", callback_data="lang_id"),
                    InlineKeyboardButton(text="🇺🇸 English", callback_data="lang_en")
                ]
            ])
            
            await message.answer("🌐 Pilih bahasa / Choose language:", reply_markup=keyboard)
            
        except Exception as e:
            logger.error("Failed to handle language command", error=str(e))
    
    # Admin Commands
    
    async def cmd_admin(self, message: Message):
        """Handle /admin command"""
        if not message.from_user or not is_admin_user(message.from_user.id):
            await message.answer("❌ Anda tidak memiliki akses admin.")
            return
        
        admin_text = """
🔧 <b>Panel Admin</b>

📊 /stats - Statistik bot
📢 /broadcast - Kirim pesan ke semua user
👥 /users - Daftar user
💹 /trading_stats - Statistik trading
🛠️ /maintenance - Mode maintenance
        """
        
        await message.answer(admin_text)
    
    async def cmd_broadcast(self, message: Message):
        """Handle /broadcast command"""
        if not message.from_user or not is_admin_user(message.from_user.id):
            return
        
        if not message.text:
            await message.answer("❌ Pesan tidak valid.")
            return
        
        command_parts = message.text.split(" ", 1)
        if len(command_parts) < 2:
            await message.answer("Format: /broadcast <pesan>")
            return
        
        broadcast_message = command_parts[1]
        await self._broadcast_message(broadcast_message)
    
    async def cmd_stats(self, message: Message):
        """Handle /stats command"""
        if not message.from_user or not is_admin_user(message.from_user.id):
            return
        
        try:
            stats = await self._get_bot_statistics()
            stats_text = self.messages.format_admin_stats(stats)
            
            await message.answer(stats_text)
            
        except Exception as e:
            logger.error("Failed to get bot statistics", error=str(e))
            await message.answer("❌ Gagal mengambil statistik.")
    
    # FSM Handlers
    
    async def process_api_key(self, message: Message, state: FSMContext):
        """Process API key input during registration"""
        try:
            if not message.text:
                await message.answer("❌ Pesan tidak valid. Silakan masukkan API key yang benar.")
                return
                
            api_key = message.text.strip()
            
            # Basic validation
            if len(api_key) < 20:
                await message.answer("❌ API key tidak valid. Silakan masukkan API key yang benar.")
                return
            
            await state.update_data(api_key=api_key)
            await state.set_state(RegistrationStates.waiting_for_secret_key)
            
            await message.answer("🔐 Sekarang masukkan Secret Key Indodax Anda:")
            
        except Exception as e:
            logger.error("Failed to process API key", error=str(e))
            await message.answer("❌ Terjadi kesalahan. Silakan coba lagi.")
    
    async def process_secret_key(self, message: Message, state: FSMContext):
        """Process secret key input during registration"""
        try:
            if not message.text:
                await message.answer("❌ Pesan tidak valid. Silakan masukkan secret key yang benar.")
                return
                
            secret_key = message.text.strip()
            
            # Basic validation
            if len(secret_key) < 20:
                await message.answer("❌ Secret key tidak valid. Silakan masukkan secret key yang benar.")
                return
            
            data = await state.get_data()
            api_key = data.get('api_key')
            
            # Show testing message
            await message.answer("🔄 Memvalidasi API credentials...")
            
            # Test API credentials
            test_api = IndodaxAPI(api_key, secret_key)
            try:
                # Call a simple API to test credentials
                result = await test_api.get_info()
                
                if result and result.get("success") == 1:
                    # Save encrypted credentials
                    user = await self._get_or_create_user(message.from_user)
                    
                    # Ensure api_key and secret_key are not None
                    if api_key and secret_key:
                        setattr(user, 'indodax_api_key', encrypt_api_key(api_key))
                        setattr(user, 'indodax_secret_key', encrypt_api_key(secret_key))
                        
                        db = get_db()
                        try:
                            db.merge(user)
                            db.commit()
                        finally:
                            db.close()
                        
                        await state.clear()
                        await message.answer("✅ Pendaftaran berhasil! Akun Anda telah terhubung dengan Indodax.")
                    else:
                        await message.answer("❌ API key atau secret key tidak valid.")
                        await state.clear()
                else:
                    error_msg = result.get("error", "Unknown error") if result else "API call failed"
                    logger.error("API validation failed", error=error_msg)
                    await message.answer(f"❌ API credentials tidak valid: {error_msg}")
                    await state.clear()
                
            except Exception as e:
                logger.error("Failed to validate API credentials", error=str(e))
                await message.answer(f"❌ Gagal memvalidasi API credentials: {str(e)}")
                await state.clear()
            
        except Exception as e:
            logger.error("Failed to process secret key", error=str(e))
            await message.answer("❌ Terjadi kesalahan. Silakan coba lagi.")
            await state.clear()
    
    # Callback Handlers
    
    async def callback_trade(self, callback: CallbackQuery, state: FSMContext):
        """Handle trading callbacks"""
        try:
            if not callback.data:
                await callback.answer("❌ Data tidak valid.")
                return
                
            data_parts = callback.data.split("_")
            action = data_parts[1]  # trade action
            
            if action == "buy" or action == "sell":
                pair_id = data_parts[2] if len(data_parts) > 2 else "btc_idr"
                user = await self._get_or_create_user(callback.from_user)
                
                if action == "buy":
                    # Show amount selection with balance and percentage options
                    await self._show_buy_amount_selection(callback.message, user, pair_id, state)
                else:
                    # For sell, show different options
                    await self._safe_edit_or_send(callback.message, f"💰 Masukkan jumlah {pair_id.split('_')[0].upper()} untuk jual:")
                    await state.set_state(TradingStates.entering_amount)
                    await state.update_data(trade_type=action, pair_id=pair_id)
            
            await callback.answer()
            
        except Exception as e:
            logger.error("Failed to handle trade callback", error=str(e))
            await callback.answer("❌ Terjadi kesalahan.")
    
    async def callback_signal(self, callback: CallbackQuery):
        """Handle signal callbacks"""
        try:
            if not callback.data:
                await callback.answer("❌ Data tidak valid.")
                return
                
            data_parts = callback.data.split("_")
            action = data_parts[1]
            
            if action == "update":
                pair_id = data_parts[2] if len(data_parts) > 2 else "btc_idr"
                
                signal = await self.signal_generator.generate_signal(pair_id)
                if signal and callback.message:
                    user = await self._get_or_create_user(callback.from_user)
                    signal_text = self.messages.format_signal(signal, getattr(user, 'language', 'id'))
                    
                    await self._safe_edit_or_send(callback.message, signal_text)
                else:
                    await callback.answer("❌ Tidak dapat memperbarui sinyal.")
            
            await callback.answer("🔄 Sinyal diperbarui!")
            
        except Exception as e:
            logger.error("Failed to handle signal callback", error=str(e))
            await callback.answer("❌ Terjadi kesalahan.")
    
    async def callback_settings(self, callback: CallbackQuery, state: FSMContext):
        """Handle settings callbacks"""
        try:
            if not callback.data or not callback.message:
                await callback.answer("❌ Data tidak valid.")
                return
                
            data_parts = callback.data.split("_")
            action = data_parts[1]
            
            user = await self._get_or_create_user(callback.from_user)
            
            if action == "stoploss":
                await self._safe_edit_or_send(callback.message, "🛑 Masukkan nilai Stop Loss (%):")
                await state.set_state(SettingsStates.editing_stop_loss)
            elif action == "takeprofit":
                await self._safe_edit_or_send(callback.message, "🎯 Masukkan nilai Take Profit (%):")
                await state.set_state(SettingsStates.editing_take_profit)
            elif action == "maxamount":
                await self._safe_edit_or_send(callback.message, "💰 Masukkan jumlah maksimal trading (IDR):")
                await state.set_state(SettingsStates.editing_max_trade_amount)
            elif action == "language":
                keyboard = InlineKeyboardMarkup(inline_keyboard=[
                    [InlineKeyboardButton(text="🇮🇩 Bahasa Indonesia", callback_data="lang_id")],
                    [InlineKeyboardButton(text="🇺🇸 English", callback_data="lang_en")]
                ])
                await self._safe_edit_or_send(callback.message, "🌐 Pilih bahasa:", keyboard)
            
            await callback.answer()
            
        except Exception as e:
            logger.error("Failed to handle settings callback", error=str(e))
            await callback.answer("❌ Terjadi kesalahan.")

    async def callback_confirm(self, callback: CallbackQuery, state: FSMContext):
        """Handle confirmation callbacks"""
        try:
            if not callback.data or not callback.message:
                await callback.answer("❌ Data tidak valid.")
                return
                
            data_parts = callback.data.split("_")
            action = data_parts[1]
            
            if action == "yes":
                # Get state data
                data = await state.get_data()
                trade_type = data.get('trade_type')
                pair_id = data.get('pair_id', 'btc_idr')
                amount = data.get('amount', 0)
                
                user = await self._get_or_create_user(callback.from_user)
                
                if trade_type and amount > 0:
                    await self._execute_trade(callback.message, user, pair_id, trade_type, amount)
                else:
                    await self._safe_edit_or_send(callback.message, "❌ Data trading tidak valid.")
                
                await state.clear()
                
            elif action == "no":
                await self._safe_edit_or_send(callback.message, "❌ Trading dibatalkan.")
                await state.clear()
            
            await callback.answer()
            
        except Exception as e:
            logger.error("Failed to handle confirmation callback", error=str(e))
            await callback.answer("❌ Terjadi kesalahan.")

    async def callback_language(self, callback: CallbackQuery):
        """Handle language selection callbacks"""
        try:
            if not callback.data or not callback.message:
                await callback.answer("❌ Data tidak valid.")
                return
                
            data_parts = callback.data.split("_")
            lang_code = data_parts[1]
            
            user = await self._get_or_create_user(callback.from_user)
            
            # Update user language
            db = get_db()
            try:
                setattr(user, 'language', lang_code)
                db.commit()
                
                lang_name = "Bahasa Indonesia" if lang_code == "id" else "English"
                await self._safe_edit_or_send(callback.message, f"✅ Bahasa berhasil diubah ke {lang_name}")
                
            finally:
                db.close()
            
            await callback.answer()
            
        except Exception as e:
            logger.error("Failed to handle language callback", error=str(e))
            await callback.answer("❌ Terjadi kesalahan.")

    async def callback_admin(self, callback: CallbackQuery):
        """Handle admin callbacks"""
        try:
            if not is_admin_user(callback.from_user.id):
                await callback.answer("❌ Akses ditolak.")
                return
            
            if not callback.data or not callback.message:
                await callback.answer("❌ Data tidak valid.")
                return
                
            data_parts = callback.data.split("_")
            action = data_parts[1]
            
            if action == "stats":
                stats = await self._get_bot_statistics()
                stats_text = f"""
📊 **Statistik Bot**

👥 Total Users: {stats.get('total_users', 0)}
✅ Registered Users: {stats.get('registered_users', 0)}
⭐ Premium Users: {stats.get('premium_users', 0)}
📈 Total Trades: {stats.get('total_trades', 0)}
"""
                await self._safe_edit_or_send(callback.message, stats_text)
            
            await callback.answer()
            
        except Exception as e:
            logger.error("Failed to handle admin callback", error=str(e))
            await callback.answer("❌ Terjadi kesalahan.")

    async def callback_backtest(self, callback_query: CallbackQuery):
        """Handle backtest-related callbacks"""
        try:
            callback_data = callback_query.data or ""
            user_id = callback_query.from_user.id if callback_query.from_user else 0
            user = await self._get_or_create_user(callback_query.from_user)
            
            if callback_data == "backtest_new":
                # User wants to run a new backtest
                if callback_query.message:
                    await self._safe_edit_or_send(
                        callback_query.message,
                        "🔄 <b>Backtest Baru</b>\n\n"
                        "Gunakan format: <code>/backtest [pair] [strategy] [period]</code>\n\n"
                        "<b>Contoh:</b>\n"
                        "<code>/backtest btc ai_signals 30d</code>\n"
                        "<code>/backtest eth buy_and_hold 90d</code>\n"
                        "<code>/backtest sol dca 7d</code>"
                    )
                await callback_query.answer("Silakan jalankan backtest baru")
                
            elif callback_data.startswith("backtest_apply_"):
                # Format: backtest_apply_[pair_id]_[strategy]
                parts = callback_data.split("_")
                if len(parts) < 4:
                    await callback_query.answer("Format callback tidak valid")
                    return
                    
                pair_id = parts[2]
                strategy = parts[3] 
                
                # Check if we need to add back the underscore for pair_id
                if "_" not in pair_id and pair_id.endswith("idr"):
                    base_symbol = pair_id[:-3]
                    pair_id = f"{base_symbol}_idr"
                
                # Show application confirmation message
                if callback_query.message:
                    await self._safe_edit_or_send(
                        callback_query.message,
                        f"⚙️ <b>Menerapkan Strategi</b>\n\n"
                        f"<b>Pair:</b> {pair_id.upper()}\n"
                        f"<b>Strategy:</b> {strategy}\n\n"
                        f"Menginisialisasi pengaturan trading..."
                    )
                
                db = None
                try:
                    # Get user settings
                    db = get_db()
                    user_settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
                    
                    if not user_settings:
                        user_settings = UserSettings(
                            user_id=user_id,
                            active_pairs=json.dumps([pair_id]),
                            strategies=json.dumps({pair_id: strategy}),
                            trade_amount_percent=10,  # Default to 10% per trade
                            risk_level="moderate",
                            auto_trade=True
                        )
                        db.add(user_settings)
                    else:
                        # Update existing settings
                        active_pairs = json.loads(user_settings.active_pairs) if user_settings.active_pairs else []
                        strategies = json.loads(user_settings.strategies) if user_settings.strategies else {}
                        
                        # Add or update pair and strategy
                        if pair_id not in active_pairs:
                            active_pairs.append(pair_id)
                        
                        strategies[pair_id] = strategy
                        
                        # Update user settings
                        user_settings.active_pairs = json.dumps(active_pairs)
                        user_settings.strategies = json.dumps(strategies)
                        user_settings.auto_trade = True
                    
                    db.commit()
                    
                    # Send success message with advanced keyboard
                    keyboard = InlineKeyboardMarkup(inline_keyboard=[
                        [
                            InlineKeyboardButton(text="📊 Settings", callback_data="settings_trading"),
                            InlineKeyboardButton(text="🔄 Run Another Backtest", callback_data="backtest_new")
                        ],
                        [
                            InlineKeyboardButton(text="📈 Portfolio", callback_data="portfolio")
                        ]
                    ])
                    
                    if callback_query.message:
                        await self._safe_edit_or_send(
                            callback_query.message,
                            f"✅ <b>Strategi berhasil diterapkan!</b>\n\n"
                            f"Strategi {strategy} telah diaktifkan untuk pair {pair_id.upper()}.\n\n"
                            f"<b>Auto Trading:</b> {'Aktif ✅' if user_settings.auto_trade else 'Tidak Aktif ❌'}\n"
                            f"<b>Risk Level:</b> {user_settings.risk_level.capitalize()}\n"
                            f"<b>Trade Amount:</b> {user_settings.trade_amount_percent}% per trade\n\n"
                            f"Bot akan menggunakan advanced AI signal generator untuk menghasilkan sinyal dengan akurasi dan performa yang tinggi.",
                            keyboard
                        )
                    
                    await callback_query.answer("Strategi berhasil diterapkan")
                    
                    # Log the action
                    logger.info(
                        "Applied backtest strategy", 
                        user_id=user_id, 
                        pair=pair_id, 
                        strategy=strategy
                    )
                    
                except Exception as db_error:
                    logger.error("Failed to apply strategy settings", error=str(db_error))
                    if callback_query.message:
                        await self._safe_edit_or_send(callback_query.message, "❌ Terjadi kesalahan saat menerapkan strategi.")
                    await callback_query.answer("Gagal menerapkan strategi")
                finally:
                    if db:
                        db.close()
            
            else:
                await callback_query.answer("Opsi tidak valid")
                
        except Exception as e:
            logger.error("Failed to process backtest callback", error=str(e))
            await callback_query.answer("Terjadi kesalahan")
    
    async def process_trade_amount(self, message: Message, state: FSMContext):
        """Process trade amount input"""
        try:
            if not message.text:
                await message.answer("❌ Pesan tidak valid.")
                return
                
            amount_text = message.text.strip()
            
            # Try to parse amount
            try:
                amount = float(amount_text.replace(",", ""))
            except ValueError:
                await message.answer("❌ Jumlah tidak valid. Masukkan angka yang benar.")
                return
            
            if amount <= 0:
                await message.answer("❌ Jumlah harus lebih besar dari 0.")
                return
            
            # Check minimum order requirement for Indodax (10,000 IDR)
            if amount < 10000:
                await message.answer("❌ Minimum order di Indodax adalah 10,000 IDR. Silakan masukkan jumlah yang lebih besar.")
                return
            
            # Get state data
            data = await state.get_data()
            trade_type = data.get('trade_type')
            pair_id = data.get('pair_id', 'btc_idr')
            
            # Update state with amount
            await state.update_data(amount=amount)
            
            # Show confirmation
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="✅ Ya", callback_data="confirm_yes"),
                    InlineKeyboardButton(text="❌ Tidak", callback_data="confirm_no")
                ]
            ])
            
            confirm_text = f"""
🔄 **Konfirmasi Trading**

Action: {str(trade_type).upper() if trade_type else 'UNKNOWN'}
Pair: {pair_id.upper()}
Amount: {format_currency(amount)} IDR

Lanjutkan trading?
"""
            
            await message.answer(confirm_text, reply_markup=keyboard)
            
        except Exception as e:
            logger.error("Failed to process trade amount", error=str(e))
            await message.answer("❌ Terjadi kesalahan. Silakan coba lagi.")
            await state.clear()

    async def process_stop_loss(self, message: Message, state: FSMContext):
        """Process stop loss setting"""
        try:
            if not message.text:
                await message.answer("❌ Pesan tidak valid.")
                return
                
            value_text = message.text.strip().replace("%", "")
            
            try:
                stop_loss = float(value_text)
            except ValueError:
                await message.answer("❌ Nilai tidak valid. Masukkan angka yang benar.")
                return
            
            if stop_loss <= 0 or stop_loss > 50:
                await message.answer("❌ Stop Loss harus antara 0.1% - 50%.")
                return
            
            user = await self._get_or_create_user(message.from_user)
            
            # Update user settings
            db = get_db()
            try:
                settings_record = db.query(UserSettings).filter(UserSettings.user_id == user.id).first()
                if not settings_record:
                    settings_record = UserSettings(user_id=user.id)
                    db.add(settings_record)
                
                setattr(settings_record, 'stop_loss_percentage', stop_loss)
                db.commit()
                
                await message.answer(f"✅ Stop Loss berhasil diubah ke {stop_loss}%")
                
            finally:
                db.close()
            
            await state.clear()
            
        except Exception as e:
            logger.error("Failed to process stop loss", error=str(e))
            await message.answer("❌ Terjadi kesalahan. Silakan coba lagi.")
            await state.clear()

    async def process_take_profit(self, message: Message, state: FSMContext):
        """Process take profit setting"""
        try:
            if not message.text:
                await message.answer("❌ Pesan tidak valid.")
                return
                
            value_text = message.text.strip().replace("%", "")
            
            try:
                take_profit = float(value_text)
            except ValueError:
                await message.answer("❌ Nilai tidak valid. Masukkan angka yang benar.")
                return
            
            if take_profit <= 0 or take_profit > 100:
                await message.answer("❌ Take Profit harus antara 0.1% - 100%.")
                return
            
            user = await self._get_or_create_user(message.from_user)
            
            # Update user settings
            db = get_db()
            try:
                settings_record = db.query(UserSettings).filter(UserSettings.user_id == user.id).first()
                if not settings_record:
                    settings_record = UserSettings(user_id=user.id)
                    db.add(settings_record)
                
                setattr(settings_record, 'take_profit_percentage', take_profit)
                db.commit()
                
                await message.answer(f"✅ Take Profit berhasil diubah ke {take_profit}%")
                
            finally:
                db.close()
            
            await state.clear()
            
        except Exception as e:
            logger.error("Failed to process take profit", error=str(e))
            await message.answer("❌ Terjadi kesalahan. Silakan coba lagi.")
            await state.clear()

    async def process_max_trade_amount(self, message: Message, state: FSMContext):
        """Process max trade amount setting"""
        try:
            if not message.text:
                await message.answer("❌ Pesan tidak valid.")
                return
                
            amount_text = message.text.strip()
            
            try:
                max_amount = float(amount_text.replace(",", "").replace(".", ""))
            except ValueError:
                await message.answer("❌ Jumlah tidak valid. Masukkan angka yang benar.")
                return
            
            if max_amount <= 0:
                await message.answer("❌ Jumlah harus lebih besar dari 0.")
                return
            
            user = await self._get_or_create_user(message.from_user)
            
            # Update user settings
            db = get_db()
            try:
                settings_record = db.query(UserSettings).filter(UserSettings.user_id == user.id).first()
                if not settings_record:
                    settings_record = UserSettings(user_id=user.id)
                    db.add(settings_record)
                
                settings_record.max_trade_amount = max_amount
                db.commit()
                
                await message.answer(f"✅ Jumlah maksimal trading berhasil diubah ke {format_currency(max_amount)} IDR")
                
            finally:
                db.close()
            
            await state.clear()
            
        except Exception as e:
            logger.error("Failed to process max trade amount", error=str(e))
            await message.answer("❌ Terjadi kesalahan. Silakan coba lagi.")
            await state.clear()
    
    async def _safe_edit_or_send(self, callback_message, text: str, reply_markup=None):
        """Safely edit message or send new one if edit fails"""
        try:
            if hasattr(callback_message, 'edit_text'):
                await callback_message.edit_text(text, reply_markup=reply_markup)
            else:
                await callback_message.answer(text, reply_markup=reply_markup)
        except Exception:
            # If edit fails, try to send new message
            try:
                await callback_message.answer(text, reply_markup=reply_markup)
            except Exception:
                # If both fail, ignore silently
                pass
    
    # Helper Methods
    
    async def _get_portfolio_data(self, user) -> Dict[str, Any]:
        """Get real portfolio data from Indodax"""
        try:
            if not getattr(user, 'indodax_api_key', None) or not getattr(user, 'indodax_secret_key', None):
                return {"error": "API credentials not set"}
            
            # Decrypt API credentials
            api_key = decrypt_api_key(str(getattr(user, 'indodax_api_key', '')))
            secret_key = decrypt_api_key(str(getattr(user, 'indodax_secret_key', '')))
            
            # Create user-specific API client
            user_api = IndodaxAPI(api_key, secret_key)
            
            # Get account info and balance
            account_info = await user_api.get_info()
            
            # Get all tickers for current prices
            tickers = await user_api.get_ticker_all()
            
            portfolio = {
                "balances": {},
                "total_idr_value": 0,
                "assets": []
            }
            
            # Process balances
            if "return" in account_info and "balance" in account_info["return"]:
                balances = account_info["return"]["balance"]
                
                for currency, balance in balances.items():
                    balance_float = float(balance)
                    if balance_float > 0:
                        asset_data = {
                            "currency": currency.upper(),
                            "balance": balance_float,
                            "idr_value": balance_float
                        }
                        
                        # Calculate IDR value for non-IDR currencies
                        if currency.lower() != "idr":
                            ticker_key = f"{currency.lower()}_idr"
                            if ticker_key in tickers.get("tickers", {}):
                                ticker = tickers["tickers"][ticker_key]
                                current_price = float(ticker.get("last", 0))
                                asset_data["current_price"] = current_price
                                asset_data["idr_value"] = balance_float * current_price
                        
                        portfolio["balances"][currency] = asset_data
                        portfolio["assets"].append(asset_data)
                        portfolio["total_idr_value"] += asset_data["idr_value"]
            
            return portfolio
            
        except Exception as e:
            logger.error("Failed to get portfolio data", user_id=user.id, error=str(e))
            return {"error": str(e)}
    
    async def _execute_trade(self, message, user, pair_id: str, trade_type: str, amount: float):
        """Execute real trade on Indodax"""
        try:
            if not getattr(user, 'indodax_api_key', None) or not getattr(user, 'indodax_secret_key', None):
                await message.answer("❌ API credentials belum diatur. Gunakan /daftar untuk setup.")
                return
            
            # Decrypt API credentials
            api_key = decrypt_api_key(str(getattr(user, 'indodax_api_key', '')))
            secret_key = decrypt_api_key(str(getattr(user, 'indodax_secret_key', '')))
            
            # Create user-specific API client
            user_api = IndodaxAPI(api_key, secret_key)
            
            # Ensure pair_id has correct format (e.g. btcidr, not btc)
            if "_" in pair_id:
                # Already in correct format like btc_idr
                ticker_pair = pair_id.replace("_", "")  # btcidr for ticker API
            else:
                # Assume it's just the symbol like "btc"
                ticker_pair = f"{pair_id}idr"
                pair_id = f"{pair_id}_idr"  # For trade API
            
            # Get current ticker to determine price
            ticker = await user_api.get_ticker(ticker_pair)
            
            if trade_type == "buy":
                # For buy orders, amount is in IDR, calculate quantity
                if "ticker" in ticker:
                    price = float(ticker["ticker"]["sell"])  # Use sell price for buying
                else:
                    price = float(ticker.get("last", ticker.get("sell", 0)))
                
                if price <= 0:
                    await message.answer("❌ Tidak dapat mendapatkan harga saat ini.")
                    return
                
                # Validate minimum order value (10,000 IDR)
                if amount < 10000:
                    await message.answer("❌ Minimum order di Indodax adalah 10,000 IDR.")
                    return
                
                # Log the calculation for debugging
                logger.info("Buy order calculation", 
                           amount_idr=amount, 
                           price=price, 
                           pair=pair_id)
                
                # Place buy order using IDR amount (let Indodax calculate quantity)
                result = await user_api.trade(
                    pair=pair_id,
                    type="buy",
                    price=price,
                    idr_amount=amount
                )
                
                # Calculate quantity for database storage
                quantity = amount / price
                
            else:  # sell
                # For sell orders, amount is also in IDR, calculate quantity to sell
                if "ticker" in ticker:
                    price = float(ticker["ticker"]["buy"])  # Use buy price for selling
                else:
                    price = float(ticker.get("last", ticker.get("buy", 0)))
                
                if price <= 0:
                    await message.answer("❌ Tidak dapat mendapatkan harga saat ini.")
                    return
                
                # Validate minimum order value (10,000 IDR)
                if amount < 10000:
                    await message.answer("❌ Minimum order di Indodax adalah 10,000 IDR.")
                    return
                
                # Calculate quantity to sell based on IDR amount
                quantity = amount / price
                
                # Double check: ensure calculated total meets minimum
                calculated_total = quantity * price
                if calculated_total < 10000:
                    await message.answer(f"❌ Order terlalu kecil. Total nilai: {format_currency(calculated_total)} IDR. Minimum: 10,000 IDR.")
                    return
                
                # Log the calculation for debugging
                logger.info("Sell order calculation", 
                           amount_idr=amount, 
                           price=price, 
                           quantity=quantity, 
                           calculated_total=calculated_total,
                           pair=pair_id)
                
                # Place sell order using coin amount
                result = await user_api.trade(
                    pair=pair_id,
                    type="sell",
                    price=price,
                    coin_amount=quantity
                )
            
            if result.get("success") == 1:
                # Save trade to database
                db = get_db()
                try:
                    trade = Trade(
                        user_id=user.id,
                        pair_id=pair_id,
                        order_id=str(result["return"]["order_id"]),
                        type=trade_type,
                        amount=quantity,  # Store quantity, not IDR amount
                        price=price,
                        total=amount,  # Store IDR amount as total
                        status="pending",
                        created_at=datetime.utcnow()
                    )
                    db.add(trade)
                    db.commit()
                finally:
                    db.close()
                
                success_msg = f"""
✅ <b>Order berhasil dibuat!</b>

Order ID: {result["return"]["order_id"]}
Type: {trade_type.upper()}
Pair: {pair_id.upper()}
Amount: {format_currency(amount)} IDR
Quantity: {quantity:.8f}
Price: {format_currency(price)} IDR

Order akan dieksekusi sesuai kondisi pasar.
                """
                await message.answer(success_msg)
                
                # Start monitoring order status
                asyncio.create_task(self._monitor_order_status(user.id, result["return"]["order_id"], pair_id))
                
                # Start monitoring order status
                asyncio.create_task(self._monitor_order_status(user.id, result["return"]["order_id"], pair_id))
            else:
                error_msg = result.get("error", "Unknown error occurred")
                await message.answer(f"❌ Gagal membuat order: {error_msg}")
            
        except Exception as e:
            logger.error("Failed to execute trade", user_id=user.id, pair_id=pair_id, trade_type=trade_type, error=str(e))
            await message.answer(f"❌ Terjadi kesalahan saat trading: {str(e)}")
    
    async def _get_bot_statistics(self) -> Dict[str, Any]:
        """Get bot statistics from database"""
        try:
            db = get_db()
            try:
                # Count users
                total_users = db.query(User).count()
                registered_users = db.query(User).filter(User.indodax_api_key.isnot(None)).count()
                premium_users = db.query(User).filter(User.is_premium == True).count()
                
                # Count trades
                total_trades = db.query(Trade).count()
                completed_trades = db.query(Trade).filter(Trade.status == "completed").count()
                
                return {
                    "total_users": total_users,
                    "registered_users": registered_users,
                    "premium_users": premium_users,
                    "total_trades": total_trades,
                    "completed_trades": completed_trades
                }
            finally:
                db.close()
        except Exception as e:
            logger.error("Failed to get bot statistics", error=str(e))
            return {}
    
    async def _broadcast_message(self, message_text: str):
        """Broadcast message to all users"""
        try:
            db = get_db()
            try:
                users = db.query(User).filter(User.is_active == True).all()
                
                success_count = 0
                error_count = 0
                
                for user in users:
                    try:
                        await self.bot.send_message(getattr(user, 'telegram_id', 0), message_text)
                        success_count += 1
                        # Small delay to avoid rate limits
                        await asyncio.sleep(0.1)
                    except Exception:
                        error_count += 1
                
                logger.info("Broadcast completed", success=success_count, errors=error_count)
                return success_count, error_count
                
            finally:
                db.close()
        except Exception as e:
            logger.error("Failed to broadcast message", error=str(e))
            return 0, 0
    
    async def _monitor_order_status(self, user_id: int, order_id: str, pair_id: str):
        """Monitor order status and notify user when executed"""
        try:
            # Get user
            db = get_db()
            try:
                user = db.query(User).filter(User.id == user_id).first()
                if not user or not getattr(user, 'indodax_api_key', None):
                    return
                
                # Decrypt API credentials
                api_key = decrypt_api_key(str(user.indodax_api_key))
                secret_key = decrypt_api_key(str(user.indodax_secret_key))
                user_api = IndodaxAPI(api_key, secret_key)
                
                # Check order status periodically
                max_checks = 60  # Check for 1 hour (60 minutes)
                check_count = 0
                
                while check_count < max_checks:
                    try:
                        # Get order history to check if executed
                        order_history = await user_api.get_order_history(pair=pair_id, count=100)
                        
                        if "return" in order_history and "orders" in order_history["return"]:
                            for order in order_history["return"]["orders"]:
                                if str(order["order_id"]) == str(order_id):
                                    if order["status"] == "filled":
                                        # Order executed, send notification
                                        await self._send_order_notification(user, order, pair_id)
                                        
                                        # Update database
                                        trade = db.query(Trade).filter(Trade.order_id == str(order_id)).first()
                                        if trade:
                                            setattr(trade, 'status', "completed")
                                            trade.updated_at = datetime.utcnow()
                                            db.commit()
                                        
                                        return  # Stop monitoring
                                    elif order["status"] == "cancelled":
                                        # Order cancelled
                                        await self._send_order_cancelled_notification(user, order, pair_id)
                                        
                                        # Update database
                                        trade = db.query(Trade).filter(Trade.order_id == str(order_id)).first()
                                        if trade:
                                            setattr(trade, 'status', "cancelled")
                                            trade.updated_at = datetime.utcnow()
                                            db.commit()
                                        
                                        return  # Stop monitoring
                        
                        # Wait 1 minute before next check
                        await asyncio.sleep(60)
                        check_count += 1
                        
                    except Exception as e:
                        logger.error("Error checking order status", order_id=order_id, error=str(e))
                        await asyncio.sleep(60)
                        check_count += 1
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error("Failed to monitor order status", order_id=order_id, error=str(e))
    
    async def _send_order_notification(self, user, order, pair_id: str):
        """Send notification when order is executed"""
        try:
            notification_text = f"""
🎉 <b>Order Tereksekusi!</b>

Order ID: {order['order_id']}
Type: {order['type'].upper()}
Pair: {pair_id.upper()}
Amount: {order['order_amount']} {pair_id.split('_')[0].upper()}
Price: {format_currency(float(order['price']))} IDR
Total: {format_currency(float(order['remain_amount']) * float(order['price']))} IDR
Status: ✅ COMPLETED

⏰ Waktu: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
            """
            
            await self.bot.send_message(user.telegram_id, notification_text)
            logger.info("Order notification sent", user_id=user.id, order_id=order['order_id'])
            
        except Exception as e:
            logger.error("Failed to send order notification", user_id=user.id, error=str(e))
    
    async def _send_order_cancelled_notification(self, user, order, pair_id: str):
        """Send notification when order is cancelled"""
        try:
            notification_text = f"""
❌ <b>Order Dibatalkan</b>

Order ID: {order['order_id']}
Type: {order['type'].upper()}
Pair: {pair_id.upper()}
Amount: {order['order_amount']} {pair_id.split('_')[0].upper()}
Price: {format_currency(float(order['price']))} IDR
Status: ❌ CANCELLED

⏰ Waktu: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
            """
            
            await self.bot.send_message(user.telegram_id, notification_text)
            logger.info("Order cancellation notification sent", user_id=user.id, order_id=order['order_id'])
            
        except Exception as e:
            logger.error("Failed to send order cancellation notification", user_id=user.id, error=str(e))

    async def callback_amount_selection(self, callback: CallbackQuery, state: FSMContext):
        """Handle amount selection callbacks"""
        try:
            if not callback.data or not callback.message:
                await callback.answer("❌ Data tidak valid.")
                return
                
            data_parts = callback.data.split("_")
            if len(data_parts) < 3:
                await callback.answer("❌ Data tidak valid.")
                return
                
            action = data_parts[1]  # percentage or custom
            percentage = data_parts[2] if len(data_parts) > 2 else "0"
            
            # Get state data
            data = await state.get_data()
            trade_type = data.get('trade_type', 'buy')
            pair_id = data.get('pair_id', 'btc_idr')
            idr_balance = data.get('idr_balance', 0)
            
            if action == "percentage":
                # Calculate amount based on percentage
                percentage_value = float(percentage)
                amount = (idr_balance * percentage_value) / 100
                
                # Validate minimum order
                if amount < 10000:
                    await callback.answer("❌ Jumlah terlalu kecil (min. 10,000 IDR).")
                    return
                
                # Update state with calculated amount
                await state.update_data(amount=amount)
                
                # Show confirmation
                keyboard = InlineKeyboardMarkup(inline_keyboard=[
                    [
                        InlineKeyboardButton(text="✅ Ya", callback_data="confirm_yes"),
                        InlineKeyboardButton(text="❌ Tidak", callback_data="confirm_no")
                    ]
                ])
                
                confirm_text = f"""
🔄 **Konfirmasi Trading**

Action: {trade_type.upper()}
Pair: {pair_id.upper()}
Amount: {format_currency(amount)} IDR ({percentage}% dari saldo)
Saldo IDR: {format_currency(idr_balance)} IDR

Lanjutkan trading?
"""
                
                await self._safe_edit_or_send(callback.message, confirm_text, keyboard)
                
            elif action == "custom":
                # Ask for custom amount input
                await self._safe_edit_or_send(callback.message, f"💰 Masukkan jumlah IDR untuk {trade_type} {pair_id.upper()} : (Min. 10,000 IDR)")
                await state.set_state(TradingStates.entering_amount)
            
            await callback.answer()
            
            await callback.answer()
            
        except Exception as e:
            logger.error("Failed to handle amount selection callback", error=str(e))
            await callback.answer("❌ Terjadi kesalahan.")
    
    async def _show_buy_amount_selection(self, message, user, pair_id: str, state: FSMContext):
        """Show buy amount selection with balance and percentage options"""
        try:
            # Get user's IDR balance
            api_key = decrypt_api_key(str(getattr(user, 'indodax_api_key', '')))
            secret_key = decrypt_api_key(str(getattr(user, 'indodax_secret_key', '')))
            user_api = IndodaxAPI(api_key, secret_key)
            
            balance_data = await user_api.get_balance()
            idr_balance = float(balance_data.get('idr', 0))
            
            # Update state with balance info
            await state.update_data(trade_type="buy", pair_id=pair_id, idr_balance=idr_balance)
            
            # Create keyboard with percentage options
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text=f"25% ({format_currency(idr_balance * 0.25)} IDR)", callback_data="amount_percentage_25"),
                    InlineKeyboardButton(text=f"50% ({format_currency(idr_balance * 0.5)} IDR)", callback_data="amount_percentage_50")
                ],
                [
                    InlineKeyboardButton(text=f"75% ({format_currency(idr_balance * 0.75)} IDR)", callback_data="amount_percentage_75"),
                    InlineKeyboardButton(text=f"100% ({format_currency(idr_balance)} IDR)", callback_data="amount_percentage_100")
                ],
                [
                    InlineKeyboardButton(text="💰 Custom Amount", callback_data="amount_custom_0")
                ]
            ])
            
            selection_text = f"""
💰 **Pilih Jumlah untuk Buy {pair_id.upper()}**

💵 Saldo IDR Tersedia: {format_currency(idr_balance)} IDR
📊 Minimum Order: 10,000 IDR

Pilih persentase dari saldo atau masukkan jumlah custom:
"""
            
            await message.answer(selection_text, reply_markup=keyboard)
            
        except Exception as e:
            logger.error("Failed to show buy amount selection", error=str(e))
            await message.answer("❌ Terjadi kesalahan saat mengambil data saldo.")

    # Main Menu Callback Handlers
    
    async def callback_portfolio(self, callback: CallbackQuery):
        """Handle portfolio callback"""
        try:
            if not callback.message:
                await callback.answer("❌ Data tidak valid.")
                return
                
            user = await self._get_or_create_user(callback.from_user)
            
            if not getattr(user, 'indodax_api_key', None):
                await self._safe_edit_or_send(callback.message, "❌ Anda belum terdaftar. Gunakan /daftar untuk mendaftar.")
                return
            
            # Get portfolio data
            portfolio_data = await self._get_portfolio_data(user)
            if "error" in portfolio_data:
                await self._safe_edit_or_send(callback.message, f"❌ Error: {portfolio_data['error']}")
                return
                
            portfolio_text = self.messages.format_portfolio(portfolio_data, getattr(user, 'language', 'id'))
            await self._safe_edit_or_send(callback.message, portfolio_text)
            await callback.answer()
            
        except Exception as e:
            logger.error("Failed to handle portfolio callback", error=str(e))
            await callback.answer("❌ Terjadi kesalahan.")

    async def callback_balance(self, callback: CallbackQuery):
        """Handle balance callback"""
        try:
            if not callback.message:
                await callback.answer("❌ Data tidak valid.")
                return
                
            user = await self._get_or_create_user(callback.from_user)
            
            if not getattr(user, 'indodax_api_key', None):
                await self._safe_edit_or_send(callback.message, "❌ Anda belum terdaftar. Gunakan /daftar untuk mendaftar.")
                return
                
            # Get balance from Indodax
            api_key = decrypt_api_key(str(getattr(user, 'indodax_api_key', '')))
            secret_key = decrypt_api_key(str(getattr(user, 'indodax_secret_key', '')))
            
            user_api = IndodaxAPI(api_key, secret_key)
            balance_data = await user_api.get_balance()
            
            balance_text = self.messages.format_balance(balance_data, getattr(user, 'language', 'id'))
            await self._safe_edit_or_send(callback.message, balance_text)
            await callback.answer()
            
        except Exception as e:
            logger.error("Failed to handle balance callback", error=str(e))
            await callback.answer("❌ Terjadi kesalahan.")

    async def callback_signals(self, callback: CallbackQuery):
        """Handle signals callback"""
        try:
            if not callback.message:
                await callback.answer("❌ Data tidak valid.")
                return
                
            user = await self._get_or_create_user(callback.from_user)
            
            # Generate signal
            signal = await self.signal_generator.generate_signal("btc_idr")
            
            if signal:
                signal_text = self.messages.format_signal(signal, getattr(user, 'language', 'id'))
                keyboard = InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(
                        text="🔄 Update Signal",
                        callback_data="signal_update_btc_idr"
                    ),
                    InlineKeyboardButton(
                        text="💹 Trade",
                        callback_data=f"trade_btc_idr_{getattr(signal, 'signal_type', 'buy')}"
                    )
                ]])
                
                try:
                    await self._safe_edit_or_send(callback.message, signal_text, keyboard)
                except Exception:
                    await self._safe_edit_or_send(callback.message, signal_text, keyboard)
            else:
                await self._safe_edit_or_send(callback.message, "❌ Tidak dapat menghasilkan sinyal saat ini.")
            
            await callback.answer()
            
        except Exception as e:
            logger.error("Failed to handle signals callback", error=str(e))
            await callback.answer("❌ Terjadi kesalahan.")

    async def callback_trading(self, callback: CallbackQuery):
        """Handle trading callback"""
        try:
            if not callback.message:
                await callback.answer("❌ Data tidak valid.")
                return
                
            user = await self._get_or_create_user(callback.from_user)
            
            if not getattr(user, 'indodax_api_key', None):
                await self._safe_edit_or_send(callback.message, "❌ Anda belum terdaftar. Gunakan /daftar untuk mendaftar.")
                return
                
            keyboard = create_trading_keyboard("buy", getattr(user, 'language', 'id'))
            await self._safe_edit_or_send(callback.message, "💹 Pilih jenis trading:", keyboard)
            await callback.answer()
            
        except Exception as e:
            logger.error("Failed to handle trading callback", error=str(e))
            await callback.answer("❌ Terjadi kesalahan.")

    async def callback_orders(self, callback: CallbackQuery):
        """Handle orders callback"""
        try:
            if not callback.message:
                await callback.answer("❌ Data tidak valid.")
                return
                
            user = await self._get_or_create_user(callback.from_user)
            
            if not getattr(user, 'indodax_api_key', None):
                await self._safe_edit_or_send(callback.message, "❌ Anda belum terdaftar. Gunakan /daftar untuk mendaftar.")
                return
            
            # Get open orders
            api_key = decrypt_api_key(str(getattr(user, 'indodax_api_key', '')))
            secret_key = decrypt_api_key(str(getattr(user, 'indodax_secret_key', '')))
            
            user_api = IndodaxAPI(api_key, secret_key)
            orders_data = await user_api.get_open_orders()
            
            orders_text = self.messages.format_orders(orders_data, getattr(user, 'language', 'id'))
            await self._safe_edit_or_send(callback.message, orders_text)
            await callback.answer()
            
        except Exception as e:
            logger.error("Failed to handle orders callback", error=str(e))
            await callback.answer("❌ Terjadi kesalahan.")

    async def callback_settings_menu(self, callback: CallbackQuery):
        """Handle settings menu callback"""
        try:
            if not callback.message:
                await callback.answer("❌ Data tidak valid.")
                return
                
            user = await self._get_or_create_user(callback.from_user)
            settings_keyboard = create_settings_keyboard(getattr(user, 'language', 'id'))
            settings_text = self.messages.get_settings_message(getattr(user, 'language', 'id'))
            
            await self._safe_edit_or_send(callback.message, settings_text, settings_keyboard)
            await callback.answer()
            
        except Exception as e:
            logger.error("Failed to handle settings callback", error=str(e))
            await callback.answer("❌ Terjadi kesalahan.")

    async def callback_help(self, callback: CallbackQuery):
        """Handle help callback"""
        try:
            if not callback.message:
                await callback.answer("❌ Data tidak valid.")
                return
                
            user = await self._get_or_create_user(callback.from_user)
            help_text = self.messages.get_help_message(getattr(user, 'language', 'id'))
            
            await self._safe_edit_or_send(callback.message, help_text)
            await callback.answer()
            
        except Exception as e:
            logger.error("Failed to handle help callback", error=str(e))
            await callback.answer("❌ Terjadi kesalahan.")
    
    async def cmd_ask(self, message: Message):
        """Handle /ask command for AI assistant"""
        try:
            user = await self._get_or_create_user(message.from_user)
            
            # Parse the question from the command
            command_parts = message.text.split(' ', 1) if message.text else []
            if len(command_parts) < 2:
                help_text = """
❓ <b>AI Assistant</b>

Gunakan: <code>/ask [pertanyaan Anda]</code>

Contoh:
• <code>/ask Bagaimana cara trading Bitcoin?</code>
• <code>/ask Apa itu DCA?</code>
• <code>/ask Kapan waktu yang tepat untuk buy?</code>
• <code>/ask Analisa chart BTC hari ini</code>

💡 AI akan membantu menjawab pertanyaan seputar trading dan cryptocurrency.
"""
                await message.answer(help_text, parse_mode="HTML")
                return
            
            question = command_parts[1].strip()
            
            # Show typing indicator
            await message.answer("🤔 AI sedang memikirkan jawaban...")
            
            # Generate AI response
            response = await self._generate_ai_response(question, user)
            
            await message.answer(response, parse_mode="HTML")
            
        except Exception as e:
            logger.error("Failed to handle ask command", error=str(e))
            await message.answer("❌ Maaf, terjadi kesalahan dalam AI assistant.")
    
    async def cmd_autotrading(self, message: Message):
        """Handle /autotrading command"""
        try:
            user = await self._get_or_create_user(message.from_user)
            
            if not getattr(user, 'indodax_api_key', None):
                await message.answer("❌ Anda belum terdaftar. Gunakan /daftar untuk mendaftar.")
                return
            
            # Check current auto-trading status
            auto_trading_enabled = getattr(user, 'auto_trading_enabled', False) or False
            current_status = "✅ Aktif" if auto_trading_enabled else "❌ Tidak Aktif"
            
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="🟢 Aktifkan" if not auto_trading_enabled else "🔴 Nonaktifkan", 
                                       callback_data="autotrading_toggle"),
                ],
                [
                    InlineKeyboardButton(text="⚙️ Pengaturan", callback_data="autotrading_settings"),
                    InlineKeyboardButton(text="📊 Status", callback_data="autotrading_status")
                ]
            ])
            
            auto_trade_amount = getattr(user, 'auto_trade_amount', None) or 100000
            
            auto_trading_text = f"""
🤖 <b>Auto Trading</b>

Status: {current_status}
Jumlah per Trade: {format_currency(float(auto_trade_amount))} IDR

<b>Fitur Auto Trading:</b>
• Eksekusi otomatis berdasarkan sinyal AI
• Risk management terintegrasi
• Stop loss dan take profit otomatis
• Monitoring 24/7

Pilih opsi di bawah:
"""
            
            await message.answer(auto_trading_text, reply_markup=keyboard)
            
        except Exception as e:
            logger.error("Failed to handle autotrading command", error=str(e))
            await message.answer("❌ Terjadi kesalahan saat mengakses auto trading.")
    
    async def cmd_dca(self, message: Message):
        """Handle /dca command"""
        try:
            user = await self._get_or_create_user(message.from_user)
            
            if not getattr(user, 'indodax_api_key', None):
                await message.answer("❌ Anda belum terdaftar. Gunakan /daftar untuk mendaftar.")
                return
            
            # Get user settings
            db = get_db()
            try:
                settings_record = db.query(UserSettings).filter(UserSettings.user_id == user.id).first()
                if not settings_record:
                    settings_record = UserSettings(user_id=user.id)
                    db.add(settings_record)
                    db.commit()
                
                dca_enabled = getattr(settings_record, 'dca_enabled', False) or False
                dca_status = "✅ Aktif" if dca_enabled else "❌ Tidak Aktif"
                dca_amount_value = getattr(settings_record, 'dca_amount', None) or 100000
                dca_amount = format_currency(float(dca_amount_value))
                dca_interval = getattr(settings_record, 'dca_interval', 'daily') or 'daily'
                dca_interval = dca_interval.capitalize()
                
            finally:
                db.close()
            
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="🟢 Aktifkan DCA" if not dca_enabled else "🔴 Nonaktifkan DCA", 
                                       callback_data="dca_toggle"),
                ],
                [
                    InlineKeyboardButton(text="💰 Set Jumlah", callback_data="dca_amount"),
                    InlineKeyboardButton(text="📅 Set Interval", callback_data="dca_interval")
                ],
                [
                    InlineKeyboardButton(text="📊 Riwayat DCA", callback_data="dca_history")
                ]
            ])
            
            dca_text = f"""
💰 <b>Dollar Cost Averaging (DCA)</b>

Status: {dca_status}
Jumlah: {dca_amount} IDR
Interval: {dca_interval}

<b>Tentang DCA:</b>
DCA adalah strategi investasi berkala dengan jumlah tetap, mengurangi dampak volatilitas harga.

<b>Keuntungan:</b>
• Mengurangi risiko timing
• Disiplin investasi jangka panjang
• Cocok untuk pemula
• Otomatis dan konsisten

Pilih opsi di bawah:
"""
            
            await message.answer(dca_text, reply_markup=keyboard)
            
        except Exception as e:
            logger.error("Failed to handle DCA command", error=str(e))
            await message.answer("❌ Terjadi kesalahan saat mengakses DCA.")
    
    async def cmd_backtest(self, message: Message):
        """Handle /backtest command"""
        try:
            user = await self._get_or_create_user(message.from_user)
            
            # Parse command arguments
            command_parts = message.text.split() if message.text else []
            
            if len(command_parts) < 4:
                # Get available pairs from API for help text
                try:
                    from core.indodax_api import indodax_api
                    pairs_data = await indodax_api.get_pairs()
                    
                    # Extract traded currencies (remove _idr suffix)
                    available_pairs = []
                    for pair in pairs_data[:15]:  # Show first 15 pairs in help
                        if pair.get('ticker_id', '').endswith('_idr'):
                            symbol = pair['ticker_id'].replace('_idr', '')
                            available_pairs.append(symbol.upper())
                    
                    pairs_text = ' • '.join([f"<code>{p.lower()}</code>" for p in available_pairs])
                    more_pairs_count = len(pairs_data) - 15 if len(pairs_data) > 15 else 0
                    
                except Exception as e:
                    logger.error("Failed to fetch pairs for help", error=str(e))
                    pairs_text = "<code>btc</code> • <code>eth</code> • <code>ada</code> • <code>sol</code> • dll"
                    more_pairs_count = 0
                
                help_text = f"""
📊 <b>Backtesting</b>

Gunakan: <code>/backtest [pair] [strategy] [period] [modal]</code>

<b>Pairs tersedia:</b>
{pairs_text}
{f"<i>...dan {more_pairs_count} pair lainnya</i>" if more_pairs_count > 0 else ""}

<b>Strategies:</b>
• <code>ai_signals</code> - Backtest AI signals
• <code>dca</code> - Backtest DCA strategy
• <code>buy_and_hold</code> - Buy and hold strategy

<b>Periods:</b>
• <code>7d</code> - 7 hari
• <code>30d</code> - 30 hari  
• <code>90d</code> - 90 hari
• <code>Xd</code> - Custom (X hari, 7-365)

<b>Modal (opsional):</b>
• Jumlah modal awal dalam IDR (default: 1.000.000 IDR)
• Contoh: 5000000 untuk 5 juta IDR

<b>Contoh:</b>
<code>/backtest btc ai_signals 30d</code>
<code>/backtest eth buy_and_hold 90d 5000000</code>
<code>/backtest sol dca 7d 10000000</code>
<code>/backtest btc ai_signals 180d</code> (6 bulan)

⚠️ <i>Parameter pair, strategy, dan period wajib diisi!</i>
"""
                await message.answer(help_text, parse_mode="HTML")
                return
            
            pair_symbol = command_parts[1].lower()
            strategy = command_parts[2].lower()
            period = command_parts[3].lower()
            
            # Check for optional initial investment parameter
            initial_balance = 1000000  # Default: 1 million IDR
            if len(command_parts) >= 5:
                try:
                    custom_balance = float(command_parts[4])
                    if custom_balance >= 10000:  # Ensure it's at least 10,000 IDR
                        initial_balance = custom_balance
                    else:
                        await message.answer("⚠️ Modal minimum adalah 10.000 IDR. Menggunakan nilai default 1.000.000 IDR.")
                except ValueError:
                    await message.answer("⚠️ Format modal tidak valid. Menggunakan nilai default 1.000.000 IDR.")
            
            # Validate pair against API data
            try:
                from core.indodax_api import indodax_api
                pairs_data = await indodax_api.get_pairs()
                
                # Extract valid pair symbols
                valid_pairs = []
                pair_found = False
                
                for pair in pairs_data:
                    if pair.get('ticker_id', '').endswith('_idr'):
                        symbol = pair['ticker_id'].replace('_idr', '')
                        valid_pairs.append(symbol.lower())
                        if symbol.lower() == pair_symbol:
                            pair_found = True
                
                if not pair_found:
                    # Show available pairs in chunks
                    pairs_list = [p.upper() for p in valid_pairs[:20]]  # Show first 20
                    remaining = len(valid_pairs) - 20 if len(valid_pairs) > 20 else 0
                    
                    pairs_str = ', '.join(pairs_list)
                    if remaining > 0:
                        pairs_str += f" (+{remaining} lainnya)"
                    
                    await message.answer(f"❌ Pair {pair_symbol.upper()} tidak tersedia.\n\n🪙 <b>Pairs tersedia:</b>\n{pairs_str}", parse_mode="HTML")
                    return
                    
            except Exception as e:
                logger.error("Failed to validate pair", error=str(e))
                await message.answer("❌ Gagal memvalidasi pair. Silakan coba lagi.")
                return
            
            # Convert to pair_id format
            pair_id = f"{pair_symbol}_idr"
            
            # Validate strategy
            valid_strategies = ["ai_signals", "dca", "buy_and_hold"]
            if strategy not in valid_strategies:
                await message.answer(f"❌ Strategy {strategy} tidak didukung. Gunakan salah satu: {', '.join(valid_strategies)}")
                return
            
            # Validate period
            valid_periods = ["7d", "30d", "90d"]
            
            # Support custom period in format Xd where X is number of days (7-365)
            custom_period = False
            days_to_backtest = 0
            if period.endswith("d") and len(period) > 1:
                try:
                    days_to_backtest = int(period[:-1])
                    if 7 <= days_to_backtest <= 365:  # Support custom period from 7 days to 1 year
                        custom_period = True
                    else:
                        await message.answer(f"❌ Period harus antara 7 dan 365 hari. Gunakan format seperti '30d' untuk 30 hari atau pilih salah satu: {', '.join(valid_periods)}")
                        return
                except ValueError:
                    pass
            
            if not custom_period and period not in valid_periods:
                await message.answer(f"❌ Period {period} tidak didukung. Gunakan salah satu: {', '.join(valid_periods)} atau custom period (misal: 60d, 120d, 365d)")
                return
            
            await message.answer(f"🔄 Menjalankan backtest {strategy} untuk {pair_symbol.upper()}... Mohon tunggu sebentar.")
            
            # Run backtest with improved AI/ML capabilities
            from core.backtester import Backtester
            from datetime import datetime, timedelta
            import asyncio
            import os  # Ensure os is imported for file operations
            from aiogram.types import FSInputFile  # Ensure FSInputFile is imported for sending files
            import asyncio
            
            # Show loading message to improve UX
            # Format period display
            period_display = f"{days_to_backtest} hari" if custom_period else period
            
            status_message = await message.answer(
                "🔄 <b>Memproses backtest</b>\n\n"
                f"<b>Pair:</b> {pair_symbol.upper()}/IDR\n"
                f"<b>Strategy:</b> {strategy}\n"
                f"<b>Period:</b> {period_display}\n"
                f"<b>Modal Awal:</b> {format_currency(initial_balance)} IDR\n\n"
                "⏳ Mengumpulkan data historis..."
            )
            
            backtester = Backtester()
            
            # Calculate dates based on period
            end_date = datetime.now()
            if custom_period:
                # Use the custom number of days
                start_date = end_date - timedelta(days=days_to_backtest)
            elif period == "7d":
                start_date = end_date - timedelta(days=7)
            elif period == "30d":
                start_date = end_date - timedelta(days=30)
            elif period == "90d":
                start_date = end_date - timedelta(days=90)
            else:
                start_date = end_date - timedelta(days=30)  # default
                
            # Update status
            await status_message.edit_text(
                "🔄 <b>Memproses backtest</b>\n\n"
                f"<b>Pair:</b> {pair_symbol.upper()}/IDR\n"
                f"<b>Strategy:</b> {strategy}\n"
                f"<b>Period:</b> {period_display}\n"
                f"<b>Modal Awal:</b> {format_currency(initial_balance)} IDR\n\n"
                "⏳ Menjalankan simulasi trading..."
            )
            
            # Set confidence threshold based on strategy and period length
            # Using higher thresholds to prioritize accuracy over quantity of signals
            confidence_threshold = 0.7  # Increased base threshold for higher accuracy
            
            if strategy == "ai_signals":
                # For higher win rate, we'll use higher confidence thresholds
                days_count = days_to_backtest if custom_period else int(period.replace('d', ''))
                
                # Default to high confidence for all periods
                confidence_threshold = 0.72  # Higher baseline for all periods
                
                # Only slightly adjust threshold based on period length
                if days_count > 90:  # For periods longer than 90 days
                    confidence_threshold = 0.68  # Still high but slightly lower for longer periods
                
                logger.info(f"Using high precision confidence threshold {confidence_threshold} for {days_count} day period")
                logger.info(f"Prioritizing signal quality over quantity for higher win rate")
            elif strategy == "lstm_prediction":
                confidence_threshold = 0.75  # Higher threshold for LSTM predictions as well
                
            # We'll use the existing signal generator but apply optimized parameters
            # in the backtester's configuration for higher accuracy
            
            # Log our approach
            logger.info(f"Running backtest with high precision mode using confidence threshold {confidence_threshold}")
            logger.info(f"Optimized for higher win rate and profit per trade")
            
            # Run the backtest with appropriate parameters and enhanced signal generator
            results = await backtester.run_backtest(
                pair_id=pair_id,
                start_date=start_date,
                end_date=end_date,
                initial_balance=initial_balance,  # Use the custom or default initial investment
                strategy=strategy,
                min_confidence=confidence_threshold,
                signal_generator=self.signal_generator  # Pass the upgraded signal generator with high threshold
            )
            
            if results:
                # Update status while generating report
                await status_message.edit_text(
                    "🔄 <b>Memproses backtest</b>\n\n"
                    f"<b>Pair:</b> {pair_symbol.upper()}/IDR\n"
                    f"<b>Strategy:</b> {strategy}\n"
                    f"<b>Period:</b> {period_display}\n"
                    f"<b>Modal Awal:</b> {format_currency(initial_balance)} IDR\n\n"
                    "⏳ Menghasilkan laporan dan visualisasi..."
                )
                
                # Generate performance chart if matplotlib is available
                chart_path = await self._generate_backtest_chart(results, pair_symbol, strategy, period, user.id)
                
                # Generate strategy-specific insights
                strategy_insights = ""
                if strategy == "ai_signals":
                    # Extract confidence scores and accuracy from trades
                    if results.trades:
                        avg_confidence = sum(trade.get('confidence', 0) for trade in results.trades if 'confidence' in trade) / len(results.trades)
                        profitable_signals = len([t for t in results.trades if t.get('pnl', 0) > 0 and t.get('confidence', 0) > 0.7])
                        high_conf_signals = len([t for t in results.trades if t.get('confidence', 0) > 0.7])
                        high_conf_accuracy = profitable_signals / high_conf_signals if high_conf_signals > 0 else 0
                        
                        # Calculate days per trade average - use custom period or parse from period string
                        if custom_period:
                            days_analyzed = days_to_backtest
                        else:
                            days_analyzed = int(period[:-1]) if period.endswith('d') and period[:-1].isdigit() else 30
                            
                        avg_days_between_trades = days_analyzed / results.total_trades if results.total_trades > 0 else 0
                        
                        # Calculate additional advanced metrics
                        avg_profit_per_winning_trade = 0
                        if results.winning_trades > 0:
                            winning_trades = [t for t in results.trades if t.get('pnl', 0) > 0]
                            avg_profit_per_winning_trade = sum(t.get('return_percent', 0) for t in winning_trades) / results.winning_trades
                            
                        # More comprehensive insights with emphasis on quality over quantity
                        strategy_insights = f"""
<b>AI Signal Insights:</b>
• Average Confidence: {avg_confidence:.2f}
• High Confidence Signal Accuracy: {high_conf_accuracy*100:.1f}%
• Best Performing Indicator: {self._get_best_indicator(results.trades)}
• Trading Frequency: 1 trade per {avg_days_between_trades:.1f} hari
• Signal Generation Rate: {(results.total_trades/days_analyzed*30):.1f} signals/bulan
• Profit per winning trade: {avg_profit_per_winning_trade:.2f}%

<b>Mode Presisi Tinggi:</b>
ℹ️ Hasil ini menggunakan threshold confidence tinggi (0.68-0.72) untuk mengutamakan kualitas sinyal daripada kuantitas.
"""
                elif strategy == "lstm_prediction":
                    if results.trades:
                        avg_prediction = sum(abs(trade.get('predicted_change', 0)) for trade in results.trades if 'predicted_change' in trade) / len(results.trades)
                        
                        strategy_insights = f"""
<b>LSTM Model Insights:</b>
• Avg Predicted Change: {avg_prediction*100:.2f}%
• Prediction Accuracy: {results.win_rate:.1f}%
• Best Prediction Horizon: {self._get_best_prediction_horizon(results.trades)}
"""
                
                # Format main result text with strategy-specific details
                if strategy == "dca":
                    # Count DCA purchases
                    dca_purchases = len([t for t in results.trades if t.get('type') == 'dca_buy'])
                    
                    # Get average cost basis if available
                    avg_cost = 0
                    final_price = 0
                    for trade in results.trades:
                        if trade.get('type') == 'final_sale':
                            final_price = trade.get('price', 0)
                        if trade.get('type') == 'dca_buy' and 'avg_cost_basis' in trade:
                            avg_cost = trade.get('avg_cost_basis', 0)
                    
                    # DCA-specific result text
                    result_text = f"""
📊 <b>Hasil Backtest DCA</b>

<b>Strategy:</b> {strategy.upper()}
<b>Period:</b> {period_display}
<b>Pair:</b> {pair_symbol.upper()}/IDR

<b>Performance:</b>
• Total Return: <b>{results.total_return_percent:.2f}%</b>
• Sharpe Ratio: {results.sharpe_ratio:.2f}
• Max Drawdown: {results.max_drawdown:.2f}%
• DCA Purchases: {dca_purchases}

<b>DCA Metrics:</b>
• Avg Purchase Price: {format_currency(avg_cost)}
• Final Market Price: {format_currency(final_price)}
• Price Change: {((final_price/avg_cost)-1)*100:.2f}% {("📈" if final_price > avg_cost else "📉")}

<b>Balance:</b>
• Initial Investment: {format_currency(results.initial_balance)} IDR
• Final Value: <b>{format_currency(results.final_balance)} IDR</b>
• Profit/Loss: {format_currency(results.total_return)} IDR {("✅" if results.total_return > 0 else "❌")}

⚠️ <i>Past performance does not guarantee future results.</i>
"""
                else:
                    # Standard result text for other strategies
                    result_text = f"""
📊 <b>Hasil Backtest</b>

<b>Strategy:</b> {strategy.upper()}
<b>Period:</b> {period_display}
<b>Pair:</b> {pair_symbol.upper()}/IDR

<b>Performance:</b>
• Total Return: <b>{results.total_return_percent:.2f}%</b>
• Sharpe Ratio: {results.sharpe_ratio:.2f}
• Max Drawdown: {results.max_drawdown:.2f}%
• Win Rate: {results.win_rate:.1f}%

<b>Trades:</b>
• Total Trades: {results.total_trades}
• Profitable: {results.winning_trades}
• Loss: {results.losing_trades}

<b>Balance:</b>
• Initial: {format_currency(results.initial_balance)} IDR
• Final: <b>{format_currency(results.final_balance)} IDR</b>
• Profit/Loss: {format_currency(results.total_return)} IDR
{strategy_insights}
⚠️ <i>Past performance does not guarantee future results.</i>
"""
                # Send chart if available
                if chart_path and os.path.exists(chart_path):
                    await self.bot.send_photo(
                        message.chat.id,
                        FSInputFile(chart_path),
                        caption=result_text,
                    )
                    # Clean up
                    try:
                        os.remove(chart_path)
                    except:
                        pass
                else:
                    await status_message.delete()
                    await message.answer(result_text)
                    
                # Store backtest result in user session data
                backtest_data = {
                    "pair": pair_symbol,
                    "strategy": strategy,
                    "period": period,
                    "return": results.total_return_percent,
                    "timestamp": datetime.now().timestamp()
                }                # Create trade button if result is profitable or show warning if not enough trades
                keyboard = InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(
                        text="🔄 Run Another Backtest",
                        callback_data=f"backtest_new"
                    )
                ]])
                    
                # Add apply strategy button only for profitable strategies
                if results.total_return_percent > 0:
                    keyboard.inline_keyboard[0].append(
                        InlineKeyboardButton(
                            text="💹 Apply Strategy",
                            callback_data=f"backtest_apply_{pair_id}_{strategy}"
                        )
                    )
                
                # Generate AI feedback with statistical validity assessment
                min_trades_for_validity = 10  # Lowered minimum trades for high-accuracy mode
                
                if results.total_trades < min_trades_for_validity:
                    insight_text = (
                        f"🤖 <b>AI Recommendation</b>\n\n"
                        f"ℹ️ <b>Mode Presisi Tinggi:</b> {results.total_trades} trades selama {period_display}.\n"
                        f"Strategi ini menggunakan filter ketat untuk mendapatkan sinyal berkualitas tinggi.\n\n"
                    )
                    
                    if results.total_trades > 0:
                        # Calculate average profit per trade for high-precision mode assessment
                        avg_profit_per_trade = results.total_return_percent / results.total_trades if results.total_trades > 0 else 0
                        win_rate_text = f"Win Rate: {results.win_rate:.1f}%" if results.win_rate > 0 else ""
                        
                        insight_text += (
                            f"<b>Analisa Kualitas:</b>\n"
                            f"• Profit rata-rata per trade: {avg_profit_per_trade:.2f}%\n"
                            f"• {win_rate_text}\n"
                            f"• Strategi {strategy} untuk {pair_symbol.upper()} " + 
                            (f"menunjukkan hasil positif {results.total_return_percent:.1f}% dengan sinyal berkualitas tinggi." 
                             if results.total_return_percent > 0 else 
                             f"tidak optimal dengan return {results.total_return_percent:.1f}% meski menggunakan filter ketat.")
                        )
                        
                        if results.win_rate >= 60:
                            insight_text += "\n\n✅ <b>Rekomendasi:</b> Win rate tinggi menunjukkan strategi ini dapat diandalkan meskipun jumlah trade sedikit."
                        elif results.total_trades < 5:
                            insight_text += "\n\n⚠️ <b>Catatan:</b> Jumlah trades sangat sedikit. Pertimbangkan untuk menguji periode lebih panjang."
                    else:
                        insight_text += "Tidak ada trade yang tereksekusi pada periode ini karena standar kualitas sinyal yang sangat tinggi. Coba periode yang lebih panjang."
                else:
                    # Calculate additional metrics for detailed analysis
                    avg_profit_per_trade = results.total_return_percent / results.total_trades if results.total_trades > 0 else 0
                    successful_trades_percentage = f"{(results.winning_trades / results.total_trades * 100):.1f}%" if results.total_trades > 0 else "0%"
                    
                    if results.total_return_percent > 0:
                        # For positive results, emphasize quality metrics
                        insight_text = (
                            f"🤖 <b>AI Recommendation</b>\n\n"
                            f"✅ Strategi {strategy} menunjukkan hasil positif untuk {pair_symbol.upper()}.\n\n"
                            f"<b>Analisa Kualitas:</b>\n"
                            f"• Total Return: <b>{results.total_return_percent:.1f}%</b>\n"
                            f"• Win Rate: <b>{results.win_rate:.1f}%</b>\n"
                            f"• Profit rata-rata per trade: <b>{avg_profit_per_trade:.2f}%</b>\n"
                            f"• Trades menguntungkan: {successful_trades_percentage}\n\n"
                        )
                        
                        # Add specific recommendations based on metrics
                        if results.win_rate >= 70:
                            insight_text += f"⭐ <b>Sangat Direkomendasikan:</b> Win rate yang tinggi ({results.win_rate:.1f}%) menunjukkan strategi ini sangat dapat diandalkan."
                        elif results.win_rate >= 50 and avg_profit_per_trade > 1.0:
                            insight_text += f"✅ <b>Direkomendasikan:</b> Win rate baik dengan profit per trade yang menarik ({avg_profit_per_trade:.2f}%)."
                        else:
                            insight_text += f"👍 <b>Dapat dipertimbangkan</b> untuk trading otomatis dengan monitoring berkala."
                    else:
                        # For negative results, provide detailed analysis why it failed
                        insight_text = (
                            f"🤖 <b>AI Recommendation</b>\n\n"
                            f"⚠️ Strategi {strategy} menunjukkan hasil negatif untuk {pair_symbol.upper()}.\n\n"
                            f"<b>Analisa:</b>\n"
                            f"• Total Return: <b>{results.total_return_percent:.1f}%</b>\n"
                            f"• Win Rate: {results.win_rate:.1f}%\n"
                            f"• Loss rata-rata per trade: {abs(avg_profit_per_trade):.2f}%\n"
                            f"• Total trades: {results.total_trades}\n\n"
                            f"❌ <b>Tidak Direkomendasikan:</b> Strategi ini tidak optimal untuk {pair_symbol.upper()} pada periode {period_display}."
                        )
                        
                        # Add specific improvement suggestions
                        if results.win_rate < 40:
                            insight_text += f"\n\n💡 <b>Saran:</b> Win rate rendah ({results.win_rate:.1f}%) menunjukkan perlu penyesuaian parameter teknikal."
                        elif abs(avg_profit_per_trade) > 2.0:
                            insight_text += f"\n\n💡 <b>Saran:</b> Loss per trade tinggi ({abs(avg_profit_per_trade):.2f}%). Pertimbangkan strategi exit yang lebih konservatif."
                
                await message.answer(insight_text, reply_markup=keyboard)
            else:
                await status_message.delete()
                await message.answer("❌ Gagal menjalankan backtest. Silakan coba lagi nanti.")
            
        except Exception as e:
            logger.error("Failed to handle backtest command", error=str(e))
            await message.answer("❌ Terjadi kesalahan saat menjalankan backtest.")

    def _get_best_indicator(self, trades):
        """Analyze trades to find which technical indicator performed best"""
        if not trades:
            return "RSI"
            
        # Count profitable trades by indicator
        indicator_performance = {
            "RSI": 0,
            "MACD": 0,
            "Moving Averages": 0,
            "Bollinger Bands": 0,
            "Volume": 0
        }
        
        indicator_usage = {k: 0 for k in indicator_performance.keys()}
        
        for trade in trades:
            # Skip trades without indicators data
            if 'indicators' not in trade:
                continue
                
            indicators = trade.get('indicators', {})
            is_profitable = trade.get('pnl', 0) > 0
            
            # Check which indicators likely triggered this trade
            if 'rsi' in indicators:
                rsi = indicators['rsi']
                if (rsi < 30 and trade['type'] == 'buy') or (rsi > 70 and trade['type'] == 'sell'):
                    indicator_usage["RSI"] += 1
                    if is_profitable:
                        indicator_performance["RSI"] += 1
            
            if 'macd' in indicators and 'macd_signal' in indicators:
                macd = indicators['macd']
                macd_signal = indicators['macd_signal']
                if (macd > macd_signal and trade['type'] == 'buy') or (macd < macd_signal and trade['type'] == 'sell'):
                    indicator_usage["MACD"] += 1
                    if is_profitable:
                        indicator_performance["MACD"] += 1
            
            if 'sma_10' in indicators and 'sma_20' in indicators:
                sma_10 = indicators['sma_10']
                sma_20 = indicators['sma_20']
                if (sma_10 > sma_20 and trade['type'] == 'buy') or (sma_10 < sma_20 and trade['type'] == 'sell'):
                    indicator_usage["Moving Averages"] += 1
                    if is_profitable:
                        indicator_performance["Moving Averages"] += 1
            
            if 'bb_upper' in indicators and 'bb_lower' in indicators:
                bb_upper = indicators['bb_upper']
                bb_lower = indicators['bb_lower']
                price = trade.get('price', 0)
                if (price < bb_lower and trade['type'] == 'buy') or (price > bb_upper and trade['type'] == 'sell'):
                    indicator_usage["Bollinger Bands"] += 1
                    if is_profitable:
                        indicator_performance["Bollinger Bands"] += 1
            
            if 'volume_ratio' in indicators:
                volume_ratio = indicators['volume_ratio']
                if volume_ratio > 1.2:
                    indicator_usage["Volume"] += 1
                    if is_profitable:
                        indicator_performance["Volume"] += 1
        
        # Calculate success rate for each indicator
        success_rates = {}
        for indicator, wins in indicator_performance.items():
            if indicator_usage[indicator] > 0:
                success_rates[indicator] = wins / indicator_usage[indicator]
            else:
                success_rates[indicator] = 0
        
        # Find indicator with highest success rate
        best_indicator = max(success_rates.items(), key=lambda x: x[1]) if success_rates else ("RSI", 0)
        
        return best_indicator[0]
    
    def _get_best_prediction_horizon(self, trades):
        """Analyze LSTM trades to find which prediction horizon worked best"""
        if not trades:
            return "1-3 days"
            
        # Count successful predictions at different horizons
        horizons = {
            "1-3 days": {"correct": 0, "total": 0},
            "4-7 days": {"correct": 0, "total": 0},
            "8+ days": {"correct": 0, "total": 0}
        }
        
        for i in range(len(trades) - 1):
            current_trade = trades[i]
            
            # Skip if not buy trade
            if current_trade['type'] != 'buy':
                continue
            
            # Find corresponding sell trade
            sell_trade = None
            for j in range(i+1, len(trades)):
                if trades[j]['type'] == 'sell':
                    sell_trade = trades[j]
                    break
            
            if not sell_trade:
                continue
                
            # Calculate days between trades
            buy_date = current_trade['date']
            sell_date = sell_trade['date']
            days_held = (sell_date - buy_date).days
            
            # Determine horizon category
            if days_held <= 3:
                horizon = "1-3 days"
            elif days_held <= 7:
                horizon = "4-7 days"
            else:
                horizon = "8+ days"
                
            # Track performance
            horizons[horizon]["total"] += 1
            if sell_trade.get('pnl', 0) > 0:
                horizons[horizon]["correct"] += 1
        
        # Find best horizon
        best_horizon = "1-3 days"  # default
        best_accuracy = 0
        
        for horizon, stats in horizons.items():
            if stats["total"] > 0:
                accuracy = stats["correct"] / stats["total"]
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_horizon = horizon
        
        return best_horizon

    async def _generate_ai_response(self, question: str, user) -> str:
        """Generate AI response for user questions"""
        try:
            # Simplified AI response generator
            # Dalam implementasi yang lebih lengkap, ini bisa menggunakan OpenAI API atau model lainnya
            
            question_lower = question.lower()
            
            # Trading-related questions
            if any(keyword in question_lower for keyword in ["bitcoin", "btc", "trading", "buy", "sell"]):
                return """
🤖 <b>AI Trading Assistant</b>

<b>Tips Trading Bitcoin:</b>
• Selalu lakukan analisis teknikal sebelum trading
• Gunakan stop loss untuk membatasi kerugian
• Jangan investasi lebih dari yang Anda mampu kehilangan
• Diversifikasi portfolio Anda

<b>Untuk analisis real-time:</b>
• Gunakan /signal untuk mendapatkan sinyal AI
• Gunakan /backtest untuk tes strategi
• Monitor portfolio dengan /portfolio

💡 <i>Gunakan fitur auto-trading untuk eksekusi otomatis berdasarkan sinyal AI</i>
"""
            
            elif any(keyword in question_lower for keyword in ["dca", "dollar cost averaging"]):
                return """
🤖 <b>Dollar Cost Averaging (DCA)</b>

<b>Apa itu DCA?</b>
DCA adalah strategi investasi berkala dengan jumlah tetap, mengurangi dampak volatilitas harga.

<b>Keuntungan DCA:</b>
• Mengurangi risiko timing market
• Disiplin investasi jangka panjang
• Cocok untuk pemula
• Otomatis dan konsisten

<b>Cara menggunakan:</b>
Gunakan /dca untuk mengatur DCA otomatis di bot ini.

💡 <i>DCA sangat efektif untuk investasi jangka panjang di cryptocurrency</i>
"""
            
            elif any(keyword in question_lower for keyword in ["analisa", "chart", "teknikal"]):
                return """
🤖 <b>Analisis Teknikal</b>

<b>Indikator yang digunakan bot:</b>
• RSI (Relative Strength Index)
• MACD (Moving Average Convergence Divergence)
• Bollinger Bands
• Moving Averages
• Volume analysis

<b>Signal AI menggunakan:</b>
• Machine Learning models
• Ensemble methods (XGBoost, LightGBM)
• Real-time market data

<b>Untuk analisis langsung:</b>
Gunakan /signal [pair] untuk analisis real-time

💡 <i>Bot menggunakan 15+ indikator teknikal untuk prediksi akurat</i>
"""
            
            elif any(keyword in question_lower for keyword in ["kapan", "timing", "waktu"]):
                return """
🤖 <b>Market Timing</b>

<b>Prinsip timing yang baik:</b>
• Beli saat RSI < 30 (oversold)
• Jual saat RSI > 70 (overbought)
• Perhatikan volume trading
• Ikuti trend jangka panjang

<b>Gunakan AI untuk timing:</b>
• /signal untuk prediksi harga
• /backtest untuk tes strategi
• Auto-trading untuk eksekusi otomatis

<b>Tips penting:</b>
• Jangan panic selling
• HODL untuk jangka panjang
• DCA untuk mengurangi risiko timing

💡 <i>AI bot dapat membantu timing dengan akurasi 70%+</i>
"""
            
            else:
                # Generic response
                return """
🤖 <b>AI Trading Assistant</b>

Maaf, saya belum bisa menjawab pertanyaan spesifik tersebut.

<b>Yang bisa saya bantu:</b>
• Analisis trading dan cryptocurrency
• Strategi investasi DCA
• Timing pasar dan analisis teknikal
• Tips trading Bitcoin dan altcoin

<b>Contoh pertanyaan:</b>
• "Bagaimana cara trading Bitcoin?"
• "Apa itu DCA?"
• "Kapan waktu yang tepat untuk buy?"
• "Analisa chart BTC hari ini"

💡 <i>Gunakan /help untuk melihat semua fitur bot</i>
"""
                
        except Exception as e:
            logger.error("Failed to generate AI response", error=str(e))
            return "❌ Maaf, terjadi kesalahan dalam AI assistant. Silakan coba lagi."

    async def _generate_backtest_chart(self, results, pair_symbol, strategy, period, user_id):
        """Generate backtest chart for visualization"""
        chart_path = None
        
        # Check if matplotlib is available
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, skipping chart generation")
            return None
            
        try:
            # Local imports to avoid scope issues
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.ticker import FuncFormatter
            import numpy as np
            import os
            from datetime import datetime
            import matplotlib.ticker as mtick  # For percentage formatting
            
            # Create equity curve chart
            plt.figure(figsize=(12, 8))
            
            # Extract data from equity curve
            dates = [point['date'] for point in results.equity_curve]
            equity = [point['equity'] for point in results.equity_curve]
            
            # Extract balance and position values if available
            balance = [point.get('balance', 0) for point in results.equity_curve]
            position_value = [point.get('position_value', 0) for point in results.equity_curve]
            
            # Create plot with clear titles and improved labels
            plt.plot(dates, equity, label='Total Equity (Cash + Crypto)', color='blue', linewidth=2.5)
            
            # Plot balance and position components for clearer understanding
            if strategy == "dca":
                plt.plot(dates, position_value, label='Crypto Holdings Value', color='purple', linestyle='-', linewidth=1.5, alpha=0.8)
                plt.plot(dates, balance, label='Remaining Cash', color='green', linestyle='-', linewidth=1.5, alpha=0.7)
            
            # Plot initial balance as horizontal line
            plt.axhline(y=results.initial_balance, color='red', linestyle='--', label='Initial Investment')
            
            # Add final balance marker with clearer annotation
            plt.scatter([dates[-1]], [equity[-1]], color='gold', marker='*', s=200, zorder=5, 
                        label=f'Final Value: {format_currency(results.final_balance)} IDR')
            
            # Add profit/loss percentage annotation at the end of chart
            plt_text = f"{results.total_return_percent:+.2f}%"
            plt.annotate(plt_text, xy=(dates[-1], equity[-1]), 
                        xytext=(10, 20), textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", fc='gold', alpha=0.7),
                        fontweight='bold', fontsize=12, color='black' if results.total_return_percent >= 0 else 'red')
            
            # Add buy/sell markers from trades with clearer positioning and visualization
            # Track the position in equity to place markers correctly
            if strategy == "dca":
                # For DCA, place markers on the total equity line at purchase/sale times
                for trade in results.trades:
                    trade_date = trade['date']
                    # Find the equity value at this date for proper marker placement
                    equity_at_date = next((point['equity'] for point in results.equity_curve if point['date'] == trade_date), None)
                    
                    if equity_at_date is not None:
                        if trade['type'] == 'dca_buy':
                            plt.scatter(trade_date, equity_at_date, color='green', marker='^', s=100, zorder=5,
                                       label='DCA Buy' if 'DCA Buy' not in plt.gca().get_legend_handles_labels()[1] else "")
                        elif trade['type'] == 'final_sale':
                            plt.scatter(trade_date, equity_at_date, color='red', marker='v', s=120, zorder=5,
                                       label='Final Sale' if 'Final Sale' not in plt.gca().get_legend_handles_labels()[1] else "")
            else:
                # For other strategies, use standard trade markers
                for trade in results.trades:
                    if trade['type'] == 'buy':
                        # Find the equity at this point for better visualization
                        trade_date = trade['date']
                        equity_at_date = next((point['equity'] for point in results.equity_curve if point['date'] == trade_date), None)
                        if equity_at_date:
                            plt.scatter(trade_date, equity_at_date, color='green', marker='^', s=100, zorder=5,
                                       label='Buy' if 'Buy' not in plt.gca().get_legend_handles_labels()[1] else "")
                    elif trade['type'] == 'sell' or trade['type'] == 'final_sale':
                        trade_date = trade['date']
                        equity_at_date = next((point['equity'] for point in results.equity_curve if point['date'] == trade_date), None)
                        if equity_at_date:
                            plt.scatter(trade_date, equity_at_date, color='red', marker='v', s=100, zorder=5,
                                       label='Sell' if 'Sell' not in plt.gca().get_legend_handles_labels()[1] else "")
            
            # Format chart with more descriptive title
            title_str = f'{pair_symbol.upper()}/IDR {strategy.upper()} Backtest ({period})'
            if strategy == "dca":
                title_str += f"\nDollar Cost Averaging - Return: {results.total_return_percent:.2f}%"
            elif strategy == "ai_signals":
                title_str += f"\nAI Signal-based Trading - Return: {results.total_return_percent:.2f}%"
            elif strategy == "buy_and_hold":
                title_str += f"\nBuy and Hold Strategy - Return: {results.total_return_percent:.2f}%"
            else:
                title_str += f"\nReturn: {results.total_return_percent:.2f}%"
                
            plt.title(title_str, fontsize=14)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Value (IDR)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Format y-axis as IDR currency with thousands separator
            def idr_formatter(x, pos):
                return f'Rp {x:,.0f}'
            plt.gca().yaxis.set_major_formatter(FuncFormatter(idr_formatter))
            
            # Format x-axis dates with better spacing
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
            plt.gcf().autofmt_xdate()  # Auto-format the date labels
            
            # Add annotations for key metrics with improved labeling
            props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
            
            # Create more detailed metrics text based on strategy
            if strategy == "dca":
                metrics_text = (f"Total Return: {results.total_return_percent:.2f}%\n"
                               f"Final Result: {'Profitable' if results.total_return_percent > 0 else 'Loss'}\n"
                               f"Max Drawdown: {results.max_drawdown:.2f}%\n"
                               f"Purchases: {len([t for t in results.trades if t['type'] == 'dca_buy'])}")
            else:
                metrics_text = (f"Total Return: {results.total_return_percent:.2f}%\n"
                               f"Win Rate: {results.win_rate:.1f}%\n"
                               f"Trades: {results.winning_trades} wins, {results.losing_trades} losses\n"
                               f"Max Drawdown: {results.max_drawdown:.2f}%")
            
            plt.annotate(metrics_text, xy=(0.02, 0.02), xycoords='axes fraction', fontsize=11,
                        bbox=props, verticalalignment='bottom')
            
            # Create a better legend
            handles, labels = plt.gca().get_legend_handles_labels()
            # Remove duplicate labels
            unique_labels = []
            unique_handles = []
            for handle, label in zip(handles, labels):
                if label not in unique_labels:
                    unique_labels.append(label)
                    unique_handles.append(handle)
            
            # Place legend at the top-right corner outside of the plot
            plt.legend(unique_handles, unique_labels, loc='upper left', framealpha=0.9, fontsize=10)
            
            # Ensure temp directory exists
            os.makedirs('temp', exist_ok=True)
            
            # Save chart to temporary file
            chart_path = f"temp/backtest_{user_id}_{int(datetime.now().timestamp())}.png"
            plt.tight_layout()
            plt.savefig(chart_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error("Failed to generate backtest chart", error=str(e))
            return None