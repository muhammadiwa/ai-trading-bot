"""
Telegram bot interface for the AI Trading Bot
"""
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
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
                    if (user.username != telegram_user.username or 
                        user.first_name != telegram_user.first_name or
                        user.last_name != telegram_user.last_name):
                        
                        user.username = telegram_user.username
                        user.first_name = telegram_user.first_name
                        user.last_name = telegram_user.last_name
                        user.updated_at = datetime.utcnow()
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
            
            welcome_text = self.messages.get_welcome_message(user.language)
            keyboard = create_main_keyboard(user.language)
            
            await message.answer(welcome_text, reply_markup=keyboard)
            
        except Exception as e:
            logger.error("Failed to handle start command", error=str(e))
            await message.answer("❌ Terjadi kesalahan. Silakan coba lagi.")
    
    async def cmd_help(self, message: Message):
        """Handle /help command"""
        try:
            user = await self._get_or_create_user(message.from_user)
            help_text = self.messages.get_help_message(user.language)
            
            await message.answer(help_text)
            
        except Exception as e:
            logger.error("Failed to handle help command", error=str(e))
            await message.answer("❌ Terjadi kesalahan. Silakan coba lagi.")
    
    async def cmd_register(self, message: Message, state: FSMContext):
        """Handle /register command"""
        try:
            user = await self._get_or_create_user(message.from_user)
            
            if user.indodax_api_key:
                await message.answer("✅ Anda sudah terdaftar dan terkoneksi dengan Indodax!")
                return
            
            register_text = self.messages.get_registration_message(user.language)
            await message.answer(register_text)
            
            await state.set_state(RegistrationStates.waiting_for_api_key)
            
        except Exception as e:
            logger.error("Failed to handle register command", error=str(e))
            await message.answer("❌ Terjadi kesalahan. Silakan coba lagi.")
    
    async def cmd_portfolio(self, message: Message):
        """Handle /portfolio command"""
        try:
            user = await self._get_or_create_user(message.from_user)
            
            if not user.indodax_api_key:
                await message.answer("❌ Anda belum terdaftar. Gunakan /daftar untuk mendaftar.")
                return
            
            # Get portfolio data
            portfolio_data = await self._get_portfolio_data(user)
            portfolio_text = self.messages.format_portfolio(portfolio_data, user.language)
            
            await message.answer(portfolio_text)
            
        except Exception as e:
            logger.error("Failed to handle portfolio command", error=str(e))
            await message.answer("❌ Terjadi kesalahan saat mengambil data portfolio.")
    
    async def cmd_balance(self, message: Message):
        """Handle /balance command"""
        try:
            user = await self._get_or_create_user(message.from_user)
            
            if not user.indodax_api_key:
                await message.answer("❌ Anda belum terdaftar. Gunakan /daftar untuk mendaftar.")
                return
            
            # Get balance from Indodax
            api_key = decrypt_api_key(str(user.indodax_api_key))
            secret_key = decrypt_api_key(str(user.indodax_secret_key))
            
            user_api = IndodaxAPI(api_key, secret_key)
            balance_data = await user_api.get_balance()
            
            logger.info("Balance data received", balance_data=balance_data, data_type=type(balance_data))
            
            balance_text = self.messages.format_balance(balance_data, user.language)
            await message.answer(balance_text)
            
        except Exception as e:
            logger.error("Failed to handle balance command", error=str(e))
            await message.answer("❌ Terjadi kesalahan saat mengambil data saldo.")
    
    async def cmd_signal(self, message: Message):
        """Handle /signal command"""
        try:
            user = await self._get_or_create_user(message.from_user)
            
            # Parse command for specific pair
            command_parts = message.text.split()
            pair_id = "btc_idr"  # default
            
            if len(command_parts) > 1:
                requested_pair = command_parts[1].lower()
                pair_id = f"{requested_pair}_idr"
            
            # Generate or get latest signal
            signal = await self.signal_generator.generate_signal(pair_id)
            
            if signal:
                signal_text = self.messages.format_signal(signal, user.language)
                keyboard = InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(
                        text="🔄 Update Signal",
                        callback_data=f"signal_update_{pair_id}"
                    ),
                    InlineKeyboardButton(
                        text="💹 Trade",
                        callback_data=f"trade_{pair_id}_{signal.signal_type}"
                    )
                ]])
                
                await message.answer(signal_text, reply_markup=keyboard)
            else:
                await message.answer("❌ Tidak dapat menghasilkan sinyal saat ini. Coba lagi nanti.")
            
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
            
            if not user.indodax_api_key:
                await message.answer("❌ Anda belum terdaftar. Gunakan /daftar untuk mendaftar.")
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
                keyboard = create_trading_keyboard(trade_type, user.language)
                trade_text = self.messages.get_trade_selection_message(trade_type, user.language)
                
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
            
            if not user.indodax_api_key:
                await message.answer("❌ Anda belum terdaftar. Gunakan /daftar untuk mendaftar.")
                return
            
            # Get open orders
            api_key = decrypt_api_key(str(user.indodax_api_key))
            secret_key = decrypt_api_key(str(user.indodax_secret_key))
            
            user_api = IndodaxAPI(api_key, secret_key)
            orders_data = await user_api.get_open_orders()
            
            orders_text = self.messages.format_orders(orders_data, user.language)
            await message.answer(orders_text)
            
        except Exception as e:
            logger.error("Failed to handle orders command", error=str(e))
            await message.answer("❌ Terjadi kesalahan saat mengambil data order.")
    
    async def cmd_settings(self, message: Message):
        """Handle /settings command"""
        try:
            user = await self._get_or_create_user(message.from_user)
            settings_keyboard = create_settings_keyboard(user.language)
            settings_text = self.messages.get_settings_message(user.language)
            
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
        if not is_admin_user(message.from_user.id):
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
        if not is_admin_user(message.from_user.id):
            return
        
        command_parts = message.text.split(" ", 1)
        if len(command_parts) < 2:
            await message.answer("Format: /broadcast <pesan>")
            return
        
        broadcast_message = command_parts[1]
        await self._broadcast_message(broadcast_message)
    
    async def cmd_stats(self, message: Message):
        """Handle /stats command"""
        if not is_admin_user(message.from_user.id):
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
                    user.indodax_api_key = encrypt_api_key(api_key)
                    user.indodax_secret_key = encrypt_api_key(secret_key)
                    
                    db = get_db()
                    try:
                        db.merge(user)
                        db.commit()
                    finally:
                        db.close()
                    
                    await state.clear()
                    await message.answer("✅ Pendaftaran berhasil! Akun Anda telah terhubung dengan Indodax.")
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
                    await callback.message.answer(f"💰 Masukkan jumlah {pair_id.split('_')[0].upper()} untuk jual:")
                    await state.set_state(TradingStates.entering_amount)
                    await state.update_data(trade_type=action, pair_id=pair_id)
            
            await callback.answer()
            
        except Exception as e:
            logger.error("Failed to handle trade callback", error=str(e))
            await callback.answer("❌ Terjadi kesalahan.")
    
    async def callback_signal(self, callback: CallbackQuery):
        """Handle signal callbacks"""
        try:
            data_parts = callback.data.split("_")
            action = data_parts[1]
            
            if action == "update":
                pair_id = data_parts[2] if len(data_parts) > 2 else "btc_idr"
                
                signal = await self.signal_generator.generate_signal(pair_id)
                if signal:
                    user = await self._get_or_create_user(callback.from_user)
                    signal_text = self.messages.format_signal(signal, user.language)
                    
                    await callback.message.edit_text(signal_text)
                else:
                    await callback.answer("❌ Tidak dapat memperbarui sinyal.")
            
            await callback.answer("🔄 Sinyal diperbarui!")
            
        except Exception as e:
            logger.error("Failed to handle signal callback", error=str(e))
            await callback.answer("❌ Terjadi kesalahan.")
    
    async def callback_settings(self, callback: CallbackQuery, state: FSMContext):
        """Handle settings callbacks"""
        try:
            data_parts = callback.data.split("_")
            action = data_parts[1]
            
            user = await self._get_or_create_user(callback.from_user)
            
            if action == "stoploss":
                await callback.message.answer("🛑 Masukkan nilai Stop Loss (%):")
                await state.set_state(SettingsStates.editing_stop_loss)
            elif action == "takeprofit":
                await callback.message.answer("🎯 Masukkan nilai Take Profit (%):")
                await state.set_state(SettingsStates.editing_take_profit)
            elif action == "maxamount":
                await callback.message.answer("💰 Masukkan jumlah maksimal trading (IDR):")
                await state.set_state(SettingsStates.editing_max_trade_amount)
            elif action == "language":
                keyboard = InlineKeyboardMarkup(inline_keyboard=[
                    [InlineKeyboardButton(text="🇮🇩 Bahasa Indonesia", callback_data="lang_id")],
                    [InlineKeyboardButton(text="🇺🇸 English", callback_data="lang_en")]
                ])
                await callback.message.edit_text("🌐 Pilih bahasa:", reply_markup=keyboard)
            
            await callback.answer()
            
        except Exception as e:
            logger.error("Failed to handle settings callback", error=str(e))
            await callback.answer("❌ Terjadi kesalahan.")

    async def callback_confirm(self, callback: CallbackQuery, state: FSMContext):
        """Handle confirmation callbacks"""
        try:
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
                    await callback.message.answer("❌ Data trading tidak valid.")
                
                await state.clear()
                
            elif action == "no":
                await callback.message.edit_text("❌ Trading dibatalkan.")
                await state.clear()
            
            await callback.answer()
            
        except Exception as e:
            logger.error("Failed to handle confirmation callback", error=str(e))
            await callback.answer("❌ Terjadi kesalahan.")

    async def callback_language(self, callback: CallbackQuery):
        """Handle language selection callbacks"""
        try:
            data_parts = callback.data.split("_")
            lang_code = data_parts[1]
            
            user = await self._get_or_create_user(callback.from_user)
            
            # Update user language
            db = get_db()
            try:
                user.language = lang_code
                db.commit()
                
                lang_name = "Bahasa Indonesia" if lang_code == "id" else "English"
                await callback.message.edit_text(f"✅ Bahasa berhasil diubah ke {lang_name}")
                
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
                await callback.message.edit_text(stats_text)
            
            await callback.answer()
            
        except Exception as e:
            logger.error("Failed to handle admin callback", error=str(e))
            await callback.answer("❌ Terjadi kesalahan.")

    async def process_trade_amount(self, message: Message, state: FSMContext):
        """Process trade amount input"""
        try:
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

Action: {trade_type.upper()}
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
                
                settings_record.stop_loss_percentage = stop_loss
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
                
                settings_record.take_profit_percentage = take_profit
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
    
    # Helper Methods
    
    async def _get_portfolio_data(self, user) -> Dict[str, Any]:
        """Get real portfolio data from Indodax"""
        try:
            if not user.indodax_api_key or not user.indodax_secret_key:
                return {"error": "API credentials not set"}
            
            # Decrypt API credentials
            api_key = decrypt_api_key(str(user.indodax_api_key))
            secret_key = decrypt_api_key(str(user.indodax_secret_key))
            
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
            if not user.indodax_api_key or not user.indodax_secret_key:
                await message.answer("❌ API credentials belum diatur. Gunakan /daftar untuk setup.")
                return
            
            # Decrypt API credentials
            api_key = decrypt_api_key(str(user.indodax_api_key))
            secret_key = decrypt_api_key(str(user.indodax_secret_key))
            
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
                        await self.bot.send_message(user.telegram_id, message_text)
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
                if not user or not user.indodax_api_key:
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
                                            trade.status = "completed"
                                            trade.updated_at = datetime.utcnow()
                                            db.commit()
                                        
                                        return  # Stop monitoring
                                    elif order["status"] == "cancelled":
                                        # Order cancelled
                                        await self._send_order_cancelled_notification(user, order, pair_id)
                                        
                                        # Update database
                                        trade = db.query(Trade).filter(Trade.order_id == str(order_id)).first()
                                        if trade:
                                            trade.status = "cancelled"
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
                
                await callback.message.edit_text(confirm_text, reply_markup=keyboard)
                
            elif action == "custom":
                # Ask for custom amount input
                await callback.message.edit_text(f"💰 Masukkan jumlah IDR untuk {trade_type} {pair_id.upper()}: (Min. 10,000 IDR)")
                await state.set_state(TradingStates.entering_amount)
            
            await callback.answer()
            
        except Exception as e:
            logger.error("Failed to handle amount selection callback", error=str(e))
            await callback.answer("❌ Terjadi kesalahan.")
    
    async def _show_buy_amount_selection(self, message, user, pair_id: str, state: FSMContext):
        """Show buy amount selection with balance and percentage options"""
        try:
            # Get user's IDR balance
            api_key = decrypt_api_key(str(user.indodax_api_key))
            secret_key = decrypt_api_key(str(user.indodax_secret_key))
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
            user = await self._get_or_create_user(callback.from_user)
            
            if not user.indodax_api_key:
                await callback.message.edit_text("❌ Anda belum terdaftar. Gunakan /daftar untuk mendaftar.")
                return
            
            # Get portfolio data
            portfolio_data = await self._get_portfolio_data(user)
            if "error" in portfolio_data:
                await callback.message.edit_text(f"❌ Error: {portfolio_data['error']}")
                return
                
            portfolio_text = self.messages.format_portfolio(portfolio_data, user.language)
            await callback.message.edit_text(portfolio_text)
            await callback.answer()
            
        except Exception as e:
            logger.error("Failed to handle portfolio callback", error=str(e))
            await callback.answer("❌ Terjadi kesalahan.")

    async def callback_balance(self, callback: CallbackQuery):
        """Handle balance callback"""
        try:
            user = await self._get_or_create_user(callback.from_user)
            
            if not user.indodax_api_key:
                await callback.message.edit_text("❌ Anda belum terdaftar. Gunakan /daftar untuk mendaftar.")
                return
                
            # Get balance from Indodax
            api_key = decrypt_api_key(str(user.indodax_api_key))
            secret_key = decrypt_api_key(str(user.indodax_secret_key))
            
            user_api = IndodaxAPI(api_key, secret_key)
            balance_data = await user_api.get_balance()
            
            balance_text = self.messages.format_balance(balance_data, user.language)
            await callback.message.edit_text(balance_text)
            await callback.answer()
            
        except Exception as e:
            logger.error("Failed to handle balance callback", error=str(e))
            await callback.answer("❌ Terjadi kesalahan.")

    async def callback_signals(self, callback: CallbackQuery):
        """Handle signals callback"""
        try:
            user = await self._get_or_create_user(callback.from_user)
            
            # Generate signal
            signal = await self.signal_generator.generate_signal("btc_idr")
            
            if signal:
                signal_text = self.messages.format_signal(signal, user.language)
                keyboard = InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(
                        text="🔄 Update Signal",
                        callback_data="signal_update_btc_idr"
                    ),
                    InlineKeyboardButton(
                        text="💹 Trade",
                        callback_data=f"trade_btc_idr_{signal.signal_type}"
                    )
                ]])
                
                await callback.message.edit_text(signal_text, reply_markup=keyboard)
            else:
                await callback.message.edit_text("❌ Tidak dapat menghasilkan sinyal saat ini.")
                
            await callback.answer()
            
        except Exception as e:
            logger.error("Failed to handle signals callback", error=str(e))
            await callback.answer("❌ Terjadi kesalahan.")

    async def callback_trading(self, callback: CallbackQuery):
        """Handle trading callback"""
        try:
            user = await self._get_or_create_user(callback.from_user)
            
            if not user.indodax_api_key:
                await callback.message.edit_text("❌ Anda belum terdaftar. Gunakan /daftar untuk mendaftar.")
                return
                
            keyboard = create_trading_keyboard("buy", user.language)
            await callback.message.edit_text("💹 Pilih jenis trading:", reply_markup=keyboard)
            await callback.answer()
            
        except Exception as e:
            logger.error("Failed to handle trading callback", error=str(e))
            await callback.answer("❌ Terjadi kesalahan.")

    async def callback_orders(self, callback: CallbackQuery):
        """Handle orders callback"""
        try:
            user = await self._get_or_create_user(callback.from_user)
            
            if not user.indodax_api_key:
                await callback.message.edit_text("❌ Anda belum terdaftar. Gunakan /daftar untuk mendaftar.")
                return
                
            # Get open orders
            api_key = decrypt_api_key(str(user.indodax_api_key))
            secret_key = decrypt_api_key(str(user.indodax_secret_key))
            
            user_api = IndodaxAPI(api_key, secret_key)
            orders_data = await user_api.get_open_orders()
            
            orders_text = self.messages.format_orders(orders_data, user.language)
            await callback.message.edit_text(orders_text)
            await callback.answer()
            
        except Exception as e:
            logger.error("Failed to handle orders callback", error=str(e))
            await callback.answer("❌ Terjadi kesalahan.")

    async def callback_settings_menu(self, callback: CallbackQuery):
        """Handle settings menu callback"""
        try:
            user = await self._get_or_create_user(callback.from_user)
            settings_keyboard = create_settings_keyboard(user.language)
            settings_text = self.messages.get_settings_message(user.language)
            
            await callback.message.edit_text(settings_text, reply_markup=settings_keyboard)
            await callback.answer()
            
        except Exception as e:
            logger.error("Failed to handle settings callback", error=str(e))
            await callback.answer("❌ Terjadi kesalahan.")

    async def callback_help(self, callback: CallbackQuery):
        """Handle help callback"""
        try:
            user = await self._get_or_create_user(callback.from_user)
            help_text = self.messages.get_help_message(user.language)
            
            await callback.message.edit_text(help_text)
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

Gunakan: <code>/backtest [pair] [strategy] [period]</code>

<b>Pairs tersedia:</b>
{pairs_text}
{f"<i>...dan {more_pairs_count} pair lainnya</i>" if more_pairs_count > 0 else ""}

<b>Strategies:</b>
• <code>ai_signals</code> - Backtest AI signals
• <code>dca</code> - Backtest DCA strategy
• <code>buy_hold</code> - Buy and hold strategy

<b>Periods:</b>
• <code>7d</code> - 7 hari
• <code>30d</code> - 30 hari  
• <code>90d</code> - 90 hari

<b>Contoh:</b>
<code>/backtest btc ai_signals 30d</code>
<code>/backtest eth buy_hold 90d</code>
<code>/backtest sol dca 7d</code>

⚠️ <i>Semua parameter wajib diisi!</i>
"""
                await message.answer(help_text, parse_mode="HTML")
                return
            
            pair_symbol = command_parts[1].lower()
            strategy = command_parts[2].lower()
            period = command_parts[3].lower()
            
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
            valid_strategies = ["ai_signals", "dca", "buy_hold"]
            if strategy not in valid_strategies:
                await message.answer(f"❌ Strategy {strategy} tidak didukung. Gunakan salah satu: {', '.join(valid_strategies)}")
                return
            
            # Validate period
            valid_periods = ["7d", "30d", "90d"]
            if period not in valid_periods:
                await message.answer(f"❌ Period {period} tidak didukung. Gunakan salah satu: {', '.join(valid_periods)}")
                return
            
            await message.answer(f"🔄 Menjalankan backtest {strategy} untuk {pair_symbol.upper()}... Mohon tunggu sebentar.")
            
            # Run backtest
            from core.backtester import Backtester
            from datetime import datetime, timedelta
            
            backtester = Backtester()
            
            # Calculate dates based on period
            end_date = datetime.now()
            if period == "7d":
                start_date = end_date - timedelta(days=7)
            elif period == "30d":
                start_date = end_date - timedelta(days=30)
            elif period == "90d":
                start_date = end_date - timedelta(days=90)
            else:
                start_date = end_date - timedelta(days=30)  # default
            
            results = await backtester.run_backtest(
                pair_id=pair_id,
                start_date=start_date,
                end_date=end_date,
                initial_balance=1000000,  # 1M IDR
                strategy=strategy
            )
            
            if results:
                result_text = f"""
📊 <b>Hasil Backtest</b>

Strategy: {strategy.upper()}
Period: {period}
Pair: {pair_symbol.upper()}/IDR

<b>Performance:</b>
• Total Return: {results.total_return_percent:.2f}%
• Sharpe Ratio: {results.sharpe_ratio:.2f}
• Max Drawdown: {results.max_drawdown:.2f}%
• Win Rate: {results.win_rate:.1f}%

<b>Trades:</b>
• Total Trades: {results.total_trades}
• Profitable: {results.winning_trades}
• Loss: {results.losing_trades}

<b>Balance:</b>
• Initial: {format_currency(results.initial_balance)} IDR
• Final: {format_currency(results.final_balance)} IDR
• Profit/Loss: {format_currency(results.total_return)} IDR

⚠️ Past performance does not guarantee future results.
"""
                await message.answer(result_text)
            else:
                await message.answer("❌ Gagal menjalankan backtest. Silakan coba lagi nanti.")
            
        except Exception as e:
            logger.error("Failed to handle backtest command", error=str(e))
            await message.answer("❌ Terjadi kesalahan saat menjalankan backtest.")

    async def _generate_ai_response(self, question: str, user) -> str:
        """Generate AI response for user questions"""
        try:
            # This is a simplified AI assistant
            # In production, you could integrate with OpenAI GPT or other LLM services
            
            question_lower = question.lower()
            
            # Trading basics
            if any(word in question_lower for word in ['trading', 'cara trading', 'bagaimana trading']):
                return """
📚 <b>Dasar-dasar Trading Cryptocurrency</b>

🔍 <b>Langkah-langkah Trading:</b>
1. Riset dan analisis pasar
2. Tentukan strategi (technical/fundamental)
3. Kelola risiko (stop loss, position sizing)
4. Eksekusi dengan disiplin
5. Evaluasi dan perbaiki

💡 <b>Tips Penting:</b>
• Jangan investasi lebih dari yang bisa Anda rugi
• Diversifikasi portfolio
• Gunakan stop loss
• Kontrol emosi saat trading

🤖 Gunakan fitur signal AI bot ini untuk bantuan analisis!
"""
            
            # DCA explanation
            elif any(word in question_lower for word in ['dca', 'dollar cost averaging']):
                return """
💰 <b>Dollar Cost Averaging (DCA)</b>

🎯 <b>DCA adalah strategi investasi:</b>
• Membeli aset secara berkala dengan jumlah tetap
• Tidak peduli harga naik atau turun
• Mengurangi dampak volatilitas harga

✅ <b>Keuntungan DCA:</b>
• Mengurangi risiko timing yang buruk
• Disiplin investasi jangka panjang
• Cocok untuk pemula
• Mengurangi stress trading

📅 <b>Cara setting DCA di bot:</b>
Gunakan menu /dca untuk mengatur

💡 Bot ini mendukung DCA otomatis untuk BTC dan ETH!
"""
            
            # Market analysis
            elif any(word in question_lower for word in ['analisa', 'analysis', 'chart', 'pasar']):
                return """
📊 <b>Analisis Pasar Cryptocurrency</b>

🔍 <b>Jenis Analisis:</b>

📈 <b>Technical Analysis:</b>
• Menggunakan chart dan indikator
• RSI, MACD, Bollinger Bands
• Support & Resistance levels
• Volume analysis

📰 <b>Fundamental Analysis:</b>
• News dan perkembangan teknologi
• Adopsi mainstream
• Regulatory developments
• Market sentiment

🤖 <b>AI Analysis:</b>
• Kombinasi technical + sentiment
• Machine learning predictions
• Real-time signal generation

💡 Gunakan /signal untuk mendapat analisis AI terbaru!
"""
            
            # When to buy/sell
            elif any(word in question_lower for word in ['kapan buy', 'when buy', 'kapan beli', 'waktu buy']):
                return """
⏰ <b>Kapan Waktu yang Tepat untuk Buy?</b>

🎯 <b>Indikator Buy Signal:</b>
• RSI < 30 (oversold)
• Price break above resistance
• Volume spike dengan price increase
• Positive news/sentiment
• AI confidence > 70%

📉 <b>Kondisi Market yang Baik:</b>
• Market dalam uptrend
• Support level yang kuat
• Low volatility periods
• After major corrections

🤖 <b>Gunakan AI Bot:</b>
• /signal untuk cek kondisi terkini
• Set auto-trading untuk eksekusi otomatis
• Monitor dengan portfolio tracker

⚠️ <b>Ingat:</b> Tidak ada waktu yang "perfect" - selalu gunakan risk management!
"""
            
            # Risk management
            elif any(word in question_lower for word in ['risk', 'risiko', 'stop loss', 'risk management']):
                return """
🛡️ <b>Risk Management dalam Trading</b>

📏 <b>Position Sizing:</b>
• Maksimal 2-5% portfolio per trade
• Diversifikasi di beberapa aset
• Jangan all-in dalam satu trade

🔴 <b>Stop Loss:</b>
• Set maksimal 3-5% loss per trade
• Gunakan technical levels sebagai guide
• Stick to your plan!

📊 <b>Portfolio Management:</b>
• Maksimal 10-20% dalam crypto
• Keep emergency fund
• Regular profit taking

⚙️ <b>Bot Features:</b>
• Auto stop loss/take profit
• Risk validation sebelum trade
• Daily trade limits
• Portfolio tracking

💡 Gunakan /settings untuk atur risk parameters!
"""
            
            # Bitcoin specific
            elif any(word in question_lower for word in ['bitcoin', 'btc']):
                return """
₿ <b>Tentang Bitcoin (BTC)</b>

🏆 <b>King of Crypto:</b>
• First cryptocurrency (2009)
• Store of value digital
• Limited supply: 21 juta BTC
• Proof of Work consensus

📈 <b>Bitcoin sebagai Investment:</b>
• Long-term appreciation potential
• Hedge against inflation
• Institutional adoption meningkat
• Portfolio diversification

🔍 <b>Trading BTC:</b>
• High liquidity
• 24/7 market
• Volatility tinggi = opportunity
• Korelasi dengan traditional markets

🤖 <b>BTC di Bot ini:</b>
• Real-time signals
• Auto-trading support
• DCA scheduling
• Portfolio tracking

💡 Gunakan /signal BTC untuk analisis terbaru!
"""
            
            # General crypto question
            elif any(word in question_lower for word in ['crypto', 'cryptocurrency']):
                return """
🚀 <b>Cryptocurrency Overview</b>

💰 <b>Apa itu Cryptocurrency?</b>
• Digital asset berbasis blockchain
• Decentralized dan secure
• Global payment system
• Investment vehicle

🔝 <b>Top Cryptocurrencies:</b>
• Bitcoin (BTC) - Digital Gold
• Ethereum (ETH) - Smart Contracts
• USDT - Stablecoin
• BNB, ADA, SOL, dll

🎯 <b>Cara Mulai:</b>
1. Pelajari basic concepts
2. Pilih exchange terpercaya (Indodax)
3. Start dengan amount kecil
4. Gunakan tools seperti bot ini

🤖 <b>Bot Features untuk Crypto:</b>
• Multi-pair trading
• AI-powered signals
• Risk management
• Portfolio analytics

📚 Pelajari lebih lanjut dengan /help command!
"""
            
            else:
                # Default response for unrecognized questions
                return f"""
🤖 <b>AI Assistant</b>

Pertanyaan Anda: "{question}"

Maaf, saya belum bisa menjawab pertanyaan spesifik ini. Tapi saya bisa membantu dengan:

📚 <b>Topics yang bisa ditanyakan:</b>
• Cara trading cryptocurrency
• Strategi DCA (Dollar Cost Averaging)
• Analisis market dan chart
• Kapan waktu buy/sell yang tepat
• Risk management dan stop loss
• Informasi Bitcoin dan crypto lainnya

💡 <b>Contoh pertanyaan:</b>
• "Apa itu DCA?"
• "Bagaimana cara trading Bitcoin?"
• "Kapan waktu yang tepat untuk buy?"

🤖 Untuk analisis real-time, gunakan /signal
📊 Untuk cek portfolio, gunakan /portfolio
⚙️ Untuk pengaturan, gunakan /settings
"""
            
        except Exception as e:
            logger.error("Failed to generate AI response", error=str(e))
            return "❌ Maaf, terjadi kesalahan dalam sistem AI. Silakan coba lagi nanti."