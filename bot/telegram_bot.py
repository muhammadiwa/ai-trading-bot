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
from core.indodax_api import indodax_api
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
            api_key = decrypt_api_key(user.indodax_api_key)
            secret_key = decrypt_api_key(user.indodax_secret_key)
            
            user_api = indodax_api.__class__(api_key, secret_key)
            balance_data = await user_api.get_balance()
            
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
            api_key = decrypt_api_key(user.indodax_api_key)
            secret_key = decrypt_api_key(user.indodax_secret_key)
            
            user_api = indodax_api.__class__(api_key, secret_key)
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
            test_api = indodax_api.__class__(api_key, secret_key)
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
                
                await callback.message.answer(f"💰 Masukkan jumlah IDR untuk {action} {pair_id.upper()}:")
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
                amount = float(amount_text.replace(",", "").replace(".", ""))
            except ValueError:
                await message.answer("❌ Jumlah tidak valid. Masukkan angka yang benar.")
                return
            
            if amount <= 0:
                await message.answer("❌ Jumlah harus lebih besar dari 0.")
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
            api_key = decrypt_api_key(user.indodax_api_key)
            secret_key = decrypt_api_key(user.indodax_secret_key)
            
            # Create user-specific API client
            user_api = indodax_api(api_key, secret_key)
            
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
            api_key = decrypt_api_key(user.indodax_api_key)
            secret_key = decrypt_api_key(user.indodax_secret_key)
            
            # Create user-specific API client
            user_api = indodax_api(api_key, secret_key)
            
            # Get current ticker to determine price
            ticker = await user_api.get_ticker(pair_id)
            
            if trade_type == "buy":
                # For buy orders, use ask price and calculate quantity
                price = float(ticker["ticker"]["buy"])
                quantity = amount / price
                
                # Place buy order
                result = await user_api.trade(
                    pair=pair_id,
                    type="buy",
                    price=price,
                    idr=amount
                )
            else:  # sell
                # For sell orders, amount is the quantity to sell
                price = float(ticker["ticker"]["sell"])
                
                # Place sell order
                result = await user_api.trade(
                    pair=pair_id,
                    type="sell",
                    price=price,
                    btc=amount  # This should be dynamic based on currency
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
                        amount=amount,
                        price=price,
                        total=amount if trade_type == "buy" else amount * price,
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
{"Amount: " + format_currency(amount) + " IDR" if trade_type == "buy" else f"Quantity: {amount:.8f}"}
Price: {format_currency(price)} IDR

Order akan dieksekusi sesuai kondisi pasar.
                """
                await message.answer(success_msg)
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
            api_key = decrypt_api_key(user.indodax_api_key)
            secret_key = decrypt_api_key(user.indodax_secret_key)
            
            user_api = indodax_api.__class__(api_key, secret_key)
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
            api_key = decrypt_api_key(user.indodax_api_key)
            secret_key = decrypt_api_key(user.indodax_secret_key)
            
            user_api = indodax_api.__class__(api_key, secret_key)
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
