"""
Keyboard layouts for Telegram bot
"""
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardMarkup, KeyboardButton
from typing import List

def create_main_keyboard(language: str = "id") -> InlineKeyboardMarkup:
    """Create main menu keyboard"""
    if language == "en":
        buttons = [
            [
                InlineKeyboardButton(text="💼 Portfolio", callback_data="portfolio"),
                InlineKeyboardButton(text="💰 Balance", callback_data="balance")
            ],
            [
                InlineKeyboardButton(text="📊 AI Signals", callback_data="signals"),
                InlineKeyboardButton(text="💹 Trading", callback_data="trading")
            ],
            [
                InlineKeyboardButton(text="📋 Orders", callback_data="orders"),
                InlineKeyboardButton(text="⚙️ Settings", callback_data="settings")
            ],
            [
                InlineKeyboardButton(text="❓ Help", callback_data="help")
            ]
        ]
    else:  # Indonesian
        buttons = [
            [
                InlineKeyboardButton(text="💼 Portfolio", callback_data="portfolio"),
                InlineKeyboardButton(text="💰 Saldo", callback_data="balance")
            ],
            [
                InlineKeyboardButton(text="📊 Sinyal AI", callback_data="signals"),
                InlineKeyboardButton(text="💹 Trading", callback_data="trading")
            ],
            [
                InlineKeyboardButton(text="📋 Order", callback_data="orders"),
                InlineKeyboardButton(text="⚙️ Pengaturan", callback_data="settings")
            ],
            [
                InlineKeyboardButton(text="❓ Bantuan", callback_data="help")
            ]
        ]
    
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def create_trading_keyboard(trade_type: str, language: str = "id") -> InlineKeyboardMarkup:
    """Create trading pair selection keyboard"""
    
    # Popular trading pairs
    pairs = [
        ("BTC/IDR", "btc_idr"),
        ("ETH/IDR", "eth_idr"),
        ("BNB/IDR", "bnb_idr"),
        ("ADA/IDR", "ada_idr"),
        ("SOL/IDR", "sol_idr"),
        ("DOT/IDR", "dot_idr"),
        ("LINK/IDR", "link_idr"),
        ("UNI/IDR", "uni_idr")
    ]
    
    buttons = []
    for i in range(0, len(pairs), 2):
        row = []
        for j in range(2):
            if i + j < len(pairs):
                pair_name, pair_id = pairs[i + j]
                row.append(InlineKeyboardButton(
                    text=pair_name,
                    callback_data=f"trade_{trade_type}_{pair_id}"
                ))
        buttons.append(row)
    
    # Add back button
    back_text = "🔙 Kembali" if language == "id" else "🔙 Back"
    buttons.append([InlineKeyboardButton(text=back_text, callback_data="back_main")])
    
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def create_settings_keyboard(language: str = "id") -> InlineKeyboardMarkup:
    """Create settings keyboard"""
    if language == "en":
        buttons = [
            [
                InlineKeyboardButton(text="🛑 Stop Loss", callback_data="settings_stop_loss"),
                InlineKeyboardButton(text="🎯 Take Profit", callback_data="settings_take_profit")
            ],
            [
                InlineKeyboardButton(text="💰 Max Trade Amount", callback_data="settings_max_amount"),
                InlineKeyboardButton(text="🔄 Auto Trading", callback_data="settings_auto_trading")
            ],
            [
                InlineKeyboardButton(text="🔔 Notifications", callback_data="settings_notifications"),
                InlineKeyboardButton(text="🌐 Language", callback_data="settings_language")
            ],
            [
                InlineKeyboardButton(text="🔑 API Keys", callback_data="settings_api_keys"),
                InlineKeyboardButton(text="⚡ DCA Settings", callback_data="settings_dca")
            ],
            [
                InlineKeyboardButton(text="🔙 Back", callback_data="back_main")
            ]
        ]
    else:  # Indonesian
        buttons = [
            [
                InlineKeyboardButton(text="🛑 Stop Loss", callback_data="settings_stop_loss"),
                InlineKeyboardButton(text="🎯 Take Profit", callback_data="settings_take_profit")
            ],
            [
                InlineKeyboardButton(text="💰 Batas Trade", callback_data="settings_max_amount"),
                InlineKeyboardButton(text="🔄 Auto Trading", callback_data="settings_auto_trading")
            ],
            [
                InlineKeyboardButton(text="🔔 Notifikasi", callback_data="settings_notifications"),
                InlineKeyboardButton(text="🌐 Bahasa", callback_data="settings_language")
            ],
            [
                InlineKeyboardButton(text="🔑 API Keys", callback_data="settings_api_keys"),
                InlineKeyboardButton(text="⚡ DCA Settings", callback_data="settings_dca")
            ],
            [
                InlineKeyboardButton(text="🔙 Kembali", callback_data="back_main")
            ]
        ]
    
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def create_signal_keyboard(pair_id: str, signal_type: str, language: str = "id") -> InlineKeyboardMarkup:
    """Create signal action keyboard"""
    
    if language == "en":
        trade_text = f"💹 {signal_type.upper()}"
        update_text = "🔄 Update"
        more_text = "📊 More Signals"
    else:
        trade_text = f"💹 {signal_type.upper()}"
        update_text = "🔄 Perbarui"
        more_text = "📊 Sinyal Lain"
    
    buttons = [
        [
            InlineKeyboardButton(text=trade_text, callback_data=f"trade_{signal_type}_{pair_id}"),
            InlineKeyboardButton(text=update_text, callback_data=f"signal_update_{pair_id}")
        ],
        [
            InlineKeyboardButton(text=more_text, callback_data="signals_more")
        ]
    ]
    
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def create_confirm_trade_keyboard(language: str = "id") -> InlineKeyboardMarkup:
    """Create trade confirmation keyboard"""
    if language == "en":
        buttons = [
            [
                InlineKeyboardButton(text="✅ Confirm", callback_data="confirm_trade_yes"),
                InlineKeyboardButton(text="❌ Cancel", callback_data="confirm_trade_no")
            ]
        ]
    else:
        buttons = [
            [
                InlineKeyboardButton(text="✅ Konfirmasi", callback_data="confirm_trade_yes"),
                InlineKeyboardButton(text="❌ Batal", callback_data="confirm_trade_no")
            ]
        ]
    
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def create_language_keyboard() -> InlineKeyboardMarkup:
    """Create language selection keyboard"""
    buttons = [
        [
            InlineKeyboardButton(text="🇮🇩 Bahasa Indonesia", callback_data="lang_id"),
            InlineKeyboardButton(text="🇺🇸 English", callback_data="lang_en")
        ]
    ]
    
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def create_admin_keyboard() -> InlineKeyboardMarkup:
    """Create admin panel keyboard"""
    buttons = [
        [
            InlineKeyboardButton(text="📊 Statistics", callback_data="admin_stats"),
            InlineKeyboardButton(text="👥 Users", callback_data="admin_users")
        ],
        [
            InlineKeyboardButton(text="📢 Broadcast", callback_data="admin_broadcast"),
            InlineKeyboardButton(text="💹 Trading Stats", callback_data="admin_trading")
        ],
        [
            InlineKeyboardButton(text="🛠️ Maintenance", callback_data="admin_maintenance"),
            InlineKeyboardButton(text="📝 Logs", callback_data="admin_logs")
        ]
    ]
    
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def create_crypto_list_keyboard(language: str = "id") -> InlineKeyboardMarkup:
    """Create cryptocurrency selection keyboard"""
    
    # Major cryptocurrencies with Indonesian names
    cryptos = [
        ("Bitcoin (BTC)", "btc_idr"),
        ("Ethereum (ETH)", "eth_idr"),
        ("Binance Coin (BNB)", "bnb_idr"),
        ("Cardano (ADA)", "ada_idr"),
        ("Solana (SOL)", "sol_idr"),
        ("Polkadot (DOT)", "dot_idr"),
        ("Chainlink (LINK)", "link_idr"),
        ("Uniswap (UNI)", "uni_idr"),
        ("Litecoin (LTC)", "ltc_idr"),
        ("Polygon (MATIC)", "matic_idr")
    ]
    
    buttons = []
    for i in range(0, len(cryptos), 2):
        row = []
        for j in range(2):
            if i + j < len(cryptos):
                crypto_name, pair_id = cryptos[i + j]
                row.append(InlineKeyboardButton(
                    text=crypto_name,
                    callback_data=f"crypto_select_{pair_id}"
                ))
        buttons.append(row)
    
    # Add back button
    back_text = "🔙 Kembali" if language == "id" else "🔙 Back"
    buttons.append([InlineKeyboardButton(text=back_text, callback_data="back_main")])
    
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def create_timeframe_keyboard(language: str = "id") -> InlineKeyboardMarkup:
    """Create timeframe selection keyboard for charts/analysis"""
    
    timeframes = [
        ("1m", "1"),
        ("15m", "15"),
        ("1h", "60"),
        ("4h", "240"),
        ("1d", "1D"),
        ("1w", "1W")
    ]
    
    buttons = []
    for i in range(0, len(timeframes), 3):
        row = []
        for j in range(3):
            if i + j < len(timeframes):
                tf_name, tf_value = timeframes[i + j]
                row.append(InlineKeyboardButton(
                    text=tf_name,
                    callback_data=f"timeframe_{tf_value}"
                ))
        buttons.append(row)
    
    return InlineKeyboardMarkup(inline_keyboard=buttons)
