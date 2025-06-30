#!/usr/bin/env python3
"""
Fix script to handle null pointer and type issues in telegram_bot.py
"""
import re

def fix_telegram_bot_file():
    # Read the file
    with open('bot/telegram_bot.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: message.text.split() calls
    content = re.sub(
        r'(\s+)command_parts = message\.text\.split\(\)',
        r'\1if message.text:\n\1    command_parts = message.text.split()\n\1else:\n\1    command_parts = []',
        content
    )
    
    # Fix 2: message.text.split(" ", 1) calls
    content = re.sub(
        r'(\s+)command_parts = message\.text\.split\(" ", 1\)',
        r'\1if message.text:\n\1    command_parts = message.text.split(" ", 1)\n\1else:\n\1    command_parts = []',
        content
    )
    
    # Fix 3: callback.data.split("_") calls
    content = re.sub(
        r'(\s+)data_parts = callback\.data\.split\("_"\)',
        r'\1if callback.data:\n\1    data_parts = callback.data.split("_")\n\1else:\n\1    data_parts = []',
        content
    )
    
    # Fix 4: callback.message.edit_text calls
    content = re.sub(
        r'(\s+)await callback\.message\.edit_text\(([^)]+)\)',
        r'\1if callback.message and hasattr(callback.message, "edit_text"):\n\1    await callback.message.edit_text(\2)\n\1else:\n\1    await callback.answer("Message cannot be edited")',
        content
    )
    
    # Fix 5: callback.message.answer calls
    content = re.sub(
        r'(\s+)await callback\.message\.answer\(([^)]+)\)',
        r'\1if callback.message and hasattr(callback.message, "answer"):\n\1    await callback.message.answer(\2)\n\1else:\n\1    await callback.answer("Message cannot be answered")',
        content
    )
    
    # Write back the file
    with open('bot/telegram_bot.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Fixed telegram_bot.py null pointer and type issues")

if __name__ == "__main__":
    fix_telegram_bot_file()
