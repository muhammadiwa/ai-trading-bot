#!/bin/bash

# AI Trading Bot Startup Script
# This script starts the Streamlit web application

echo "🤖 Starting AI Trading Bot Dashboard..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
echo "Checking dependencies..."
pip install -r requirements.txt --quiet

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your actual API keys and configuration"
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Set up environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Start Streamlit application
echo "🚀 Starting Streamlit dashboard..."
echo "📱 Dashboard will be available at: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the application"

# Run Streamlit with custom configuration
streamlit run streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.fileWatcherType none \
    --browser.gatherUsageStats false \
    --theme.primaryColor "#667eea" \
    --theme.backgroundColor "#ffffff" \
    --theme.secondaryBackgroundColor "#f0f2f6" \
    --theme.textColor "#262730"
