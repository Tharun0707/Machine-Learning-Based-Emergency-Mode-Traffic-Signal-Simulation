#!/bin/bash

echo "Starting Ambulance Detection Server"
echo "==================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Running setup..."
    python3 setup.py
    if [ $? -ne 0 ]; then
        echo "Setup failed. Please check the errors above."
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if model file exists
if [ ! -f "models/best.pt" ]; then
    echo ""
    echo "❌ Model file not found!"
    echo "Please place your 'best.pt' file in the 'models' folder"
    echo ""
    exit 1
fi

# Start the server
echo "Starting server..."
python model_server.py
