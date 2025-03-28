#!/bin/bash

# Update package list
echo "Updating package list..."
apt-get update

# Install system dependencies
echo "Installing system dependencies..."
apt-get install -y python3-pip stockfish

# Create and activate virtual environment (optional)
echo "Creating virtual environment..."
python3 -m venv chess_env
source chess_env/bin/activate

# Install Python packages
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "All dependencies installed successfully!"
echo "config wandb with: export WANDB_API_KEY=<your_api_key>"