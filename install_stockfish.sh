#!/bin/bash

echo "Installing Stockfish chess engine..."

# Create directory
STOCKFISH_DIR="$HOME/stockfish_install"
mkdir -p $STOCKFISH_DIR
cd $STOCKFISH_DIR

# Check if we can install via apt
if command -v apt-get &> /dev/null; then
    echo "Debian/Ubuntu detected, trying to install via apt-get..."
    sudo apt-get update
    if sudo apt-get install -y stockfish; then
        echo "Stockfish installed successfully via apt-get"
        stockfish_path=$(which stockfish)
        echo "Stockfish installed at: $stockfish_path"
        
        # Create symlink in home directory for consistency
        mkdir -p "$HOME/stockfish"
        ln -sf "$stockfish_path" "$HOME/stockfish/stockfish"
        
        # Test that it works
        echo "Testing Stockfish installation:"
        stockfish --version
        exit 0
    else
        echo "apt-get installation failed, falling back to binary download..."
    fi
fi

if [[ "$(uname)" == "Darwin" ]]; then
    # macOS
    echo "Detected macOS, downloading pre-compiled binary..."
    
    # Check if we're on Apple Silicon (arm64)
    if [[ "$(uname -m)" == "arm64" ]]; then
        echo "Apple Silicon (ARM64) detected"
        STOCKFISH_URL="https://github.com/official-stockfish/Stockfish/releases/download/sf_16/stockfish-macos-x86-64-modern"
        # Direct download without archive
        curl -L "$STOCKFISH_URL" -o stockfish
    else
        # Intel Mac
        echo "Intel processor detected"
        STOCKFISH_URL="https://github.com/official-stockfish/Stockfish/releases/download/sf_16/stockfish-macos-x86-64-modern"
        curl -L "$STOCKFISH_URL" -o stockfish
    fi
    
    # Check if download was successful
    if [ ! -s stockfish ]; then
        echo "Download failed. Attempting alternate URL..."
        curl -L "https://github.com/official-stockfish/Stockfish/releases/download/sf_16/stockfish-macos-x86-64-avx2" -o stockfish
    fi
    
elif [[ "$(uname)" == "Linux" ]]; then
    # Linux
    echo "Detected Linux, downloading pre-compiled binary..."
    
    # First try direct binary download (more reliable)
    STOCKFISH_URL="https://github.com/official-stockfish/Stockfish/releases/download/sf_16/stockfish-ubuntu-x86-64-avx2"
    curl -L "$STOCKFISH_URL" -o stockfish
    
    # Check if download was successful
    if [ ! -s stockfish ]; then
        echo "Download failed. Attempting alternate URL..."
        STOCKFISH_URL="https://github.com/official-stockfish/Stockfish/releases/download/sf_16/stockfish-linux-x86-64"
        curl -L "$STOCKFISH_URL" -o stockfish
    fi
else
    echo "Unsupported OS. Please install Stockfish manually."
    exit 1
fi

# Check if we have a valid Stockfish binary
if [ -s stockfish ]; then
    # Make executable
    chmod +x stockfish
    
    # Create stockfish directory if it doesn't exist
    mkdir -p "$HOME/stockfish"
    
    # Copy the binary to the final location
    cp stockfish "$HOME/stockfish/stockfish"
    chmod +x "$HOME/stockfish/stockfish"
    echo "Successfully installed Stockfish binary"
else
    echo "Binary download failed. Attempting to compile from source..."
    
    # Try to compile from source
    echo "Downloading Stockfish source code..."
    rm -rf Stockfish
    git clone https://github.com/official-stockfish/Stockfish.git
    cd Stockfish/src
    
    # Install build dependencies if on Ubuntu/Debian
    if command -v apt-get &> /dev/null; then
        echo "Installing build dependencies..."
        sudo apt-get update
        sudo apt-get install -y g++ make
    fi
    
    # Compile
    echo "Compiling Stockfish from source..."
    make -j$(nproc) build
    
    # Check if compilation was successful
    if [ -f "stockfish" ]; then
        mkdir -p "$HOME/stockfish"
        cp stockfish "$HOME/stockfish/stockfish"
        chmod +x "$HOME/stockfish/stockfish"
        echo "Successfully compiled and installed Stockfish from source"
    else
        echo "Compilation failed. Please install Stockfish manually."
        exit 1
    fi
fi

echo "Stockfish installed at $HOME/stockfish/stockfish"

# Test that it works
if [ -f "$HOME/stockfish/stockfish" ]; then
    echo "Testing Stockfish installation:"
    "$HOME/stockfish/stockfish" --version
else
    echo "ERROR: Stockfish binary not found at expected location"
    exit 1
fi 