#!/bin/bash

echo "Installing Stockfish chess engine..."

# Create directory
STOCKFISH_DIR="$HOME/stockfish_install"
mkdir -p $STOCKFISH_DIR
cd $STOCKFISH_DIR

# Instead of compiling, download a pre-built binary
if [[ "$(uname)" == "Darwin" ]]; then
    # macOS - Download macOS version
    echo "Detected macOS, downloading pre-compiled binary..."
    # Use GitHub releases which are more reliable
    curl -L "https://github.com/official-stockfish/Stockfish/releases/download/sf_16/stockfish-macos-x86-64-avx2.tar.gz" -o stockfish.tar.gz
    
    # Extract the tarball
    tar -xzf stockfish.tar.gz
    
    # Create stockfish directory if it doesn't exist
    mkdir -p "$HOME/stockfish"
    
    # Copy the binary to the final location
    if [ -f "stockfish/stockfish-macos-x86-64-avx2" ]; then
        cp "stockfish/stockfish-macos-x86-64-avx2" "$HOME/stockfish/stockfish"
        chmod +x "$HOME/stockfish/stockfish"
        echo "Successfully installed Stockfish"
    else
        echo "Could not find Stockfish binary in downloaded package"
        # Fallback - try to find any executable
        STOCKFISH_BIN=$(find . -name "stockfish*" -type f -perm +111 | head -1)
        if [ -n "$STOCKFISH_BIN" ]; then
            cp "$STOCKFISH_BIN" "$HOME/stockfish/stockfish"
            chmod +x "$HOME/stockfish/stockfish"
            echo "Found and installed alternative Stockfish binary"
        else
            echo "Failed to find any Stockfish binary"
            exit 1
        fi
    fi
    
elif [[ "$(uname)" == "Linux" ]]; then
    # Linux - Download Linux version
    echo "Detected Linux, downloading pre-compiled binary..."
    # Use GitHub releases which are more reliable
    curl -L "https://github.com/official-stockfish/Stockfish/releases/download/sf_16/stockfish-ubuntu-x86-64-avx2.tar.gz" -o stockfish.tar.gz
    
    # Extract the tarball
    tar -xzf stockfish.tar.gz
    
    # Create stockfish directory if it doesn't exist
    mkdir -p "$HOME/stockfish"
    
    # Copy the binary to the final location
    if [ -f "stockfish/stockfish-ubuntu-x86-64-avx2" ]; then
        cp "stockfish/stockfish-ubuntu-x86-64-avx2" "$HOME/stockfish/stockfish"
        chmod +x "$HOME/stockfish/stockfish"
        echo "Successfully installed Stockfish"
    else
        echo "Could not find Stockfish binary in downloaded package"
        # Fallback - try to find any executable
        STOCKFISH_BIN=$(find . -name "stockfish*" -type f -executable | head -1)
        if [ -n "$STOCKFISH_BIN" ]; then
            cp "$STOCKFISH_BIN" "$HOME/stockfish/stockfish"
            chmod +x "$HOME/stockfish/stockfish"
            echo "Found and installed alternative Stockfish binary"
        else
            echo "Failed to find any Stockfish binary"
            exit 1
        fi
    fi
else
    echo "Unsupported OS. Please install Stockfish manually."
    exit 1
fi

echo "Stockfish installed at $HOME/stockfish/stockfish"

# Test that it works
if [ -f "$HOME/stockfish/stockfish" ]; then
    "$HOME/stockfish/stockfish" --version
else
    echo "ERROR: Stockfish binary not found at expected location"
    exit 1
fi 