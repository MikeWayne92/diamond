#!/bin/bash

# Exit on error
set -e

# Function to print error messages
error() {
    echo "Error: $1" >&2
    exit 1
}

# Function to print status messages
info() {
    echo "Info: $1"
}

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0")
if (( $(echo "$PYTHON_VERSION < 3.8" | bc -l) )); then
    error "Python 3.8 or higher is required (found $PYTHON_VERSION)"
fi

# Check if virtualenv is installed
if ! command -v python3 -m venv &> /dev/null; then
    error "python3-venv is not installed. Please install it first."
fi

# Create and activate virtual environment
info "Creating virtual environment..."
python3 -m venv venv || error "Failed to create virtual environment"
source venv/bin/activate || error "Failed to activate virtual environment"

# Upgrade pip
info "Upgrading pip..."
python -m pip install --upgrade pip || error "Failed to upgrade pip"

# Install wheel first
info "Installing wheel..."
pip install wheel || error "Failed to install wheel"

# Install core requirements
info "Installing core requirements..."
pip install -r requirements-core.txt || error "Failed to install core requirements"

# Ask if user wants to install full requirements
read -p "Do you want to install full analysis requirements? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    info "Installing full requirements..."
    pip install -r requirements-full.txt || error "Failed to install full requirements"
fi

# Install the package in development mode
info "Installing package in development mode..."
pip install -e . || error "Failed to install package"

info "Installation completed successfully!"
info "To activate the virtual environment, run: source venv/bin/activate" 