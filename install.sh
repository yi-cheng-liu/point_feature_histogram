#!/bin/bash

# List of Python packages to install
PACKAGES=(
    "open3d"
    "numpy"
    "pandas"
)

# Function to install Python packages
install_packages() {
    for package in "${PACKAGES[@]}"; do
        echo "Installing $package..."
        pip install "$package"
    done
}

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "pip could not be found, please install it first."
    exit 1
fi

# Run the installation function
install_packages
