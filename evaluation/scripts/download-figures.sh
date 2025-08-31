#!/bin/bash

# Default source and destination directories
DEFAULT_SOURCE_DIR="evaluation/figures"
DEFAULT_DEST_DIR="/mnt/c/Users/xiang/Documents/Working/2024 BitNet/mobicom26/figures/eval/"

# Use default values or command line arguments
SOURCE_DIR="${1:-$DEFAULT_SOURCE_DIR}"
DEST_DIR="${2:-$DEFAULT_DEST_DIR}"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist."
    exit 1
fi

# Create destination directory if it doesn't exist
if [ ! -d "$DEST_DIR" ]; then
    mkdir -p "$DEST_DIR"
    echo "Created destination directory: $DEST_DIR"
fi

# Find all PNG and PDF files and copy them to destination
echo "Finding and copying PNG files from '$SOURCE_DIR' to '$DEST_DIR'..."
find "$SOURCE_DIR" -type f -name "*.png" -exec cp {} "$DEST_DIR" \;
find "$SOURCE_DIR" -type f -name "*.pdf" -exec cp {} "$DEST_DIR" \;

echo "Transfer complete!"