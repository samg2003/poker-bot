#!/bin/bash
# Remove all .DS_Store files recursively from the project directory
find "$(dirname "$0")/.." -name ".DS_Store" -type f -delete
echo "All .DS_Store files removed."
