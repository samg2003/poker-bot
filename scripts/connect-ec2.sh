#!/bin/bash
# Connect to an EC2 instance via SSH

read -p "Path to PEM file: " PEM_PATH
read -p "EC2 IP address: " IP_ADDR
read -p "Username (default: ubuntu): " USERNAME
USERNAME=${USERNAME:-ubuntu}

# Strip surrounding quotes if user added them
PEM_PATH=$(echo "$PEM_PATH" | tr -d "'" | tr -d '"')
IP_ADDR=$(echo "$IP_ADDR" | tr -d "'" | tr -d '"')

# Fix permissions
chmod 400 "$PEM_PATH"

echo "Connecting to $USERNAME@$IP_ADDR..."
ssh -i "$PEM_PATH" "$USERNAME@$IP_ADDR"
