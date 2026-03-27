#!/bin/bash
# Connect to an EC2 instance via SSH

DEFAULT_PEM="/Users/sambhavgupta/Desktop/code-poker-bot/carte6.pem"
DEFAULT_IP="54.193.114.122"
DEFAULT_USER="ubuntu"

read -p "Path to PEM file [$DEFAULT_PEM]: " PEM_PATH
PEM_PATH=${PEM_PATH:-$DEFAULT_PEM}

read -p "EC2 IP address [$DEFAULT_IP]: " IP_ADDR
IP_ADDR=${IP_ADDR:-$DEFAULT_IP}

read -p "Username [$DEFAULT_USER]: " USERNAME
USERNAME=${USERNAME:-$DEFAULT_USER}

# Strip surrounding quotes if user added them
PEM_PATH=$(echo "$PEM_PATH" | tr -d "'" | tr -d '"')
IP_ADDR=$(echo "$IP_ADDR" | tr -d "'" | tr -d '"')

# Fix permissions
chmod 400 "$PEM_PATH"

echo "Connecting to $USERNAME@$IP_ADDR..."
ssh -i "$PEM_PATH" "$USERNAME@$IP_ADDR"
