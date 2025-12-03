#!/usr/bin/env bash
set -e

echo "Starting Micro XRCE-DDS Agent on UDP port 8888..."
echo "Command: MicroXRCEAgent udp4 -p 8888"

# Start the agent
MicroXRCEAgent udp4 -p 8888