#!/bin/bash
set -e

echo "============================================"
echo " Vision Guard – Starting..."
echo "============================================"

# Start REST API server
exec python3 /app/server.py
