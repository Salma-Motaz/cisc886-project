#!/bin/bash
set -e
echo "[bootstrap] Installing dependencies on all nodes..."

# Install system-wide so ALL users including YARN containers can find them
sudo pip3 install matplotlib boto3 pandas --quiet

# Verify
python3 -c "import matplotlib; import boto3; import pandas; print('[bootstrap] OK')"
echo "[bootstrap] Done."