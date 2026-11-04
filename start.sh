#!/bin/bash

# Start Flask backend on internal port 5060 (not exposed to the internet)
echo "🚀 Starting Flask Backend on port 5060..."
gunicorn app:app --bind 0.0.0.0:5060 --timeout 300 --workers 1 &

# Give Flask a moment to boot
sleep 3

# Start Streamlit frontend on Render's public $PORT
echo "🚀 Starting Streamlit Frontend on port $PORT..."
streamlit run frontend.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false
