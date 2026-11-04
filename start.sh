#!/bin/bash
set -e

# Internal port for Flask (localhost only)
export FLASK_PORT=5060

# Start Flask backend directly with unbuffered python
echo "🚀 Starting Flask Backend on 127.0.0.1:$FLASK_PORT..."
python -u app.py &
FLASK_PID=$!

# Wait for Flask backend to be ready
echo "⏳ Waiting for Flask backend to respond on http://127.0.0.1:$FLASK_PORT/health..."
MAX_RETRIES=60
COUNT=0
READY=0

while [ $COUNT -lt $MAX_RETRIES ]; do
    if curl -s -f "http://127.0.0.1:$FLASK_PORT/health" > /dev/null 2>&1; then
        READY=1
        break
    fi
    COUNT=$((COUNT+1))
    sleep 1
done

if [ $READY -eq 1 ]; then
    echo "✅ Flask Backend is active and healthy on 127.0.0.1:$FLASK_PORT!"
else
    echo "⚠️ Flask backend took longer than 60s to report healthy, continuing..."
fi

# Start Streamlit frontend on Render's public $PORT
echo "🚀 Starting Streamlit Frontend on public port $PORT..."
exec streamlit run frontend.py \
    --server.port "$PORT" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
