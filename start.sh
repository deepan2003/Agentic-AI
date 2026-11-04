#!/bin/bash
set -e

# Internal port for Flask
export FLASK_PORT=5060

# Start Flask backend directly with unbuffered python
echo "🚀 Starting Flask Backend on port $FLASK_PORT..."
python -u app.py &
FLASK_PID=$!

# Wait for Flask backend to be ready
echo "⏳ Waiting for Flask backend to respond on http://127.0.0.1:$FLASK_PORT/health..."
MAX_RETRIES=30
COUNT=0
until curl -s "http://127.0.0.1:$FLASK_PORT/health" > /dev/null 2>&1; do
    COUNT=$((COUNT+1))
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo "❌ Flask backend failed to start within 30 seconds."
        break
    fi
    sleep 1
done

echo "✅ Flask Backend is active and healthy!"

# Start Streamlit frontend on Render's public $PORT
echo "🚀 Starting Streamlit Frontend on public port $PORT..."
exec streamlit run frontend.py \
    --server.port "$PORT" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
