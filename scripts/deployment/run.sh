#!/bin/bash
# 基础启动脚本，可根据需要修改

# 获取脚本所在目录并定位项目根目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Function to stop the services
stop_services() {
    echo "Stopping services..."
    pkill -f "npm run server"
    pkill -f "uvicorn src.main:app --host 0.0.0.0 --port 5000 --reload"
    exit
}

# Trap signals to stop services
trap stop_services SIGINT SIGTERM

# Start the server
cd "$PROJECT_ROOT" || exit
uvicorn src.main:app --host 0.0.0.0 --port 5000 --reload &

# Start the frontend service
cd "$PROJECT_ROOT/web" || exit
npm run server &

# Wait for all background jobs to finish
wait

