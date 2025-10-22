#!/bin/bash
# Docker 部署辅助脚本

# 获取脚本所在目录并定位项目根目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 检查是否提供了命令行参数
if [ $# -eq 0 ]; then
    echo "请提供命令参数，例如：up -d 或 down"
    exit 1
fi

# 将所有命令行参数传递给 docker compose 命令
docker compose -f "$PROJECT_ROOT/docker/docker-compose.dev.yml" --env-file "$PROJECT_ROOT/.env" "$@"
