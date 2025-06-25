#!/bin/bash

# 蓝海智询启动脚本
# 作者: 蓝海智询团队
# 启动前请先按照 DEPLOYMENT_GUIDE.md 完成环境配置

# 设置颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印横幅
print_banner() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                        🌊 蓝海智询                          ║"
    echo "║            基于RAG知识库与知识图谱的水生动物疾病问答平台      ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}[ERROR]${NC} $1 未安装或未添加到PATH"
        return 1
    fi
    return 0
}

# 检查Python版本
check_python() {
    echo -e "${BLUE}[INFO]${NC} 检查Python环境..."
    
    if check_command python3; then
        PYTHON_CMD="python3"
    elif check_command python; then
        PYTHON_CMD="python"
    else
        echo -e "${RED}[ERROR]${NC} Python未安装！请安装Python 3.8+"
        exit 1
    fi
    
    # 检查Python版本
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    echo -e "${GREEN}[INFO]${NC} 找到Python $PYTHON_VERSION"
}

# 创建虚拟环境
setup_venv() {
    if [ ! -d "venv" ]; then
        echo -e "${BLUE}[INFO]${NC} 创建Python虚拟环境..."
        $PYTHON_CMD -m venv venv
        if [ $? -ne 0 ]; then
            echo -e "${RED}[ERROR]${NC} 创建虚拟环境失败！"
            exit 1
        fi
    fi
    
    echo -e "${BLUE}[INFO]${NC} 激活虚拟环境..."
    source venv/bin/activate
}

# 安装依赖
install_dependencies() {
    if [ -f "requirements.txt" ]; then
        echo -e "${BLUE}[INFO]${NC} 安装Python依赖包..."
        pip install -r requirements.txt
        if [ $? -ne 0 ]; then
            echo -e "${YELLOW}[WARNING]${NC} 依赖包安装可能不完整"
        fi
    else
        echo -e "${YELLOW}[WARNING]${NC} requirements.txt 文件不存在"
    fi
}

# 运行快速检查
run_quick_check() {
    if [ -f "quick_start.py" ]; then
        echo -e "${BLUE}[INFO]${NC} 运行环境检查..."
        $PYTHON_CMD quick_start.py
        echo
        echo -e "${BLUE}[INFO]${NC} 环境检查完成，按回车键继续启动服务..."
        read -r
        echo
    fi
}

# 启动服务
start_service() {
    echo -e "${GREEN}[INFO]${NC} 启动蓝海智询服务..."
    echo -e "${GREEN}[INFO]${NC} API文档: http://localhost:8000/docs"
    echo -e "${GREEN}[INFO]${NC} 健康检查: http://localhost:8000/health"
    echo -e "${GREEN}[INFO]${NC} 按 Ctrl+C 停止服务"
    echo
    
    $PYTHON_CMD src/main.py
}

# 主函数
main() {
    print_banner
    
    echo -e "${BLUE}[INFO]${NC} 正在启动蓝海智询系统..."
    echo
    
    check_python
    setup_venv
    install_dependencies
    run_quick_check
    start_service
    
    echo
    echo -e "${BLUE}[INFO]${NC} 服务已停止"
}

# 信号处理
trap 'echo -e "\n${YELLOW}[INFO]${NC} 收到中断信号，正在停止服务..."; exit 0' INT TERM

# 执行主函数
main "$@" 