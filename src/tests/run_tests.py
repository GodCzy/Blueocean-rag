"""
运行所有测试的脚本

确保在虚拟环境中执行
"""
import unittest
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

if __name__ == "__main__":
    # 发现并运行所有测试
    test_suite = unittest.defaultTestLoader.discover(
        start_dir=os.path.dirname(__file__),
        pattern="test_*.py"
    )
    
    # 运行测试
    unittest.TextTestRunner(verbosity=2).run(test_suite) 