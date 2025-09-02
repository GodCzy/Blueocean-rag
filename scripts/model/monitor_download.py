#!/usr/bin/env python3
"""
监控OceanGPT模型下载进度

 与 `scripts/model/manage.py` 脚本配合使用。
"""

import os
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

def format_file_size(size_bytes):
    """格式化文件大小"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def monitor_download_progress():
    """监控下载进度"""
    model_path = REPO_ROOT / "models/OceanGPT-o-7B-v0.1/ZJUNLP/OceanGPT-o-7B"
    
    if not model_path.exists():
        print("❌ 模型目录不存在")
        return
    
    print("🔍 监控OceanGPT-o-7B模型下载进度...")
    print("=" * 60)
    
    while True:
        if not model_path.exists():
            print("❌ 模型目录不存在")
            break
            
        # 统计文件
        all_files = list(model_path.glob("*"))
        total_files = len(all_files)
        total_size = sum(f.stat().st_size for f in all_files if f.is_file())
        
        # 检查关键文件
        key_files = {
            "config.json": model_path / "config.json",
            "tokenizer.json": model_path / "tokenizer.json", 
            "tokenizer_config.json": model_path / "tokenizer_config.json",
        }
        
        # 查找.safetensors文件
        safetensors_files = list(model_path.glob("*.safetensors"))
        
        print(f"⏰ {time.strftime('%H:%M:%S')}")
        print(f"📁 总文件数: {total_files}")
        print(f"📊 总大小: {format_file_size(total_size)}")
        
        print("\n🔑 关键文件状态:")
        for name, file_path in key_files.items():
            status = "✅" if file_path.exists() else "❌"
            size = format_file_size(file_path.stat().st_size) if file_path.exists() else "0B"
            print(f"  {status} {name}: {size}")
        
        print(f"\n⚖️  模型权重文件 (.safetensors): {len(safetensors_files)} 个")
        for sf in safetensors_files:
            size = format_file_size(sf.stat().st_size)
            print(f"  ✅ {sf.name}: {size}")
        
        # 检查是否下载完成
        has_safetensors = len(safetensors_files) > 0
        has_key_files = all(kf.exists() for kf in key_files.values())
        
        if has_safetensors and has_key_files:
            print("\n🎉 模型下载可能已完成！")
            print("建议运行验证脚本确认文件完整性")
            break
        
        print("\n⏳ 下载中... (30秒后刷新)")
        print("=" * 60)
        
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            print("\n\n⏹️  监控已停止")
            break

if __name__ == "__main__":
    monitor_download_progress() 