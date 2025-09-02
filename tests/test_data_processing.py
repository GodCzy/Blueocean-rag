#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_data_processing.py - 数据解析和标记功能测试

作者: 团队成员

依赖 datasets 目录下的样本数据
"""

import os
import json
from pathlib import Path
import pytest
from datetime import datetime

from src.data.parser import parse_disease_data, parse_ocean_data
from src.data.tagger import tag_disease_data, tag_ocean_data

# 测试数据目录
TEST_DATA_DIR = Path(__file__).parent.parent / "data" / "examples"
TEST_OUTPUT_DIR = Path(__file__).parent / "test_output"

def setup_module():
    """测试模块初始化"""
    # 创建测试输出目录
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def teardown_module():
    """测试模块清理"""
    # 清理测试输出目录
    if TEST_OUTPUT_DIR.exists():
        for file in TEST_OUTPUT_DIR.glob("*"):
            file.unlink()
        TEST_OUTPUT_DIR.rmdir()

def test_parse_disease_data():
    """测试疾病数据解析"""
    # 解析疾病数据
    disease_data = parse_disease_data(TEST_DATA_DIR)
    
    # 验证解析结果
    assert len(disease_data) > 0
    for data in disease_data:
        assert hasattr(data, "name")
        assert hasattr(data, "symptoms")
        assert hasattr(data, "causes")
        assert hasattr(data, "treatments")
        assert hasattr(data, "prevention")
        assert hasattr(data, "severity")
        assert hasattr(data, "affected_species")

def test_parse_ocean_data():
    """测试海洋环境数据解析"""
    # 解析海洋环境数据
    ocean_data = parse_ocean_data(TEST_DATA_DIR)
    
    # 验证解析结果
    assert len(ocean_data) > 0
    for data in ocean_data:
        assert hasattr(data, "location")
        assert hasattr(data, "timestamp")
        assert hasattr(data, "temperature")
        assert hasattr(data, "salinity")
        assert hasattr(data, "ph")
        assert hasattr(data, "dissolved_oxygen")
        assert hasattr(data, "turbidity")
        assert hasattr(data, "ammonia")
        assert hasattr(data, "nitrite")
        assert hasattr(data, "nitrate")

def test_tag_disease_data():
    """测试疾病数据标记"""
    # 解析疾病数据
    disease_data = parse_disease_data(TEST_DATA_DIR)
    
    # 标记疾病数据
    tagged_data = tag_disease_data(disease_data, TEST_OUTPUT_DIR)
    
    # 验证标记结果
    assert len(tagged_data) > 0
    for data in tagged_data:
        assert hasattr(data, "disease_id")
        assert hasattr(data, "name")
        assert hasattr(data, "category")
        assert hasattr(data, "tags")
        assert hasattr(data, "severity")
        assert hasattr(data, "affected_species")
        assert hasattr(data, "created_at")
        assert hasattr(data, "updated_at")
        
        # 验证标记文件是否存在
        file_path = TEST_OUTPUT_DIR / f"{data.disease_id}.json"
        assert file_path.exists()
        
        # 验证标记文件内容
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)
            assert content["disease_id"] == data.disease_id
            assert content["name"] == data.name
            assert content["category"] == data.category
            assert content["tags"] == data.tags
            assert content["severity"] == data.severity
            assert content["affected_species"] == data.affected_species

def test_tag_ocean_data():
    """测试海洋环境数据标记"""
    # 解析海洋环境数据
    ocean_data = parse_ocean_data(TEST_DATA_DIR)
    
    # 标记海洋环境数据
    tagged_data = tag_ocean_data(ocean_data, TEST_OUTPUT_DIR)
    
    # 验证标记结果
    assert len(tagged_data) > 0
    for data in tagged_data:
        assert hasattr(data, "location_id")
        assert hasattr(data, "location")
        assert hasattr(data, "region")
        assert hasattr(data, "tags")
        assert hasattr(data, "risk_level")
        assert hasattr(data, "created_at")
        assert hasattr(data, "updated_at")
        
        # 验证标记文件是否存在
        file_path = TEST_OUTPUT_DIR / f"{data.location_id}.json"
        assert file_path.exists()
        
        # 验证标记文件内容
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)
            assert content["location_id"] == data.location_id
            assert content["location"] == data.location
            assert content["region"] == data.region
            assert content["tags"] == data.tags
            assert content["risk_level"] == data.risk_level

if __name__ == "__main__":
    pytest.main([__file__]) 