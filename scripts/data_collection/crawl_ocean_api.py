#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
crawl_ocean_api.py - 抓取海洋环境数据

该脚本用于从各种公开API或网站抓取海洋和淡水环境数据，包括水温、溶氧量、
pH值等参数，并保存为结构化格式用于后续分析和可视化。

用法:
    python crawl_ocean_api.py --output_dir datasets/ocean_data --days 30 --api noaa,usgs

作者: 成员B (数据工程师)
"""

import os
import json
import argparse
import logging
import requests
import time
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from tqdm import tqdm
import concurrent.futures

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("crawl_ocean_api")

# 定义API配置
API_CONFIGS = {
    "noaa": {
        "name": "NOAA海洋数据API",
        "description": "美国国家海洋和大气管理局数据",
        "base_url": "https://tidesandcurrents.noaa.gov/api/datagetter",
        "requires_key": False,
        "default_params": {
            "format": "json",
            "units": "metric",
            "time_zone": "gmt"
        }
    },
    "usgs": {
        "name": "USGS水数据API",
        "description": "美国地质调查局水数据",
        "base_url": "https://waterservices.usgs.gov/nwis/iv/",
        "requires_key": False,
        "default_params": {
            "format": "json",
            "siteStatus": "active"
        }
    },
    "nmdis": {
        "name": "中国海洋环境数据中心",
        "description": "中国国家海洋信息中心数据",
        "base_url": "http://www.nmdis.org.cn/data/ocean",
        "requires_key": True,
        "default_params": {}
    },
    "weather": {
        "name": "OpenWeatherMap API",
        "description": "全球天气数据",
        "base_url": "https://api.openweathermap.org/data/2.5/weather",
        "requires_key": True,
        "default_params": {
            "units": "metric",
            "lang": "zh_cn"
        }
    }
}

# 中国主要水产养殖区域
AQUACULTURE_REGIONS = [
    {"name": "长江下游", "lat": 30.25, "lon": 119.17, "water_type": "freshwater"},
    {"name": "珠江三角洲", "lat": 22.84, "lon": 113.26, "water_type": "freshwater"},
    {"name": "太湖", "lat": 31.23, "lon": 120.13, "water_type": "freshwater"},
    {"name": "洪泽湖", "lat": 33.27, "lon": 118.67, "water_type": "freshwater"},
    {"name": "鄱阳湖", "lat": 29.12, "lon": 116.00, "water_type": "freshwater"},
    {"name": "洞庭湖", "lat": 29.31, "lon": 112.95, "water_type": "freshwater"},
    {"name": "黄河三角洲", "lat": 37.76, "lon": 118.98, "water_type": "brackish"},
    {"name": "渤海湾", "lat": 38.50, "lon": 119.84, "water_type": "marine"},
    {"name": "舟山群岛", "lat": 30.01, "lon": 122.21, "water_type": "marine"},
    {"name": "北部湾", "lat": 21.63, "lon": 108.65, "water_type": "marine"}
]

class OceanDataCrawler:
    """海洋环境数据抓取类"""
    
    def __init__(self, output_dir: str, api_keys: Dict[str, str] = None):
        """初始化抓取器
        
        Args:
            output_dir: 数据保存目录
            api_keys: API密钥字典，格式为 {api_name: key}
        """
        self.output_dir = output_dir
        self.api_keys = api_keys or {}
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 会话对象，用于复用连接
        self.session = requests.Session()
        
        # 请求头
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/json, text/plain, */*"
        })
    
    def _get_api_key(self, api_name: str) -> Optional[str]:
        """获取API密钥
        
        Args:
            api_name: API名称
            
        Returns:
            API密钥，如果未配置则返回None
        """
        # 先从实例变量获取
        key = self.api_keys.get(api_name)
        
        # 如果未配置，尝试从环境变量获取
        if not key:
            env_var = f"{api_name.upper()}_API_KEY"
            key = os.environ.get(env_var)
        
        return key
    
    def fetch_noaa_data(self, days: int = 30) -> List[Dict[str, Any]]:
        """从NOAA API抓取海洋数据
        
        Args:
            days: 获取过去多少天的数据
            
        Returns:
            抓取到的数据列表
        """
        api_config = API_CONFIGS["noaa"]
        results = []
        
        # 计算日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # 格式化日期
        date_start = start_date.strftime("%Y%m%d")
        date_end = end_date.strftime("%Y%m%d")
        
        # 选择一些NOAA的观测站点
        stations = ["8410140", "8413320", "8418150", "8443970", "8447386"]
        
        for station in tqdm(stations, desc="抓取NOAA数据"):
            # 不同数据产品
            products = ["water_temperature", "water_level", "air_temperature", "wind"]
            
            station_data = {"station_id": station, "measurements": []}
            
            for product in products:
                try:
                    # 构建请求参数
                    params = {
                        **api_config["default_params"],
                        "product": product,
                        "station": station,
                        "begin_date": date_start,
                        "end_date": date_end,
                        "datum": "MLLW" if product == "water_level" else None
                    }
                    
                    # 发送请求
                    response = self.session.get(api_config["base_url"], params=params)
                    response.raise_for_status()
                    
                    # 解析响应
                    data = response.json()
                    
                    # 添加到结果
                    if "data" in data:
                        station_data["measurements"].append({
                            "product": product,
                            "data": data["data"]
                        })
                    
                    # 避免请求过快
                    time.sleep(0.5)
                
                except Exception as e:
                    logger.warning(f"抓取NOAA数据失败 (station={station}, product={product}): {e}")
            
            if station_data["measurements"]:
                results.append(station_data)
        
        return results
    
    def fetch_usgs_data(self, days: int = 30) -> List[Dict[str, Any]]:
        """从USGS API抓取水数据
        
        Args:
            days: 获取过去多少天的数据
            
        Returns:
            抓取到的数据列表
        """
        api_config = API_CONFIGS["usgs"]
        results = []
        
        # 计算日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # 格式化日期
        date_start = start_date.strftime("%Y-%m-%d")
        date_end = end_date.strftime("%Y-%m-%d")
        
        # 选择一些USGS的观测站点
        sites = ["01646500", "01578310", "02174000", "02226000", "02236125"]
        
        # 参数代码: https://help.waterdata.usgs.gov/codes-and-parameters/parameters
        parameter_codes = {
            "00010": "Temperature, water",
            "00095": "Specific conductance",
            "00300": "Dissolved oxygen",
            "00400": "pH",
            "63680": "Turbidity"
        }
        
        for site in tqdm(sites, desc="抓取USGS数据"):
            site_data = {"site_id": site, "measurements": []}
            
            try:
                # 构建请求参数
                params = {
                    **api_config["default_params"],
                    "sites": site,
                    "startDT": date_start,
                    "endDT": date_end,
                    "parameterCd": ",".join(parameter_codes.keys())
                }
                
                # 发送请求
                response = self.session.get(api_config["base_url"], params=params)
                response.raise_for_status()
                
                # 解析响应
                data = response.json()
                
                # 处理时间序列数据
                if "value" in data and "timeSeries" in data["value"]:
                    for time_series in data["value"]["timeSeries"]:
                        param_code = time_series.get("variable", {}).get("variableCode", [{}])[0].get("value")
                        if param_code in parameter_codes:
                            param_name = parameter_codes[param_code]
                            values = time_series.get("values", [{}])[0].get("value", [])
                            
                            if values:
                                site_data["measurements"].append({
                                    "parameter": param_name,
                                    "parameter_code": param_code,
                                    "data": values
                                })
                
                # 避免请求过快
                time.sleep(0.5)
            
            except Exception as e:
                logger.warning(f"抓取USGS数据失败 (site={site}): {e}")
            
            if site_data["measurements"]:
                results.append(site_data)
        
        return results
    
    def fetch_weather_data(self) -> List[Dict[str, Any]]:
        """从OpenWeatherMap API抓取天气数据
        
        Returns:
            抓取到的数据列表
        """
        api_config = API_CONFIGS["weather"]
        api_key = self._get_api_key("weather")
        
        if not api_key:
            logger.warning("未配置OpenWeatherMap API密钥，跳过抓取")
            return []
        
        results = []
        
        for region in tqdm(AQUACULTURE_REGIONS, desc="抓取天气数据"):
            try:
                # 构建请求参数
                params = {
                    **api_config["default_params"],
                    "lat": region["lat"],
                    "lon": region["lon"],
                    "appid": api_key
                }
                
                # 发送请求
                response = self.session.get(api_config["base_url"], params=params)
                response.raise_for_status()
                
                # 解析响应
                data = response.json()
                
                # 提取需要的数据
                if "main" in data and "weather" in data:
                    result = {
                        "region": region["name"],
                        "water_type": region["water_type"],
                        "lat": region["lat"],
                        "lon": region["lon"],
                        "timestamp": datetime.now().isoformat(),
                        "temperature": data["main"].get("temp"),
                        "pressure": data["main"].get("pressure"),
                        "humidity": data["main"].get("humidity"),
                        "weather_main": data["weather"][0].get("main"),
                        "weather_description": data["weather"][0].get("description"),
                        "wind_speed": data["wind"].get("speed") if "wind" in data else None,
                        "wind_direction": data["wind"].get("deg") if "wind" in data else None,
                        "cloud_coverage": data["clouds"].get("all") if "clouds" in data else None,
                        "rain_1h": data["rain"].get("1h") if "rain" in data else None,
                    }
                    
                    results.append(result)
                
                # 避免请求过快
                time.sleep(0.5)
            
            except Exception as e:
                logger.warning(f"抓取天气数据失败 (region={region['name']}): {e}")
        
        return results
    
    def generate_simulated_data(self, days: int = 90) -> Dict[str, Any]:
        """生成模拟的水产养殖监测数据
        
        用于在无法获取真实API数据时使用
        
        Args:
            days: 生成多少天的数据
            
        Returns:
            模拟数据字典
        """
        # 生成日期序列
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        date_range = [start_date + timedelta(days=i) for i in range(days + 1)]
        date_strings = [d.strftime("%Y-%m-%d") for d in date_range]
        
        # 随机种子，确保可重现
        np.random.seed(42)
        
        results = {
            "metadata": {
                "type": "simulated_data",
                "generated_at": datetime.now().isoformat(),
                "days": days,
                "regions": []
            },
            "regions": {}
        }
        
        for region in AQUACULTURE_REGIONS:
            region_name = region["name"]
            water_type = region["water_type"]
            
            # 添加地区元数据
            results["metadata"]["regions"].append({
                "name": region_name,
                "lat": region["lat"],
                "lon": region["lon"],
                "water_type": water_type
            })
            
            # 生成地区数据
            region_data = {
                "water_temperature": {},
                "dissolved_oxygen": {},
                "ph": {},
                "ammonia": {},
                "nitrate": {},
                "turbidity": {}
            }
            
            # 水温 - 季节性变化 + 噪声
            # 根据水体类型设置基础温度和波动范围
            if water_type == "freshwater":
                base_temp = 20
                temp_range = 15
            elif water_type == "brackish":
                base_temp = 18
                temp_range = 12
            else:  # marine
                base_temp = 15
                temp_range = 10
            
            for i, date in enumerate(date_strings):
                # 添加季节性变化 - 假设从冬季开始
                day_of_year = (start_date + timedelta(days=i)).timetuple().tm_yday
                seasonal_component = -temp_range/2 * np.cos(2 * np.pi * day_of_year / 365)
                
                # 添加随机波动
                noise = np.random.normal(0, 1)
                
                # 计算最终温度
                temperature = base_temp + seasonal_component + noise
                region_data["water_temperature"][date] = round(temperature, 2)
                
                # 根据温度生成溶氧量 (温度越高，溶氧越低)
                do_base = 9.0 - (temperature - 15) * 0.2
                do_noise = np.random.normal(0, 0.5)
                do = max(0.1, do_base + do_noise)  # 确保不小于0.1
                region_data["dissolved_oxygen"][date] = round(do, 2)
                
                # pH值 (通常在6.5-9.0之间)
                if water_type == "freshwater":
                    ph_base = 7.0
                elif water_type == "brackish":
                    ph_base = 7.5
                else:  # marine
                    ph_base = 8.2
                
                ph_noise = np.random.normal(0, 0.2)
                ph = ph_base + ph_noise
                region_data["ph"][date] = round(ph, 2)
                
                # 氨氮 (mg/L) - 通常低于0.5是安全的
                ammonia_base = 0.1
                # 温度高时氨氮升高
                ammonia_temp_factor = max(0, (temperature - 25) / 10)
                ammonia_noise = np.random.exponential(0.1)
                ammonia = ammonia_base + ammonia_temp_factor + ammonia_noise
                region_data["ammonia"][date] = round(ammonia, 3)
                
                # 硝酸盐 (mg/L)
                nitrate_base = 5.0
                nitrate_noise = np.random.normal(0, 1.5)
                nitrate = max(0, nitrate_base + nitrate_noise)
                region_data["nitrate"][date] = round(nitrate, 2)
                
                # 浊度 (NTU)
                turbidity_base = 15.0
                # 模拟季节性降雨导致的浊度增加
                turbidity_seasonal = 10 * np.sin(2 * np.pi * day_of_year / 365 + np.pi/2)
                turbidity_noise = np.random.exponential(5)
                turbidity = max(0, turbidity_base + turbidity_seasonal + turbidity_noise)
                region_data["turbidity"][date] = round(turbidity, 1)
            
            results["regions"][region_name] = region_data
        
        return results
    
    def save_data(self, data, api_name: str):
        """保存抓取的数据
        
        Args:
            data: 要保存的数据
            api_name: API名称，用于文件命名
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"{api_name}_{timestamp}.json")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"数据已保存到 {output_file}")
            
            # 如果是模拟数据，额外保存CSV格式
            if api_name == "simulated":
                self._save_simulated_csv(data)
        
        except Exception as e:
            logger.error(f"保存数据失败: {e}")
    
    def _save_simulated_csv(self, data: Dict[str, Any]):
        """将模拟数据保存为CSV格式
        
        Args:
            data: 模拟数据字典
        """
        # 为每种参数创建CSV文件
        parameters = ["water_temperature", "dissolved_oxygen", "ph", "ammonia", "nitrate", "turbidity"]
        
        for param in parameters:
            try:
                output_file = os.path.join(self.output_dir, f"simulated_{param}.csv")
                
                # 创建DataFrame
                df_data = {}
                
                # 获取日期列表 (所有地区相同)
                first_region = next(iter(data["regions"]))
                dates = list(data["regions"][first_region][param].keys())
                df_data["date"] = dates
                
                # 添加每个地区的数据
                for region_name, region_data in data["regions"].items():
                    df_data[region_name] = [region_data[param].get(date, None) for date in dates]
                
                # 创建DataFrame并保存
                df = pd.DataFrame(df_data)
                df.to_csv(output_file, index=False)
                logger.info(f"CSV数据已保存到 {output_file}")
            
            except Exception as e:
                logger.error(f"保存CSV数据失败 (参数={param}): {e}")

def main():
    parser = argparse.ArgumentParser(description="抓取海洋环境数据")
    parser.add_argument("--output_dir", default="datasets/ocean_data", 
                        help="输出目录")
    parser.add_argument("--days", type=int, default=30,
                        help="获取过去多少天的数据")
    parser.add_argument("--api", default="simulated",
                        help="使用的API，多个用逗号分隔，可选值: noaa,usgs,weather,simulated")
    parser.add_argument("--weather_api_key", default=None,
                        help="OpenWeatherMap API密钥")
    
    args = parser.parse_args()
    
    # 解析API列表
    apis = args.api.split(",")
    
    # API密钥
    api_keys = {}
    if args.weather_api_key:
        api_keys["weather"] = args.weather_api_key
    
    # 创建抓取器
    crawler = OceanDataCrawler(args.output_dir, api_keys)
    
    # 抓取数据
    for api in apis:
        api = api.strip().lower()
        
        if api == "noaa":
            logger.info("开始抓取NOAA数据...")
            data = crawler.fetch_noaa_data(args.days)
            crawler.save_data(data, "noaa")
        
        elif api == "usgs":
            logger.info("开始抓取USGS数据...")
            data = crawler.fetch_usgs_data(args.days)
            crawler.save_data(data, "usgs")
        
        elif api == "weather":
            logger.info("开始抓取天气数据...")
            data = crawler.fetch_weather_data()
            crawler.save_data(data, "weather")
        
        elif api == "simulated":
            logger.info("生成模拟数据...")
            data = crawler.generate_simulated_data(args.days)
            crawler.save_data(data, "simulated")
        
        else:
            logger.warning(f"未知API: {api}")

if __name__ == "__main__":
    main()
