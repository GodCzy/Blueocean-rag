#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
parse_fish_data.py - 解析水生动物疾病文档

该脚本用于处理水生动物疾病文本文档，提取结构化信息，
并进行实体识别、文本切块等预处理工作。

用法:
    python parse_fish_data.py --input_dir datasets/fish_docs --output_dir datasets/processed --chunk_size 512

作者: 成员B (数据工程师)
"""

import os
import re
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import spacy
import numpy as np
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("parse_fish_data")

# 加载标签数据
def load_tags(tags_file: str) -> Dict[str, Any]:
    """加载标签体系"""
    try:
        with open(tags_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载标签文件失败: {e}")
        return {}

# 实体识别相关函数
class NERProcessor:
    """命名实体识别处理器"""
    
    def __init__(self, tags_data: Dict[str, Any]):
        """初始化NER处理器"""
        self.tags_data = tags_data
        self.categories = tags_data.get("categories", {})
        
        # 实体词典，用于基于规则的NER
        self.entity_dict = self._build_entity_dict()
        
        # 尝试加载spaCy模型（如果可用）
        try:
            self.nlp = spacy.load("zh_core_web_sm")
            self.use_spacy = True
            logger.info("加载spaCy模型成功")
        except Exception as e:
            logger.warning(f"加载spaCy模型失败: {e}, 将使用规则匹配进行实体识别")
            self.use_spacy = False
    
    def _build_entity_dict(self) -> Dict[str, List[Tuple[str, str]]]:
        """构建实体词典，用于规则匹配"""
        entity_dict = {}
        
        # 遍历所有类别及其标签
        for category_key, category_data in self.categories.items():
            if "tags" in category_data:
                for tag in category_data["tags"]:
                    if tag not in entity_dict:
                        entity_dict[tag] = []
                    entity_dict[tag].append((category_key, tag))
            
            # 处理子类别
            if "subcategories" in category_data:
                for subcategory_key, subcategory_data in category_data["subcategories"].items():
                    if "tags" in subcategory_data:
                        for tag in subcategory_data["tags"]:
                            if tag not in entity_dict:
                                entity_dict[tag] = []
                            # 使用类别.子类别格式
                            entity_dict[tag].append((f"{category_key}.{subcategory_key}", tag))
        
        return entity_dict
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """从文本中提取实体"""
        entities = []
        
        if self.use_spacy:
            # 使用spaCy进行实体识别，并结合规则匹配
            doc = self.nlp(text)
            
            # spaCy实体
            for ent in doc.ents:
                # 尝试匹配已知实体
                if ent.text in self.entity_dict:
                    for category, value in self.entity_dict[ent.text]:
                        entities.append({
                            "text": ent.text,
                            "start": ent.start_char,
                            "end": ent.end_char,
                            "category": category,
                            "value": value,
                            "source": "spacy+rules"
                        })
                else:
                    # 未知实体，仅使用spaCy标签
                    entities.append({
                        "text": ent.text,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "category": ent.label_,
                        "value": ent.text,
                        "source": "spacy"
                    })
        
        # 规则匹配
        for entity_text, category_info in self.entity_dict.items():
            # 查找所有出现位置
            for match in re.finditer(re.escape(entity_text), text):
                for category, value in category_info:
                    entities.append({
                        "text": entity_text,
                        "start": match.start(),
                        "end": match.end(),
                        "category": category,
                        "value": value,
                        "source": "rules"
                    })
        
        # 去重并排序
        unique_entities = []
        seen = set()
        for entity in entities:
            # 使用实体文本、位置和类别作为唯一标识
            key = (entity["text"], entity["start"], entity["category"])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        # 按起始位置排序
        return sorted(unique_entities, key=lambda x: x["start"])
    
    def extract_relations(self, entities: List[Dict[str, Any]], 
                          context_window: int = 100) -> List[Dict[str, Any]]:
        """从实体中提取关系
        
        Args:
            entities: 实体列表
            context_window: 上下文窗口大小，用于判断两个实体是否有关联
            
        Returns:
            关系列表
        """
        relations = []
        relation_definitions = self.tags_data.get("relations", [])
        
        # 关系定义字典
        relation_dict = {r["name"]: r for r in relation_definitions}
        
        # 按类别组织实体
        entities_by_category = {}
        for entity in entities:
            category = entity["category"]
            if category not in entities_by_category:
                entities_by_category[category] = []
            entities_by_category[category].append(entity)
        
        # 遍历关系定义，寻找符合条件的实体对
        for relation in relation_definitions:
            source_category = relation["source"]
            target_category = relation["target"]
            relation_name = relation["name"]
            
            # 检查是否有相关类别的实体
            if source_category not in entities_by_category or target_category not in entities_by_category:
                continue
            
            # 遍历可能的实体对
            for source_entity in entities_by_category[source_category]:
                for target_entity in entities_by_category[target_category]:
                    # 实体不能自己关联自己
                    if source_entity["start"] == target_entity["start"]:
                        continue
                    
                    # 检查两个实体是否在上下文窗口内
                    distance = abs(source_entity["start"] - target_entity["start"])
                    if distance <= context_window:
                        relations.append({
                            "relation_type": relation_name,
                            "source": source_entity,
                            "target": target_entity,
                            "confidence": 1.0 - (distance / context_window)  # 根据距离计算置信度
                        })
        
        return relations

# 文档解析函数
class DocumentParser:
    """文档解析类"""
    
    def __init__(self, ner_processor: NERProcessor, chunk_size: int = 512, chunk_overlap: int = 128):
        """初始化文档解析器
        
        Args:
            ner_processor: 实体识别处理器
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
        """
        self.ner_processor = ner_processor
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 段落分隔模式
        self.paragraph_pattern = re.compile(r'\n\s*\n')
        
        # 疾病文档字段模式
        self.field_patterns = {
            "name": r"疾病名称[:：]\s*(.*?)(?:\n|$)",
            "type": r"类型[:：]\s*(.*?)(?:\n|$)",
            "pathogen": r"病原体[:：]\s*(.*?)(?:\n|$)",
            "symptoms": r"(?:主要症状|症状表现)[:：](.*?)(?:\n\s*\n|$)",
            "affected_species": r"(?:易感鱼类|宿主)[:：](.*?)(?:\n\s*\n|$)",
            "seasons": r"(?:发病季节|流行季节)[:：](.*?)(?:\n\s*\n|$)",
            "environmental_factors": r"环境因素[:：](.*?)(?:\n\s*\n|$)",
            "diagnosis_methods": r"诊断方法[:：](.*?)(?:\n\s*\n|$)",
            "treatments": r"治疗方法[:：](.*?)(?:\n\s*\n|$)",
            "medications": r"(?:推荐用药|用药)[:：](.*?)(?:\n\s*\n|$)",
            "prevention": r"预防措施[:：](.*?)(?:\n\s*\n|$)",
            "notes": r"(?:注意事项|备注)[:：](.*?)(?:\n\s*\n|$)",
        }
    
    def parse_document(self, file_path: str) -> Dict[str, Any]:
        """解析文档文件
        
        Args:
            file_path: 文档文件路径
            
        Returns:
            解析后的文档数据
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取字段
            fields = self._extract_fields(content)
            
            # 切块
            chunks = self._chunk_text(content)
            
            # 实体识别
            entities = self.ner_processor.extract_entities(content)
            
            # 关系提取
            relations = self.ner_processor.extract_relations(entities)
            
            # 生成解析结果
            result = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "fields": fields,
                "chunks": chunks,
                "entities": entities,
                "relations": relations,
                "raw_text": content
            }
            
            return result
        
        except Exception as e:
            logger.error(f"解析文档 {file_path} 失败: {e}")
            return {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "error": str(e)
            }
    
    def _extract_fields(self, text: str) -> Dict[str, Any]:
        """提取文本中的字段
        
        Args:
            text: 要处理的文本
            
        Returns:
            提取的字段字典
        """
        fields = {}
        
        # 使用正则表达式匹配各个字段
        for field_name, pattern in self.field_patterns.items():
            match = re.search(pattern, text, re.DOTALL)
            if match:
                # 清理提取的内容
                content = match.group(1).strip()
                
                # 对于列表字段，按行分割并清理
                if field_name in ["symptoms", "environmental_factors", 
                                 "diagnosis_methods", "treatments", 
                                 "medications", "prevention"]:
                    # 尝试提取编号列表项
                    items = re.findall(r'\d+\.\s*(.*?)(?:\n|$)', content)
                    if items:
                        fields[field_name] = [item.strip() for item in items]
                    else:
                        # 如果没有编号，按行分割
                        fields[field_name] = [item.strip() for item in content.split('\n') if item.strip()]
                else:
                    fields[field_name] = content
        
        return fields
    
    def _chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """将文本分割成块
        
        Args:
            text: 要分块的文本
            
        Returns:
            文本块列表，每个块包含文本内容、起始位置和编号
        """
        chunks = []
        
        # 按段落分割
        paragraphs = self.paragraph_pattern.split(text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        current_chunk = ""
        chunk_start = 0
        
        for paragraph in paragraphs:
            # 如果添加这段会超出最大块大小，且当前块不为空，则保存当前块
            if len(current_chunk) + len(paragraph) + 1 > self.chunk_size and current_chunk:
                chunks.append({
                    "text": current_chunk,
                    "start": chunk_start,
                    "end": chunk_start + len(current_chunk),
                    "chunk_id": len(chunks)
                })
                
                # 计算新块的起始位置，考虑重叠
                overlap_char_count = min(len(current_chunk), self.chunk_overlap)
                current_chunk = current_chunk[-overlap_char_count:] if overlap_char_count > 0 else ""
                chunk_start = chunk_start + len(current_chunk) - overlap_char_count
            
            # 添加段落到当前块
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
        
        # 添加最后一个块
        if current_chunk:
            chunks.append({
                "text": current_chunk,
                "start": chunk_start,
                "end": chunk_start + len(current_chunk),
                "chunk_id": len(chunks)
            })
        
        return chunks

def process_directory(input_dir: str, output_dir: str, tags_file: str, 
                     chunk_size: int = 512, chunk_overlap: int = 128) -> None:
    """处理目录中的所有文档
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        tags_file: 标签文件
        chunk_size: 块大小
        chunk_overlap: 块重叠大小
    """
    # 加载标签
    tags_data = load_tags(tags_file)
    if not tags_data:
        logger.error(f"无法加载标签文件: {tags_file}")
        return
    
    # 创建NER处理器和文档解析器
    ner_processor = NERProcessor(tags_data)
    document_parser = DocumentParser(ner_processor, chunk_size, chunk_overlap)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有文本文件
    input_path = Path(input_dir)
    text_files = list(input_path.glob("**/*.txt"))
    
    if not text_files:
        logger.warning(f"在 {input_dir} 中未找到文本文件")
        return
    
    logger.info(f"找到 {len(text_files)} 个文本文件")
    
    # 处理每个文件
    results = []
    for file_path in tqdm(text_files, desc="处理文档"):
        # 解析文档
        result = document_parser.parse_document(str(file_path))
        
        # 保存解析结果
        output_path = Path(output_dir) / f"{file_path.stem}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 收集结果
        results.append({
            "file_name": file_path.name,
            "output_file": str(output_path),
            "entities_count": len(result.get("entities", [])),
            "chunks_count": len(result.get("chunks", [])),
            "has_error": "error" in result
        })
    
    # 保存处理摘要
    summary_path = Path(output_dir) / "processing_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        summary = {
            "processed_files": len(results),
            "total_entities": sum(r["entities_count"] for r in results),
            "total_chunks": sum(r["chunks_count"] for r in results),
            "error_files": sum(1 for r in results if r["has_error"]),
            "files": results
        }
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info(f"处理完成，结果保存在 {output_dir}")
    logger.info(f"处理了 {len(results)} 个文件，提取了 {summary['total_entities']} 个实体，"
               f"生成了 {summary['total_chunks']} 个文本块")

def main():
    parser = argparse.ArgumentParser(description="解析水生动物疾病文档")
    parser.add_argument("--input_dir", default="datasets/fish_docs", 
                        help="输入文档目录")
    parser.add_argument("--output_dir", default="datasets/processed", 
                        help="输出目录")
    parser.add_argument("--tags_file", default="datasets/tags.json",
                        help="标签文件路径")
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="文本块大小")
    parser.add_argument("--chunk_overlap", type=int, default=128,
                        help="文本块重叠大小")
    
    args = parser.parse_args()
    
    # 处理文档
    process_directory(
        args.input_dir, 
        args.output_dir, 
        args.tags_file,
        args.chunk_size,
        args.chunk_overlap
    )

if __name__ == "__main__":
    main()
