#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tag_and_index.py - 为解析后的文档添加标签并构建向量索引

该脚本用于处理已经解析的水生动物疾病文档，根据标签体系进行分类标注，
并构建向量索引用于后续的检索增强生成。

用法:
    python tag_and_index.py --input_dir datasets/processed --output_dir datasets/indexed --model_name BAAI/bge-large-zh-v1.5

作者: 成员B (数据工程师)
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from tqdm import tqdm
import faiss
import pickle
import time
from datetime import datetime

# 尝试导入不同的嵌入模型库
try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False

try:
    from FlagEmbedding import FlagModel
    HAVE_FLAGEMBEDDING = True
except ImportError:
    HAVE_FLAGEMBEDDING = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tag_and_index")

class Embedder:
    """文本嵌入器类，负责生成文本向量表示"""
    
    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5", device: str = "cpu"):
        """初始化嵌入器
        
        Args:
            model_name: 模型名称
            device: 设备类型 ('cpu' 或 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.dimension = None
        
        self._load_model()
    
    def _load_model(self):
        """加载嵌入模型"""
        try:
            # 尝试使用FlagEmbedding
            if HAVE_FLAGEMBEDDING and ("bge" in self.model_name.lower()):
                logger.info(f"使用FlagEmbedding加载模型: {self.model_name}")
                self.model = FlagModel(
                    self.model_name,
                    query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                    device=self.device
                )
                
                # 获取模型维度
                if "large" in self.model_name.lower():
                    self.dimension = 1024
                elif "base" in self.model_name.lower():
                    self.dimension = 768
                else:
                    # 通过计算一个示例来获取维度
                    sample_vector = self.model.encode_queries(["测试维度"])[0]
                    self.dimension = len(sample_vector)
                
                logger.info(f"模型加载成功，向量维度: {self.dimension}")
                return
            
            # 尝试使用SentenceTransformers
            if HAVE_SENTENCE_TRANSFORMERS:
                logger.info(f"使用SentenceTransformers加载模型: {self.model_name}")
                self.model = SentenceTransformer(self.model_name, device=self.device)
                
                # 获取模型维度
                sample_vector = self.model.encode("测试维度")
                self.dimension = len(sample_vector)
                
                logger.info(f"模型加载成功，向量维度: {self.dimension}")
                return
            
            # 没有可用的嵌入模型库
            raise ImportError("未找到可用的嵌入模型库。请安装 sentence-transformers 或 FlagEmbedding")
            
        except Exception as e:
            logger.error(f"加载嵌入模型失败: {e}")
            raise
    
    def encode_text(self, text: str) -> np.ndarray:
        """编码单条文本
        
        Args:
            text: 要编码的文本
            
        Returns:
            文本向量表示
        """
        if not text or not text.strip():
            logger.warning("空文本无法编码，将返回零向量")
            return np.zeros(self.dimension)
        
        try:
            if isinstance(self.model, SentenceTransformer):
                return self.model.encode(text)
            else:  # FlagModel
                return self.model.encode_queries([text])[0]
        except Exception as e:
            logger.error(f"编码文本时出错: {e}")
            return np.zeros(self.dimension)
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """批量编码多条文本
        
        Args:
            texts: 要编码的文本列表
            
        Returns:
            文本向量表示数组
        """
        if not texts:
            return np.array([])
        
        # 过滤空文本
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return np.zeros((0, self.dimension))
        
        try:
            if isinstance(self.model, SentenceTransformer):
                return self.model.encode(valid_texts)
            else:  # FlagModel
                return np.array(self.model.encode_corpus(valid_texts))
        except Exception as e:
            logger.error(f"批量编码文本时出错: {e}")
            return np.zeros((len(valid_texts), self.dimension))

class IndexBuilder:
    """向量索引构建器"""
    
    def __init__(self, embedder: Embedder, output_dir: str):
        """初始化索引构建器
        
        Args:
            embedder: 文本嵌入器
            output_dir: 输出目录
        """
        self.embedder = embedder
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 索引和元数据
        self.index = None
        self.documents = []
        self.document_map = {}  # 文档ID到文档信息的映射
    
    def create_index(self):
        """创建FAISS索引"""
        try:
            # 创建L2距离索引
            self.index = faiss.IndexFlatL2(self.embedder.dimension)
            logger.info(f"创建了FAISS IndexFlatL2索引, 维度: {self.embedder.dimension}")
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            raise
    
    def add_document(self, document: Dict[str, Any], embedding_field: str = "text"):
        """添加文档到索引
        
        Args:
            document: 文档数据
            embedding_field: 用于生成向量的字段名
        """
        # 为文档分配ID
        doc_id = len(self.documents)
        
        # 提取文本并生成向量
        text = document.get(embedding_field, "")
        vector = self.embedder.encode_text(text)
        
        # 添加到索引
        self.index.add(np.array([vector]).astype('float32'))
        
        # 保存文档和映射
        self.documents.append(document)
        self.document_map[doc_id] = document
        
        return doc_id
    
    def add_documents(self, documents: List[Dict[str, Any]], embedding_field: str = "text"):
        """批量添加文档到索引
        
        Args:
            documents: 文档列表
            embedding_field: 用于生成向量的字段名
        """
        if not documents:
            return []
        
        # 提取文本
        texts = [doc.get(embedding_field, "") for doc in documents]
        
        # 生成向量
        vectors = self.embedder.encode_texts(texts)
        
        # 添加到索引
        self.index.add(vectors.astype('float32'))
        
        # 为文档分配ID并保存
        start_id = len(self.documents)
        doc_ids = list(range(start_id, start_id + len(documents)))
        
        for i, doc_id in enumerate(doc_ids):
            self.documents.append(documents[i])
            self.document_map[doc_id] = documents[i]
        
        return doc_ids
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """搜索与查询最相关的文档
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            相关文档列表
        """
        if not self.index:
            logger.error("索引未创建")
            return []
        
        try:
            # 生成查询向量
            query_vector = self.embedder.encode_text(query)
            
            # 搜索最相似的向量
            distances, indices = self.index.search(
                np.array([query_vector]).astype('float32'), k
            )
            
            # 构建结果
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx < len(self.documents):  # 有效索引
                    result = {
                        "document": self.document_map[idx],
                        "distance": float(distances[0][i]),
                        "score": 1.0 / (1.0 + float(distances[0][i]))  # 转换距离为相似度分数
                    }
                    results.append(result)
            
            return results
        
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def save_index(self):
        """保存索引和相关数据"""
        if not self.index:
            logger.error("索引未创建，无法保存")
            return False
        
        try:
            # 保存索引文件
            index_file = os.path.join(self.output_dir, "faiss_index.bin")
            faiss.write_index(self.index, index_file)
            logger.info(f"索引已保存到 {index_file}")
            
            # 保存文档信息
            doc_file = os.path.join(self.output_dir, "documents.pkl")
            with open(doc_file, 'wb') as f:
                pickle.dump(self.documents, f)
            
            # 保存文档映射
            map_file = os.path.join(self.output_dir, "document_map.pkl")
            with open(map_file, 'wb') as f:
                pickle.dump(self.document_map, f)
            
            # 保存元数据
            metadata = {
                "dimension": self.embedder.dimension,
                "model_name": self.embedder.model_name,
                "document_count": len(self.documents),
                "created_at": datetime.now().isoformat(),
                "index_type": "IndexFlatL2"
            }
            
            meta_file = os.path.join(self.output_dir, "metadata.json")
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"保存了 {len(self.documents)} 个文档和元数据")
            return True
        
        except Exception as e:
            logger.error(f"保存索引失败: {e}")
            return False
    
    def load_index(self):
        """加载索引和相关数据"""
        try:
            # 加载索引文件
            index_file = os.path.join(self.output_dir, "faiss_index.bin")
            if os.path.exists(index_file):
                self.index = faiss.read_index(index_file)
                logger.info(f"从 {index_file} 加载了索引")
            else:
                logger.warning(f"索引文件 {index_file} 不存在，将创建新索引")
                self.create_index()
            
            # 加载文档信息
            doc_file = os.path.join(self.output_dir, "documents.pkl")
            if os.path.exists(doc_file):
                with open(doc_file, 'rb') as f:
                    self.documents = pickle.load(f)
            
            # 加载文档映射
            map_file = os.path.join(self.output_dir, "document_map.pkl")
            if os.path.exists(map_file):
                with open(map_file, 'rb') as f:
                    self.document_map = pickle.load(f)
            
            logger.info(f"加载了 {len(self.documents)} 个文档")
            return True
        
        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            return False

class Tagger:
    """文档标记器，为文档添加分类标签"""
    
    def __init__(self, tags_file: str):
        """初始化标记器
        
        Args:
            tags_file: 标签体系文件路径
        """
        self.tags_file = tags_file
        self.tags_data = self._load_tags()
        
    def _load_tags(self) -> Dict[str, Any]:
        """加载标签数据"""
        try:
            with open(self.tags_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"从 {self.tags_file} 加载了标签数据")
            return data
        except Exception as e:
            logger.error(f"加载标签数据失败: {e}")
            return {}
    
    def classify_disease(self, disease_info: Dict[str, Any]) -> Dict[str, Any]:
        """根据病信息进行疾病分类
        
        Args:
            disease_info: 疾病字段信息
            
        Returns:
            添加了分类标签的疾病信息
        """
        if not disease_info or not self.tags_data:
            return disease_info
        
        # 复制原始数据
        result = disease_info.copy()
        
        # 初始化标签字段
        if "tags" not in result:
            result["tags"] = {}
        
        # 疾病类型分类
        disease_type = result.get("type", "")
        if disease_type:
            disease_types = self.tags_data.get("categories", {}).get("disease_types", {}).get("tags", [])
            for dtype in disease_types:
                if dtype in disease_type:
                    result["tags"]["disease_type"] = dtype
                    break
        
        # 提取症状标签
        symptoms = result.get("symptoms", [])
        if symptoms:
            known_symptoms = self.tags_data.get("categories", {}).get("symptoms", {}).get("tags", [])
            matched_symptoms = []
            
            for symptom_text in symptoms:
                for known_symptom in known_symptoms:
                    if known_symptom in symptom_text:
                        matched_symptoms.append(known_symptom)
            
            if matched_symptoms:
                result["tags"]["symptoms"] = matched_symptoms
        
        # 提取鱼类种类
        species_text = result.get("affected_species", "")
        if species_text:
            known_species = self.tags_data.get("categories", {}).get("fish_species", {}).get("tags", [])
            matched_species = []
            
            for species in known_species:
                if species in species_text:
                    matched_species.append(species)
            
            if matched_species:
                result["tags"]["fish_species"] = matched_species
        
        # 提取季节信息
        seasons_text = result.get("seasons", "")
        if seasons_text:
            known_seasons = self.tags_data.get("categories", {}).get("seasons", {}).get("tags", [])
            matched_seasons = []
            
            for season in known_seasons:
                if season in seasons_text:
                    matched_seasons.append(season)
            
            if matched_seasons:
                result["tags"]["seasons"] = matched_seasons
        
        # 提取治疗方法
        treatments = result.get("treatments", [])
        if treatments:
            known_treatments = self.tags_data.get("categories", {}).get("treatments", {}).get("tags", [])
            matched_treatments = []
            
            for treatment_text in treatments:
                for known_treatment in known_treatments:
                    if known_treatment in treatment_text:
                        matched_treatments.append(known_treatment)
            
            if matched_treatments:
                result["tags"]["treatments"] = matched_treatments
        
        # 提取用药信息
        medications = result.get("medications", [])
        if medications:
            known_medications = self.tags_data.get("categories", {}).get("medications", {}).get("tags", [])
            matched_medications = []
            
            for medication_text in medications:
                for known_medication in known_medications:
                    if known_medication in medication_text:
                        matched_medications.append(known_medication)
            
            if matched_medications:
                result["tags"]["medications"] = matched_medications
        
        return result

def process_documents(input_dir: str, output_dir: str, tags_file: str, 
                     model_name: str, device: str = "cpu") -> None:
    """处理解析后的文档，添加标签并构建索引
    
    Args:
        input_dir: 输入目录，包含已解析的文档
        output_dir: 输出目录
        tags_file: 标签文件路径
        model_name: 嵌入模型名称
        device: 设备类型
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化标记器
    tagger = Tagger(tags_file)
    
    # 初始化嵌入器和索引构建器
    embedder = Embedder(model_name, device)
    index_builder = IndexBuilder(embedder, os.path.join(output_dir, "index"))
    index_builder.create_index()
    
    # 获取输入文件
    input_path = Path(input_dir)
    json_files = list(input_path.glob("**/*.json"))
    
    # 过滤掉处理摘要文件
    json_files = [f for f in json_files if f.name != "processing_summary.json"]
    
    if not json_files:
        logger.warning(f"在 {input_dir} 中未找到JSON文件")
        return
    
    logger.info(f"找到 {len(json_files)} 个JSON文件")
    
    # 处理每个文件
    processed_docs = []
    chunks_for_indexing = []
    
    for file_path in tqdm(json_files, desc="处理文档"):
        try:
            # 加载文档
            with open(file_path, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)
            
            # 检查是否有错误
            if "error" in doc_data:
                logger.warning(f"跳过有错误的文档: {file_path}")
                continue
            
            # 提取字段数据
            fields = doc_data.get("fields", {})
            
            # 使用标记器添加分类标签
            tagged_fields = tagger.classify_disease(fields)
            doc_data["fields"] = tagged_fields
            
            # 处理文本块，为每个块添加元数据
            chunks = doc_data.get("chunks", [])
            for chunk in chunks:
                # 添加文档名称和标签信息
                chunk["document_name"] = doc_data.get("file_name", "")
                chunk["document_fields"] = tagged_fields
                chunk["tags"] = tagged_fields.get("tags", {})
                
                # 收集用于索引的块
                chunks_for_indexing.append(chunk)
            
            # 保存处理后的文档
            output_file = Path(output_dir) / "documents" / file_path.name
            os.makedirs(output_file.parent, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(doc_data, f, ensure_ascii=False, indent=2)
            
            # 收集处理后的文档
            processed_docs.append({
                "file_name": file_path.name,
                "tags": tagged_fields.get("tags", {}),
                "chunks_count": len(chunks),
                "output_file": str(output_file)
            })
            
        except Exception as e:
            logger.error(f"处理文档 {file_path} 时出错: {e}")
    
    # 为文本块构建索引
    logger.info(f"为 {len(chunks_for_indexing)} 个文本块构建索引")
    
    # 批量添加到索引
    batch_size = 100
    for i in tqdm(range(0, len(chunks_for_indexing), batch_size), desc="构建索引"):
        batch = chunks_for_indexing[i:i+batch_size]
        index_builder.add_documents(batch, embedding_field="text")
    
    # 保存索引
    index_builder.save_index()
    
    # 保存处理摘要
    summary_path = Path(output_dir) / "indexing_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        summary = {
            "processed_files": len(processed_docs),
            "total_chunks_indexed": len(chunks_for_indexing),
            "model_name": model_name,
            "embedding_dimension": embedder.dimension,
            "timestamp": datetime.now().isoformat(),
            "files": processed_docs
        }
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info(f"处理完成，结果保存在 {output_dir}")
    logger.info(f"处理了 {len(processed_docs)} 个文档，索引了 {len(chunks_for_indexing)} 个文本块")

def main():
    parser = argparse.ArgumentParser(description="为解析后的文档添加标签并构建向量索引")
    parser.add_argument("--input_dir", default="datasets/processed", 
                        help="输入目录，包含已解析的文档")
    parser.add_argument("--output_dir", default="datasets/indexed", 
                        help="输出目录")
    parser.add_argument("--tags_file", default="datasets/tags.json",
                        help="标签文件路径")
    parser.add_argument("--model_name", default="BAAI/bge-large-zh-v1.5",
                        help="嵌入模型名称")
    parser.add_argument("--device", default="cpu",
                        help="设备类型 (cpu 或 cuda)")
    
    args = parser.parse_args()
    
    # 处理文档
    process_documents(
        args.input_dir, 
        args.output_dir, 
        args.tags_file,
        args.model_name,
        args.device
    )

if __name__ == "__main__":
    main()
