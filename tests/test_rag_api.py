"""
RAG API单元测试
详见 docs/api.md
"""
import unittest
import os
import shutil
import tempfile
import asyncio
from unittest.mock import patch, MagicMock

from src.api.rag_api import RAGService, get_rag_service


class TestRAGService(unittest.TestCase):
    """RAG服务单元测试"""
    
    def setUp(self):
        """测试前准备工作"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "data")
        self.index_dir = os.path.join(self.temp_dir, "index")
        
        # 创建测试数据目录
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 创建测试文件
        with open(os.path.join(self.data_dir, "test1.txt"), "w", encoding="utf-8") as f:
            f.write("鱼病测试文档1。这是一个关于水生动物疾病的测试文档。鲤鱼常见病包括烂鳃病、肠炎和水霉病。")
        
        with open(os.path.join(self.data_dir, "test2.txt"), "w", encoding="utf-8") as f:
            f.write("鱼病测试文档2。草鱼常见病包括出血病、肠炎和烂尾病。烂鳃病主要症状为鳃部发白、烂鳃。")
    
    def tearDown(self):
        """测试后清理工作"""
        # 删除临时目录
        shutil.rmtree(self.temp_dir)
    
    @patch("llama_index.core.VectorStoreIndex")
    def test_init_service(self, mock_vector_index):
        """测试初始化RAG服务"""
        # 设置模拟对象
        mock_index = MagicMock()
        mock_vector_index.from_documents.return_value = mock_index
        
        # 创建RAG服务
        rag_service = RAGService(
            data_dir=self.data_dir,
            index_path=self.index_dir,
            top_k=3
        )
        
        # 验证参数
        self.assertEqual(rag_service.data_dir, self.data_dir)
        self.assertEqual(rag_service.index_path, self.index_dir)
        self.assertEqual(rag_service.top_k, 3)
    
    @patch("llama_index.core.VectorStoreIndex")
    @patch("llama_index.vector_stores.faiss.FaissVectorStore")
    def test_ask(self, mock_faiss, mock_vector_index):
        """测试问答功能"""
        # 设置模拟对象
        mock_index = MagicMock()
        mock_retriever = MagicMock()
        mock_query_engine = MagicMock()
        mock_response = MagicMock()
        
        mock_node = MagicMock()
        mock_node.node.text = "测试文档内容"
        mock_retriever.retrieve.return_value = [mock_node]
        
        mock_index.as_retriever.return_value = mock_retriever
        mock_index.as_query_engine.return_value = mock_query_engine
        mock_query_engine.query.return_value = mock_response
        mock_response.response = "测试回答内容"
        
        mock_vector_index.from_documents.return_value = mock_index
        
        # 创建RAG服务
        rag_service = RAGService(
            data_dir=self.data_dir,
            index_path=self.index_dir
        )
        
        # 执行问答
        result = asyncio.run(rag_service.ask("鲤鱼有哪些常见病？"))
        
        # 验证结果
        self.assertIn("answer", result)
        self.assertIn("source_documents", result)
        self.assertIn("elapsed_time", result)
        self.assertEqual(result["answer"], "测试回答内容")
        self.assertEqual(result["source_documents"], ["测试文档内容"])
    
    @patch("llama_index.core.VectorStoreIndex")
    def test_diagnose(self, mock_vector_index):
        """测试诊断功能"""
        # 设置模拟对象
        mock_index = MagicMock()
        mock_vector_index.from_documents.return_value = mock_index
        
        # 模拟ask方法返回值
        def mock_ask(query, response_mode):
            return {
                "answer": "这可能是烂鳃病，建议使用XXX药物治疗。",
                "source_documents": ["测试文档内容"],
                "elapsed_time": 0.5
            }
        
        # 创建RAG服务，并替换ask方法
        rag_service = RAGService(
            data_dir=self.data_dir,
            index_path=self.index_dir
        )
        rag_service.ask = mock_ask
        
        # 执行诊断
        symptoms = ["鳃部发白", "活动减少"]
        species = "鲤鱼"
        result = asyncio.run(rag_service.diagnose(symptoms, species))
        
        # 验证结果
        self.assertIn("answer", result)
        self.assertIn("source_documents", result)
        self.assertIn("symptoms", result)
        self.assertIn("species", result)
        self.assertIn("diagnosis_time", result)
        self.assertEqual(result["symptoms"], symptoms)
        self.assertEqual(result["species"], species)
    
    @patch("llama_index.core.VectorStoreIndex")
    def test_singleton(self, mock_vector_index):
        """测试单例模式"""
        # 设置模拟对象
        mock_index = MagicMock()
        mock_vector_index.from_documents.return_value = mock_index
        
        # 获取两个实例
        service1 = get_rag_service(data_dir=self.data_dir)
        service2 = get_rag_service(data_dir=self.data_dir)
        
        # 验证是否为同一个实例
        self.assertIs(service1, service2)


if __name__ == "__main__":
    unittest.main() 
