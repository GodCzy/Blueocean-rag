"""
RAG Router单元测试
"""
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from src.routers.rag_router import rag, AskRequest, DiagnoseRequest
from src.api.rag_api import RAGService


class TestRAGRouter(unittest.TestCase):
    """RAG Router单元测试"""
    
    def setUp(self):
        """测试前准备工作"""
        # 创建FastAPI应用
        self.app = FastAPI()
        self.app.include_router(rag)
        self.client = TestClient(self.app)
    
    @patch("src.routers.rag_router.get_rag")
    def test_ask_endpoint(self, mock_get_rag):
        """测试问答接口"""
        # 模拟RAG服务返回值
        mock_rag_service = MagicMock(spec=RAGService)
        mock_rag_service.ask = AsyncMock(return_value={
            "answer": "鲤鱼的常见病包括烂鳃病、肠炎和水霉病。",
            "source_documents": ["测试文档内容"],
            "elapsed_time": 0.5
        })
        mock_get_rag.return_value = mock_rag_service
        
        # 发送请求
        request_data = {
            "query": "鲤鱼有哪些常见病？",
            "response_mode": "compact"
        }
        response = self.client.post("/rag/ask", json=request_data)
        
        # 验证结果
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("answer", data)
        self.assertIn("source_documents", data)
        self.assertIn("elapsed_time", data)
        self.assertEqual(data["answer"], "鲤鱼的常见病包括烂鳃病、肠炎和水霉病。")
        
        # 验证服务调用
        mock_rag_service.ask.assert_awaited_once_with(
            "鲤鱼有哪些常见病？",
            "compact"
        )
    
    @patch("src.routers.rag_router.get_rag")
    def test_diagnose_endpoint(self, mock_get_rag):
        """测试诊断接口"""
        # 模拟RAG服务返回值
        mock_rag_service = MagicMock(spec=RAGService)
        mock_rag_service.diagnose = AsyncMock(return_value={
            "answer": "根据症状判断，这可能是烂鳃病。建议使用高锰酸钾溶液进行消毒。",
            "source_documents": ["测试文档内容"],
            "symptoms": ["鳃部发白", "活动减少"],
            "species": "鲤鱼",
            "elapsed_time": 0.5,
            "diagnosis_time": 0.3
        })
        mock_get_rag.return_value = mock_rag_service
        
        # 发送请求
        request_data = {
            "symptoms": ["鳃部发白", "活动减少"],
            "species": "鲤鱼"
        }
        response = self.client.post("/rag/disease/diagnose", json=request_data)
        
        # 验证结果
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("answer", data)
        self.assertIn("source_documents", data)
        self.assertIn("symptoms", data)
        self.assertIn("species", data)
        self.assertEqual(data["symptoms"], ["鳃部发白", "活动减少"])
        self.assertEqual(data["species"], "鲤鱼")
        
        # 验证服务调用
        mock_rag_service.diagnose.assert_awaited_once_with(
            ["鳃部发白", "活动减少"],
            "鲤鱼"
        )
    
    @patch("src.routers.rag_router.get_rag")
    def test_rebuild_index_endpoint(self, mock_get_rag):
        """测试重建索引接口"""
        # 模拟RAG服务返回值
        mock_rag_service = MagicMock(spec=RAGService)
        mock_rag_service.rebuild_index.return_value = {
            "status": "success", 
            "message": "Index rebuilt successfully"
        }
        mock_get_rag.return_value = mock_rag_service
        
        # 发送请求
        response = self.client.post("/rag/rebuild")
        
        # 验证结果
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["message"], "Index rebuilt successfully")
        
        # 验证服务调用
        mock_rag_service.rebuild_index.assert_called_once()
    
    @patch("src.routers.rag_router.get_rag")
    def test_error_handling(self, mock_get_rag):
        """测试错误处理"""
        # 模拟服务抛出异常
        mock_rag_service = MagicMock(spec=RAGService)
        mock_rag_service.ask = AsyncMock(side_effect=Exception("测试异常"))
        mock_get_rag.return_value = mock_rag_service
        
        # 发送请求
        request_data = {
            "query": "测试问题",
            "response_mode": "compact"
        }
        response = self.client.post("/rag/ask", json=request_data)
        
        # 验证结果
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("问答处理失败", data["detail"])


if __name__ == "__main__":
    unittest.main() 