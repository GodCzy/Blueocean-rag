import unittest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.auth import api_key_auth

# 简单认证测试
from src.routers.stats import router as stats_router

class TestAuth(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.dependency_overrides[api_key_auth] = lambda: True
        self.app.include_router(stats_router)
        self.client = TestClient(self.app)

    def test_stats_endpoint(self):
        resp = self.client.get('/stats/diagnosis')
        self.assertEqual(resp.status_code, 200)
        self.assertIn('diagnosis_count', resp.json())

if __name__ == '__main__':
    unittest.main()
