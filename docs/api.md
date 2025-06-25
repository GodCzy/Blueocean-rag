# API 文档

> 仅展示常用接口，更全面的说明请查看项目根目录的 `README.md`。

## 诊断统计
- **URL**: `/api/stats/diagnosis`
- **方法**: `GET`
- **认证**: 需要在请求头 `X-API-Key` 中携带有效的 API key
- **响应示例**:
```json
{
  "diagnosis_count": 12
}
```

## 诊断接口示例请求
```bash
curl -X POST http://localhost:8000/api/diagnosis/diagnose \
     -H "Content-Type: application/json" \
     -H "X-API-Key: <your key>" \
     -d '{"animal_type": "草鱼", "symptoms": ["体表白斑"], "water_parameters": {"temperature": 26, "ph": 7.2}}'
```

