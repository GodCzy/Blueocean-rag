from fastapi import APIRouter
from src.routers.chat_router import chat
from src.routers.data_router import data
from src.routers.base_router import base
from src.routers.tool_router import tool
from src.routers.rag_router import rag
from src.routers.diagnosis import router as diagnosis
from src.routers.ocean_env import router as ocean_env
from src.routers.admin import router as admin

router = APIRouter()
router.include_router(base)
router.include_router(chat)
router.include_router(data)
router.include_router(tool)
router.include_router(rag)
router.include_router(diagnosis)
router.include_router(ocean_env)
router.include_router(admin)
