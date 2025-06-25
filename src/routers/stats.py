from fastapi import APIRouter, Depends

from src.auth import api_key_auth

# 统计相关路由

router = APIRouter(prefix="/stats", tags=["统计"], dependencies=[Depends(api_key_auth)])

_diagnosis_count = 0


def increment_diagnosis():
    global _diagnosis_count
    _diagnosis_count += 1


@router.get("/diagnosis")
async def get_diagnosis_count():
    """Return number of diagnosis requests since startup."""
    return {"diagnosis_count": _diagnosis_count}

