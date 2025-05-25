from fastapi import APIRouter, Depends
from api.services.redis_service import get_redis_client
import os
import httpx
import asyncio

router = APIRouter(tags=["Health Check"])


@router.get("/health")
async def health_check(redis_client=Depends(get_redis_client)):
    """Enhanced health check dengan MLflow monitoring"""
    health_status = {
        "status": "healthy",
        "version": "2.0.0-async",
        "services": {}
    }

    # Test Redis connection
    try:
        await redis_client.ping()
        health_status["services"]["redis"] = "connected"
    except Exception as e:
        health_status["services"]["redis"] = f"disconnected: {str(e)}"
        health_status["status"] = "degraded"

    # Test MLflow connection
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://mlflow:5000")
            if response.status_code == 200:
                health_status["services"]["mlflow"] = "connected"
            else:
                health_status["services"]["mlflow"] = f"unhealthy: HTTP {response.status_code}"
                health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["mlflow"] = f"disconnected: {str(e)}"
        health_status["status"] = "degraded"

    # Check model files
    model_files = [
        "models/meal_data.json",
        "models/tfidf_vectorizer.pkl",
        "models/scaler.pkl"
    ]

    model_status = {}
    missing_models = 0
    for file_path in model_files:
        exists = os.path.exists(file_path)
        model_status[file_path] = "found" if exists else "missing"
        if not exists:
            missing_models += 1

    health_status["models"] = model_status

    if missing_models > 0:
        health_status["status"] = "degraded"
        health_status["warnings"] = f"{missing_models} model files missing"

    # Check volumes
    volume_status = {}
    volume_paths = ["/app/mlruns", "/app/mlartifacts", "/app/models", "/app/output"]

    for path in volume_paths:
        volume_status[path] = "mounted" if os.path.exists(path) else "missing"

    health_status["volumes"] = volume_status

    return health_status


@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed system health check"""
    return {
        "system": {
            "python_version": os.sys.version,
            "working_directory": os.getcwd(),
            "environment_variables": {
                "REDIS_URL": os.getenv("REDIS_URL", "not_set"),
                "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "not_set")
            }
        },
        "filesystem": {
            "disk_usage": _get_disk_usage(),
            "directory_structure": _get_directory_structure()
        }
    }


def _get_disk_usage():
    """Get basic disk usage info"""
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        return {
            "total_gb": round(total / (1024 ** 3), 2),
            "used_gb": round(used / (1024 ** 3), 2),
            "free_gb": round(free / (1024 ** 3), 2)
        }
    except:
        return "unavailable"


def _get_directory_structure():
    """Get basic directory structure"""
    structure = {}
    base_paths = ["/app", "/mlruns", "/mlartifacts"]

    for path in base_paths:
        if os.path.exists(path):
            try:
                structure[path] = {
                    "exists": True,
                    "is_directory": os.path.isdir(path),
                    "contents": len(os.listdir(path)) if os.path.isdir(path) else 0
                }
            except:
                structure[path] = {"exists": True, "accessible": False}
        else:
            structure[path] = {"exists": False}

    return structure