from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as redis
import os

from api.routes.meal_plan import router as meal_plan_router
from api.routes.health import router as health_router

app = FastAPI(
    title="PSO Meal Scheduler API - Async",
    description="Optimized meal planning using PSO algorithm",
    version="2.0.0"
)

# CORS untuk Next.js Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://dietify-app.vercel.app",
        "http://localhost:3000",  # Local development
        "http://127.0.0.1:3000",
        "https://dietify-app.vercel.app/"
        "https://dietify-app.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis connection
redis_client = None


@app.on_event("startup")
async def startup_event():
    global redis_client
    # Local development fallback
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

    # For local testing without Redis
    try:
        redis_client = redis.from_url(redis_url, decode_responses=True)
        await redis_client.ping()
        print(f"✅ Connected to Redis: {redis_url}")
    except Exception as e:
        print(f"⚠️ Redis not available: {e}")
        print("💡 Start Redis: docker run -d -p 6379:6379 redis:alpine")
        # Optional: Use in-memory fallback for development
        redis_client = None

@app.on_event("shutdown")
async def shutdown_event():
    if redis_client:
        await redis_client.close()

# Include routers
app.include_router(health_router, prefix="/api/v1")
app.include_router(meal_plan_router, prefix="/api/v1")

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "PSO Meal Scheduler API - Ready for Expo!",
        "version": "2.0.0-async",
        "endpoints": {
            "health": "/api/v1/health",
            "meal_plan": "/api/v1/meal-plan",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)