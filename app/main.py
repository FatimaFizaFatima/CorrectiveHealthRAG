from fastapi import FastAPI
from app.router.corrective_health_router import router as corrective_health_router


app = FastAPI()

app.include_router(corrective_health_router)
