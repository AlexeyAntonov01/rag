from fastapi import FastAPI,Request
from app.api.endpoints import router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from app.core.lifespan import lifespan
from loguru import logger
import os

os.makedirs("data/logs", exist_ok=True)
logger.remove()

logger.add(
    "data/logs/app.log", 
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO",      
    rotation="10 MB", 
    retention="5 days"
)

app = FastAPI(lifespan=lifespan)
app.include_router(router)


templates = Jinja2Templates(directory="app/static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

@app.get("/")
async def main_page(request: Request):

	pass
    #return templates.TemplateResponse(request=request,name='main.html', context={'request': request})