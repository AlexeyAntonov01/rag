from fastapi import FastAPI,Request
from app.api.endponts import router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from app.core.lifespan import lifespan

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

    return templates.TemplateResponse(request=request,name='main.html', context={'request': request})