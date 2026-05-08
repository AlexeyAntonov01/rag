from app.core.proccesor import RagManager,VectorStore 
from fastapi import FastAPI
from contextlib import asynccontextmanager
import asyncio
from bot.bot_manager import bot,dp
from aiogram.types import BotCommand 

@asynccontextmanager
async def lifespan(app: FastAPI):

    rag = RagManager()
    app.state.rag = rag

    await app.state.rag._init_db()

    main_menu_commands = [
        BotCommand(command="/start", description="Старт"),
        BotCommand(command="/clear", description="Удалить историю")
    ]
    await bot.set_my_commands(main_menu_commands)

    bot_task = asyncio.create_task(dp.start_polling(bot,rag=rag))
    yield
    bot_task.cancel()
    await bot.session.close()
    del app.state.rag