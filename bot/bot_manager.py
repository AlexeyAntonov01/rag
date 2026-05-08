import asyncio
import os
from aiogram import Bot, Dispatcher
from aiogram.types import Message 
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import Command
from app.core.proccesor import RagManager
from aiogram import BaseMiddleware
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram import F

ALLOWED_USERS = [
    int(user_id.strip()) 
    for user_id in os.getenv("ALLOWED_USERS", "").split(",") 
    if user_id.strip()
]

class AccessMiddleware(BaseMiddleware):
    async def __call__(self, handler, event: Message, data):
        if event.from_user.id not in ALLOWED_USERS:
            await event.answer("Доступ запрещен. Обратитесь к администратору.")
            return 
        return await handler(event, data)

PROXY_URL = os.getenv('PROXY')
BOT_TOKEN  = os.getenv('TELEGRAM_TOKEN')

dp = Dispatcher()
dp.message.outer_middleware(AccessMiddleware())
session = AiohttpSession(proxy=PROXY_URL)
bot = Bot(token=BOT_TOKEN,session=session)

def get_main_kb():

    buttons = [
        [
        KeyboardButton(text = 'Удалить историю'),
        KeyboardButton(text = 'Задать вопрос'),
        ]
    ]
    return ReplyKeyboardMarkup(keyboard=buttons, resize_keyboard=True)

@dp.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer("Я - база знаний Axapta. Задай вопрос",reply_markup=get_main_kb())
    

@dp.message(F.text == "Удалить историю")
async def cmd_clear(message: Message,rag: RagManager):
    
    if hasattr(rag,'clearHistory'):
        try:

            rag.clearHistory(message.from_user.id)

            await message.answer('История удалена.')

        except Exception as e:
            await message.answer(f"Ошибка при очистке: {e}")

@dp.message(F.text == "Задать вопрос")
async def btn_ask_info(message: Message):
    await message.answer("Просто введи свой вопрос текстом, и я постараюсь найти ответ в базе знаний.")

@dp.message(F.text)
async def handle_message(message: Message,rag: RagManager):
    try:
        if message.text:

            status_msg = await message.answer("Ollama думает, она польность не влезла в gpu, поэтому надо подождать..")
            await message.bot.send_chat_action(
            chat_id=message.chat.id,
            action="typing"
        )

            answer = await rag.ask(message.text,
                                message.from_user.id)
            await status_msg.edit_text(answer)
    except Exception as e:
        await message.answer(f"Произошла ошибка: {str(e)}")