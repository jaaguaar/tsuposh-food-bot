from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from bot.fsm.session import set_user_language
from bot.fsm.states import FoodBotStates
from bot.services.language import detect_language

router = Router()


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext) -> None:
    user_text = message.text or ""
    language = detect_language(user_text)

    await set_user_language(state, language)
    await state.set_state(FoodBotStates.idle)

    if language == "ru":
        text = (
            "Привет! 👋\n"
            "Я бот-помощник по выбору блюд азиатской кухни.\n\n"
            "Могу:\n"
            "• помочь выбрать блюдо\n"
            "• рассказать о конкретном блюде\n"
            "• предложить похожие варианты\n\n"
            "Напиши, что тебе хочется 🙂"
        )
    else:
        text = (
            "Привіт! 👋\n"
            "Я бот-помічник з вибору страв азіатської кухні.\n\n"
            "Можу:\n"
            "• допомогти обрати страву\n"
            "• розповісти про конкретну страву\n"
            "• запропонувати схожі варіанти\n\n"
            "Напиши, що тобі хочеться 🙂"
        )

    await message.answer(text)