from aiogram import Router
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from bot.fsm.session import (
    get_user_language,
    set_current_intent,
    set_user_language,
)
from bot.fsm.states import FoodBotStates
from bot.services.language import detect_language

router = Router()


def _is_smalltalk(text: str) -> bool:
    normalized = text.strip().lower()
    keywords = {
        "ua": ["привіт", "вітаю", "добрий день", "дякую", "дякс", "як справи"],
        "ru": ["привет", "здравствуйте", "добрый день", "спасибо", "как дела"],
    }
    return any(k in normalized for k in keywords["ua"] + keywords["ru"])


def _looks_like_recommendation_request(text: str) -> bool:
    normalized = text.strip().lower()
    triggers = [
        "порадь",
        "порадити",
        "що поїсти",
        "хочу щось",
        "допоможи обрати",
        "посоветуй",
        "что поесть",
        "хочу что-то",
        "помоги выбрать",
    ]
    return any(t in normalized for t in triggers)


def _looks_like_dish_question(text: str) -> bool:
    normalized = text.strip().lower()
    triggers = [
        "що таке",
        "розкажи про",
        "що за страва",
        "what is",
        "что такое",
        "расскажи про",
        "что за блюдо",
    ]
    return any(t in normalized for t in triggers)


@router.message()
async def handle_text_message(message: Message, state: FSMContext) -> None:
    if not message.text:
        return

    text = message.text.strip()
    detected_language = detect_language(text)
    await set_user_language(state, detected_language)

    current_language = await get_user_language(state)

    if _is_smalltalk(text):
        await set_current_intent(state, "smalltalk")
        await state.set_state(FoodBotStates.idle)

        if current_language == "ru":
            await message.answer(
                "Привет! 👋 Я помогу с выбором блюд азиатской кухни.\n"
                "Можешь написать, например:\n"
                "• Хочу что-то острое\n"
                "• Посоветуй суп\n"
                "• Расскажи про том ям"
            )
        else:
            await message.answer(
                "Привіт! 👋 Я допоможу з вибором страв азіатської кухні.\n"
                "Можеш написати, наприклад:\n"
                "• Хочу щось гостре\n"
                "• Порадь суп\n"
                "• Розкажи про том ям"
            )
        return

    if _looks_like_recommendation_request(text):
        await set_current_intent(state, "recommendation_request")
        await state.set_state(FoodBotStates.clarifying_recommendation)

        if current_language == "ru":
            await message.answer(
                "С удовольствием помогу 🙂\n"
                "Чтобы лучше подобрать блюдо, скажи, пожалуйста:\n"
                "Что тебе больше хочется — суп, лапша, рис, роллы или закуска?"
            )
        else:
            await message.answer(
                "Із задоволенням допоможу 🙂\n"
                "Щоб краще підібрати страву, підкажи, будь ласка:\n"
                "Що тобі більше хочеться — суп, локшина, рис, роли чи закуска?"
            )
        return

    if _looks_like_dish_question(text):
        await set_current_intent(state, "dish_question")
        await state.set_state(FoodBotStates.showing_dish_details)

        if current_language == "ru":
            await message.answer(
                "Отличный вопрос 👨‍🍳\n"
                "На следующем этапе я подключу «шефа» и смогу красиво рассказать о блюде, "
                "его вкусе и с чем оно лучше сочетается."
            )
        else:
            await message.answer(
                "Чудове запитання 👨‍🍳\n"
                "На наступному етапі я підключу «шефа» і зможу красиво розповісти про страву, "
                "її смак і з чим вона найкраще поєднується."
            )
        return

    await set_current_intent(state, "clarify_needed")
    await state.set_state(FoodBotStates.idle)

    if current_language == "ru":
        await message.answer(
            "Понял 👍\n"
            "Я пока в режиме MVP и лучше всего помогаю с выбором еды.\n"
            "Можешь написать, например:\n"
            "• Посоветуй что-нибудь не острое\n"
            "• Хочу роллы\n"
            "• Расскажи про пад тай"
        )
    else:
        await message.answer(
            "Зрозумів 👍\n"
            "Я поки в режимі MVP і найкраще допомагаю з вибором їжі.\n"
            "Можеш написати, наприклад:\n"
            "• Порадь щось не гостре\n"
            "• Хочу роли\n"
            "• Розкажи про пад тай"
        )