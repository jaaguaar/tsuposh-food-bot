from aiogram.fsm.state import State, StatesGroup


class FoodBotStates(StatesGroup):
    idle = State()

    # Recommendation flow
    clarifying_recommendation = State()
    awaiting_more_preferences = State()
    showing_recommendations = State()

    # Dish details flow
    showing_dish_details = State()