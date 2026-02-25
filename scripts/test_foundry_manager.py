from __future__ import annotations

import json

from bot.ai.client import AzureAiInferenceClient, AiClientError
from bot.ai.prompts import MANAGER_SYSTEM_PROMPT


def main() -> None:
    client = AzureAiInferenceClient()

    print(f"Client enabled: {client.enabled}")
    if not client.enabled:
        print("Check .env values for AZURE_OPENAI_BASE_URL / AZURE_OPENAI_API_KEY / AZURE_OPENAI_DEPLOYMENT")
        return

    tests = [
        "Привіт",
        "Порадь щось гостре з куркою",
        "Розкажи про том ям",
        "Расскажи про рамен",
        "Какая погода сегодня?",
    ]

    for msg in tests:
        print("=" * 80)
        print(f"USER: {msg}")
        try:
            payload = client.complete_json(
                system_prompt=MANAGER_SYSTEM_PROMPT,
                user_prompt=msg,
                temperature=0.15,
                max_tokens=450,
            )
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        except AiClientError as ex:
            print(f"AI CLIENT ERROR: {ex}")
        except Exception as ex:
            print(f"UNEXPECTED ERROR: {ex}")


if __name__ == "__main__":
    main()
