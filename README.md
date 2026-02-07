# TsuPosh Food Bot

Telegram bot for an Asian fast-food place.  
It stores a structured menu in PostgreSQL and recommends dishes based on user preferences (diet, allergens, spiciness, budget).

## Features
- Menu stored in PostgreSQL (categories + items).
- Idempotent seed/import (upsert by `slug`) from `seed/menu_seed.json`.
- aiogram-based Telegram bot (webhook or polling).
- Ready for Azure AI Foundry integration (intent extraction + dish ranking).

## Tech stack
- Python 3.11+
- aiogram
- PostgreSQL
- (optional) Azure AI Foundry for LLM

## Local development (Docker)
1) Copy env:
```bash
cp .env.example .env
