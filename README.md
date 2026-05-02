# Vera Merchant Bot - Magicpin Challenge

This project implements a merchant engagement bot for the magicpin challenge.

## Overview

This bot uses FastAPI to expose the 5 required endpoints and a deterministic local composer to produce grounded Vera messages without needing an external API key.

The browser UI at `/` is a manual console for local checking. It can load the expanded demo dataset, run `/v1/tick`, open generated conversations, send merchant replies to `/v1/reply`, and show the raw JSON responses.

## Approach

1. In-memory datastore: the bot keeps categories, merchants, customers, triggers, conversations, and suppression keys in memory.
2. Context passing: when `/v1/tick` is called, the bot combines category, merchant, trigger, and optional customer context.
3. Structured outputs: every composer path returns `body`, `cta`, `send_as`, `suppression_key`, and `rationale`.
4. Auto-reply detection: common canned replies and repeated messages return `wait` or `end`.
5. Intent transition: replies like "yes", "send it", or "go ahead" switch directly to action mode.

## Tradeoffs

- The bot ranks available triggers by urgency and merchant/customer state, then returns at most 20 actions per tick.
- Conversation history is in memory. A production system would use Redis or a database to survive service restarts.
- The UI is for human checking only. The official judge still calls the HTTP API endpoints.

## Required Endpoints

- `GET /v1/healthz`
- `GET /v1/metadata`
- `POST /v1/context`
- `POST /v1/tick`
- `POST /v1/reply`

## How to Run Locally

1. Set up virtual environment and install requirements:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Start the bot:

   ```powershell
   venv\Scripts\python.exe -m uvicorn bot:app --host 127.0.0.1 --port 8081
   ```

3. Open the local UI:

   ```text
   http://127.0.0.1:8081
   ```

4. In the UI, click `Load Demo Dataset`, then `Run Tick`, then select a conversation and send replies.

5. For the official simulator, set `BOT_URL` in `judge_simulator.py`:

   ```python
   BOT_URL = "http://127.0.0.1:8081"
   ```

   Then run:

   ```powershell
   python judge_simulator.py
   ```

## Deploy

Use the included `render.yaml` or `Procfile`:

```text
uvicorn bot:app --host 0.0.0.0 --port $PORT
```

Set these environment variables on the host before final submission:

```text
TEAM_NAME=Your team name
TEAM_MEMBERS=Your Name
CONTACT_EMAIL=your@email.com
```

Submit the deployed base URL, not a localhost URL.
