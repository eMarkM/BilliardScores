# BilliardScores (prototype)

Goal: turn a NIL (N. Illinois Pool Players League) paper scoresheet photo into structured data.

## v1 (today)
- Creates a simple CSV from a scoresheet photo (handwriting-friendly) using a **vision model**.
- First pass extracts ONLY player scores:
  - player, rating, game1..game6, total, side (home/visiting)

Next step: add marks (BR/TR/WZ/WF), opponents, and a review/fix UI.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=...  # required
```

## Run (CLI)
```bash
python3 extract_nil_sample_to_csv.py --image ../NilSample.jpeg
```

Outputs:
- `out/<image>.extracted.json` (raw model output)
- `out/<image>.csv` (normalized)

## Run (Telegram bot demo)
1) Create a bot with **@BotFather** and copy the token.
2) Setup env:
```bash
cp .env.example .env
# edit .env and fill TELEGRAM_BOT_TOKEN and OPENAI_API_KEY
# optional: set CAPTAINS=comma,separated,usernames for /status
```
3) Run:
```bash
source .venv/bin/activate
set -a; source .env; set +a
python3 telegram_bot.py
```
4) In Telegram, send the bot a scoresheet photo. It replies with a CSV.

Bot commands:
- `/status` — team upload status since Monday (None/Pending/Confirmed)
- `/confirm` — confirm your pending upload
- `/fixname` — fix a player name by player number
- `/fixscore` — fix a score by player number + game number
- `/help` — show help text + examples

League config:
- `.env`: set `TEAMS_COUNT=14` (or whatever)
