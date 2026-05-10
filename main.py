
import os
import json
import re
import sqlite3
import hashlib
import threading
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from flask import Flask, render_template, request, jsonify, session, redirect, url_for




app = Flask(__name__)


def load_dotenv(path=".env"):
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()

app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")

# =========================
# GROQ CONFIG
# =========================

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", GROQ_MODEL).strip()
APP_TIMEZONE = os.getenv("APP_TIMEZONE", "America/Vancouver")
CHAT_HISTORY_LIMIT = int(os.getenv("CHAT_HISTORY_LIMIT", "8"))
MEMORY_FACT_LIMIT = int(os.getenv("MEMORY_FACT_LIMIT", "12"))
MEMORY_UPDATE_INTERVAL = int(os.getenv("MEMORY_UPDATE_INTERVAL", "4"))
MEMORY_MIN_USER_CHARS = int(os.getenv("MEMORY_MIN_USER_CHARS", "24"))
GOALS_CONTEXT_MAX_CHARS = int(os.getenv("GOALS_CONTEXT_MAX_CHARS", "2500"))
COUNCIL_CACHE_TTL_SECONDS = int(os.getenv("COUNCIL_CACHE_TTL_SECONDS", "90"))
COUNCIL_CONTEXT_MAX_CHARS = int(os.getenv("COUNCIL_CONTEXT_MAX_CHARS", "320"))

# =========================
# DATABASE
# =========================

DB = "memory.db"


def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            memory TEXT,
            created_at TEXT
        )
        """
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            role TEXT,
            content TEXT,
            created_at TEXT
        )
        """
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS long_term_memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            memory TEXT,
            created_at TEXT
        )
        """
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS context_stories (
            user_id TEXT PRIMARY KEY,
            story TEXT,
            updated_at TEXT
        )
        """
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT
        )
        """
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS founder_profiles (
            user_id TEXT PRIMARY KEY,
            tools TEXT,
            updated_at TEXT
        )
        """
    )

    conn.commit()
    conn.close()


init_db()
COUNCIL_CACHE = {}


# =========================
# MEMORY SYSTEM
# =========================


def save_long_term_memory(user_id, memory):
    if not memory:
        return

    conn = sqlite3.connect(DB)
    c = conn.cursor()

    c.execute(
        "INSERT INTO long_term_memories (user_id, memory, created_at) VALUES (?, ?, ?)",
        (user_id, memory, datetime.now().isoformat())
    )

    conn.commit()
    conn.close()


def normalize_memory_text(memory):
    return re.sub(r"\s+", " ", memory.strip().lower())


def replace_long_term_memories(user_id, memories, limit=30):
    cleaned = []
    seen = set()

    for memory in memories:
        if not isinstance(memory, str):
            continue

        text = re.sub(r"\s+", " ", memory).strip()
        key = normalize_memory_text(text)

        if not text or key in seen:
            continue

        cleaned.append(text)
        seen.add(key)

        if len(cleaned) >= limit:
            break

    conn = sqlite3.connect(DB)
    c = conn.cursor()

    c.execute(
        "SELECT memory, created_at FROM long_term_memories WHERE user_id = ?",
        (user_id,)
    )
    existing_dates = {
        normalize_memory_text(memory): created_at
        for memory, created_at in c.fetchall()
    }

    c.execute("DELETE FROM long_term_memories WHERE user_id = ?", (user_id,))

    now = datetime.now().isoformat()
    for memory in reversed(cleaned):
        created_at = existing_dates.get(normalize_memory_text(memory), now)
        c.execute(
            "INSERT INTO long_term_memories (user_id, memory, created_at) VALUES (?, ?, ?)",
            (user_id, memory, created_at)
        )

    conn.commit()
    conn.close()


def clear_user_memory(user_id):
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    c.execute("DELETE FROM long_term_memories WHERE user_id = ?", (user_id,))
    c.execute("DELETE FROM context_stories WHERE user_id = ?", (user_id,))
    c.execute("DELETE FROM chat_messages WHERE user_id = ?", (user_id,))

    conn.commit()
    conn.close()


def wants_memory_clear(message):
    lower = message.lower()
    return (
        "clear" in lower
        and any(word in lower for word in ["memory", "memories", "remember", "know"])
    ) or "forget everything" in lower or "start fresh" in lower


def requested_app_reads(message):
    lower = message.lower()
    tools = []

    if any(phrase in lower for phrase in [
        "what do you know",
        "what all do you know",
        "what al do u know",
        "what do u know",
        "read memory",
        "show memory",
        "memory"
    ]):
        tools.append("Memory")

    if any(word in lower for word in ["goal", "goals", "tasks", "plan", "priority"]):
        tools.append("Goals")

    return tools



def save_chat_message(user_id, role, content):
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    c.execute(
        "INSERT INTO chat_messages (user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (user_id, role, content, datetime.now().isoformat())
    )

    conn.commit()
    conn.close()


def get_short_term_messages(user_id, limit=15):
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    c.execute(
        """
        SELECT role, content FROM chat_messages
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (user_id, limit)
    )

    rows = c.fetchall()
    conn.close()

    return [{"role": row[0], "content": row[1]} for row in reversed(rows)]


def get_long_term_memories(user_id, limit=30):
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    c.execute(
        """
        SELECT memory FROM long_term_memories
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (user_id, limit)
    )

    rows = c.fetchall()
    conn.close()

    return [row[0] for row in rows]


def get_context_story(user_id):
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    c.execute("SELECT story FROM context_stories WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    conn.close()

    return row[0] if row else ""


def save_context_story(user_id, story):
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    c.execute(
        """
        INSERT INTO context_stories (user_id, story, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            story = excluded.story,
            updated_at = excluded.updated_at
        """,
        (user_id, story, datetime.now().isoformat())
    )

    conn.commit()
    conn.close()


def get_current_time_context():
    now = datetime.now(ZoneInfo(APP_TIMEZONE))

    return (
        f"Current date and time: {now.strftime('%A, %B %d, %Y at %I:%M %p %Z')}.\n"
        f"Timezone: {APP_TIMEZONE}.\n"
        f"ISO timestamp: {now.isoformat()}."
    )


# =========================
# SYSTEM PROMPT
# =========================

SYSTEM_PROMPT = """
You are the user's AI cofounder.

Your personality:
- smart
- calm, direct, and easy to talk to
- plainspoken, not performative
- practical and action-focused
- encouraging without hype
- honest when an idea is hard, but still helpful
- opinionated enough to make decisions with incomplete info

You help founders:
- stay focused
- avoid distractions
- find startup ideas
- validate products
- find cofounders
- stay accountable
- plan launches
- think bigger
- choose a practical tech stack
- decide whether they need a cofounder, contractor, designer, or can solo-build
- figure out what website, landing page, demo, prototype, or MVP they need next
- turn vague ambition into a concrete project plan

IMPORTANT:
- Talk like a capable cofounder sitting next to the user, not like a corporate consultant.
- Default texting style for this user right now: short, chill, casual, and straight to the point. Sound like a smart cofounder texting, not a polished assistant.
- Voice target: a locked-in teenage cofounder who is smart, blunt, and useful. Not cringe, not corporate, not teacher-ish.
- Default rhythm: quick take, one move, done.
- Match the user's casual vibe without copying every typo. Shortcuts like "u", "rn", "tbh", "imo", "kinda", "gonna", "lmk", "bet", and "nah" are okay when natural.
- Use smaller/simple words when possible: say "use", not "utilize"; "make", not "construct"; "ship", not "execute"; "show", not "demonstrate".
- Be sharp, not overly nice. If an idea is vague, say so quickly and give the next move.
- Cut filler like "How's it going", "Want to", "Consider", and "I recommend" unless it genuinely helps.
- Use at most one emoji in a message, even if the user asks for emojis, unless they explicitly ask for a lot of emojis.
- Never answer with a menu after a casual message. No "Need help with..." lists.
- Keep responses concise by default: 1-2 short paragraphs or at most 3 bullets. Most normal replies should be under 70 words.
- Do not use headings like "Next Steps", "Breakdown", "Updated Goal", "Goal Update", or "Your turn" in normal chat.
- Avoid long lists unless the user explicitly asks for a deep breakdown.
- Do not greet casual messages with a menu of options.
- Do not mirror slang awkwardly. Avoid lines like "let's vibe out", "Nodex is fire", or "what's your favorite AI use case" unless the user actually asks for casual brainstorming.
- Do not write generic disclaimers unless there is real risk.
- Start with a direct, human reaction to the user's idea, or skip the reaction when the user asks for action.
- Give exactly one concrete next move by default.
- If you use bullets, make each bullet short. No paragraph-sized bullets.
- Ask at most one strong question at the end, and only when the answer is needed to move forward.
- If the user says "you decide", "u decide", "idk", "whatever", or asks you to choose, make the decision yourself and explain it briefly.
- Do not respond to "you decide" with more open-ended options.
- Prefer making one small execution step over listing possibilities.
- Maintain conversational momentum: if you need clarity, ask one tight question and mentally queue the next step. After the user answers, continue from that queued thought instead of restarting.
- Do not say "I'll update the goals page" or "Goal Update" unless you are actually including a valid app command and the user clearly asked for the app to change.
- Never output raw JSON to the user. App commands must be hidden inside <app_command> tags only, and only when the user clearly asks to update the app.
- Do not write "App Update" in the visible reply. The UI handles update labels.
- If the user asks what an app update means, say simply: "It means I changed something in the workspace, usually Goals. The gray row is just the receipt."
- If the user asks for a huge outcome, translate it into a serious operating target without lecturing. Example: "$1M is fine as a north star; the useful version is $1M ARR by a date, with the next milestone being the first 5 paying teams."
- When an idea is broad, choose a wedge: target customer, first product, MVP, landing page, stack, and validation step.
- If the user seems non-technical or early, suggest the simplest stack and path first.
- If a website would help, say exactly what kind: landing page, waitlist, demo page, pitch page, or marketplace.
- If another person would help, say who: technical cofounder, designer, salesperson, mechanical engineer, domain expert, or first customer.
- Keep momentum. Every answer should either decide, build, validate, recruit, or clarify one important thing.
- Format responses for readability with short paragraphs.
- Use bullets only when they make the answer easier to act on; never make a giant checklist by default.
- Bold the most important words or labels with **double asterisks**, but do not overdo it.
- Remember the user's goals.
- Never sound robotic.
- Push the user toward action without being annoying.

Style examples:
- If the user says "wsp": "Yo. For Nodex, I’d use this session to lock the first demo workflow, not brainstorm more features."
- If the user says "idk u decide": "Bet. Start with one workflow: support teams building AI automations. Today, sketch the 3-node demo: input message, classify intent, draft reply."
- If the user says "what should I do next": "Do this: define the first demo workflow in 3 nodes. That’ll make Nodex way easier to explain."
- If the user says "launch faster": "Bet. Cut scope: build one clickable demo and one waitlist page. Don’t touch the big platform yet."
- If the user asks "what app can I use": "Use Carrd for the landing page. Fastest path: headline, 3 bullets, waitlist form. Don’t overbuild it."

Better style examples to follow over the older ones:
- If the user says "wsp": "Yo. For Nodex, I'd use this session to lock the first demo workflow, not brainstorm more features."
- If the user says "idk u decide": "Bet. Start with one workflow: support teams building AI automations. Today, sketch the 3-node demo: input msg, classify intent, draft reply."
- If the user says "what should I do next": "Do this rn: define the first demo workflow in 3 nodes. That makes Nodex way easier to explain."
- If the user says "launch faster": "Bet. Cut scope: one clickable demo + one waitlist page. Don't touch the big platform yet."
- If the user asks "what app can I use": "Use Carrd. Fastest path: headline, 3 bullets, waitlist form. Don't overbuild it."
- If the user asks for a deep plan: "Yeah, here's the clean plan:" then use short bullets.

When the user has a huge ambition, do not shut it down.
Help them shrink it into a first real step, prototype, wedge, or experiment.

Decision framework:
- If the user wants to start a company, pick a narrow first wedge.
- If the user has no product yet, define the smallest MVP.
- If the user has no audience yet, suggest a landing page and 10 target people to interview.
- If the product needs hard engineering, separate the software prototype from the physical product.
- If the user asks what to do next, give one decisive next action, plus one sentence explaining why.

Goal-setting framework:
- The overall goal should be a realistic 1-year company outcome.
- Write the overall goal in a proper measurable format: "By [date/timeframe], we will have [specific outcome] by [how/market/customer/product]."
- The goal can be revenue, users, customers, launches, waitlist, retention, market validation, product readiness, fundraising, or operations, depending on the company.
- The overall goal should cover the company as a whole, not just one task like "make a logo".
- The monthly goal should be the next meaningful milestone toward the 1-year goal.
- The weekly goal should be the next concrete push toward the monthly goal.
- The one-week work chart should assign each person a task that directly supports the weekly goal.
- Avoid vague goals like "grow the business" or "build brand identity". Make them measurable, grounded, and useful.
- If details are missing, make reasonable assumptions and choose a realistic first version.

Example style:
"I’d pick this: a waitlist page for a low-cost electric microcar concept, not a full car company yet. We validate demand first. Build a landing page, collect 50 emails, then talk to 10 people who commute under 10km."

App skills:
- In normal chat, you may read the Goals page but you must not change it.
- Never change Goals from normal chat, even if the user asks. Tell them to use the Goals page/generate button.
- You may only control Goals when the request is coming from a dedicated goal-generation flow.
- When goal writing is allowed, add exactly one hidden app command at the end of your reply.
- Never show, mention, or explain app-command JSON to the user.
- Keep the visible reply short and human, then append the app command only when goal writing is allowed.
- For anything in the app, think in four operations: add, edit, read, delete.
- If the user asks to read/show goals, tasks, priorities, or the company plan, answer from the Current Goal Board State.
- If the user asks what you know or asks about memory, answer only from Long-Term Memory and Contextual Story So Far. If those are empty, say you do not know anything specific yet.
- If the user asks to clear memory, the server will do it. Keep the visible response simple.
- Be action-capable: summarize memory/goals or produce a useful next move instead of generic advice.

Available app commands:
1. Update the Goals page:
<app_command>
{"action":"goal_update","payload":{"overallGoal":"By [timeframe], we will have [measurable company outcome] by [strategy/customer/product].","weeklyGoal":"One measurable whole-company goal for this week that moves toward the monthly goal.","monthlyGoal":"One measurable whole-company milestone for this month that moves toward the 1-year goal.","people":[{"name":"Person or role","role":"Role","task":"This week's task","detail":"Specific detail that supports the weekly goal","status":"Today"}]}}
</app_command>

2. Clear the Goals page:
<app_command>
{"action":"goal_clear"}
</app_command>

Use app commands when:
- The user asks to change goals, set goals, make team goals, assign tasks, clear goals, change the company direction, or work on goals.
- The user expects the app UI to change, not just a text answer.
- If the user changes company direction, replace the goal board with the new company.
- Goals page model: overall goal first, then weekly and monthly company-wide goal bubbles, then a one-week work chart by person.
- When updating goals, make the board feel like an operating plan: 1-year outcome, monthly milestone, weekly goal, person-by-person work.
- For now the real team is only the user and the AI assistant. Extra collaborators are placeholders/invites with role or skill set until accounts exist.

App navigation:
- The app has two views the user can navigate to: Goals and Dashboard (the chat).
- When it would be genuinely helpful to send the user to a view, embed a navigation chip using this exact syntax: [→ Goals] or [→ Dashboard]
- Only use navigation chips when it adds real value (e.g. "Check your [→ Goals] to see what's next.").
- Do not overuse them. One per reply at most.
- Never explain what the chip does, just include it naturally in the sentence.
"""


# =========================
# GROQ CHAT
# =========================


def call_groq(messages, temperature=0.55, max_tokens=1000):
    if not GROQ_API_KEY:
        raise RuntimeError("Missing GROQ_API_KEY in environment.")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
    if not response.ok:
        detail = ""
        try:
            body = response.json()
            detail = body.get("error", {}).get("message") or body.get("message") or str(body)
        except Exception:
            detail = response.text[:300]
        raise RuntimeError(f"Groq API error {response.status_code}: {detail}")
    data = response.json()

    return data["choices"][0]["message"]["content"]


def call_llm(messages, temperature=0.55, max_tokens=1000):
    provider = LLM_PROVIDER
    if provider == "groq":
        return call_groq(messages, temperature=temperature, max_tokens=max_tokens)
    raise RuntimeError(f"Unsupported LLM_PROVIDER '{provider}'. Set LLM_PROVIDER=groq or add a provider adapter.")


def parse_json_response(text):
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1:
        cleaned = cleaned[start:end + 1]

    return json.loads(cleaned)


def safe_council_payload():
    return {
        "ok": True,
        "provider": LLM_PROVIDER,
        "model": LLM_MODEL,
        "members": [
            {"id": "contrarian", "name": "The Contrarian", "focus": "Skepticism", "stance": "Higher risk than expected. Validate demand first.", "score": 48},
            {"id": "first_principles", "name": "First Principles", "focus": "Fundamentals", "stance": "Define core pain and buyer before building.", "score": 55},
            {"id": "executor", "name": "The Executor", "focus": "Execution", "stance": "Ship tiny MVP and get 10 real user calls.", "score": 62},
            {"id": "customer_advocate", "name": "Customer Advocate", "focus": "Customer", "stance": "Message is unclear. Tighten value proposition.", "score": 58},
            {"id": "market_strategist", "name": "Market Strategist", "focus": "Market", "stance": "Niche down first, then expand distribution.", "score": 57},
        ],
        "debate": [
            {"speaker": "The Contrarian", "target": "The Executor", "tone": "challenge", "message": "Too much build risk before proof."},
            {"speaker": "The Executor", "target": "All", "tone": "pushback", "message": "Small MVP gets proof fastest."},
            {"speaker": "Customer Advocate", "target": "All", "tone": "support", "message": "Need stronger pain framing first."},
            {"speaker": "Market Strategist", "target": "All", "tone": "support", "message": "Start with one tight customer segment."}
        ],
        "metrics": {"viability": 56, "executionEase": 61, "marketPull": 52, "riskLevel": 63},
        "startup": {
            "neededToStartUSD": 5000,
            "runwayMonths": 4,
            "teamNeeded": ["Founder", "Builder"],
            "assetsNeeded": ["Landing page", "CRM", "Analytics"],
            "profitMarginPct": 28
        },
        "kpis": [
            {"name": "CAC Payback", "valuePct": 35},
            {"name": "Lead->Paid", "valuePct": 12},
            {"name": "30d Retention", "valuePct": 48}
        ],
        "financialDiagram": {
            "months": ["M1", "M2", "M3", "M4", "M5", "M6"],
            "revenue": [1000, 2000, 3500, 5000, 7000, 9500],
            "costs": [3000, 3200, 3600, 4200, 4800, 5500],
            "profit": [-2000, -1200, -100, 800, 2200, 4000]
        },
        "nextSteps": ["Niche customer", "Run 10 interviews", "Ship MVP fast"],
        "chatVerdict": "This is promising, but not fully de-risked yet. Tight niche plus fast validation gives it a real shot."
    }


def strip_goal_update(text):
    without_fences = re.sub(
        r"```(?:json|goal_update|app_command)?\s*\{[\s\S]*?\"action\"\s*:\s*\"goal_(?:update|clear)\"[\s\S]*?\}\s*```",
        "",
        text,
        flags=re.IGNORECASE
    )
    without_tags = re.sub(
        r"<app_command>\s*[\s\S]*?\s*</app_command>",
        "",
        without_fences,
        flags=re.IGNORECASE
    )
    return re.sub(
        r"\{[\s\S]*?\"action\"\s*:\s*\"goal_(?:update|clear)\"[\s\S]*\}\s*$",
        "",
        without_tags,
        flags=re.IGNORECASE
    ).strip()


def has_app_command(text):
    return bool(
        re.search(r"<app_command>\s*[\s\S]*?\s*</app_command>", text, flags=re.IGNORECASE)
        or re.search(r"```(?:goal_update|app_command)\s*[\s\S]*?```", text, flags=re.IGNORECASE)
        or re.search(r"\{[\s\S]*?\"action\"\s*:\s*\"goal_(?:update|clear)\"[\s\S]*\}", text, flags=re.IGNORECASE)
    )


def user_allows_app_update(message):
    lower = message.lower()
    update_words = [
        "update", "set", "change", "edit", "assign", "save", "clear", "replace",
        "chage", "chaneg", "chnage"
    ]
    app_words = ["app", "goal", "goals", "task", "tasks", "page", "board"]

    return (
        "update the app" in lower
        or (
            any(word in lower for word in update_words)
            and any(word in lower for word in app_words)
        )
    )


def clean_visible_response(text):
    cleaned = strip_goal_update(text)
    cleaned = re.sub(r"(?:<\|?\s*)?(?:start|end)_header_id(?:\s*\|?>)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bend_header_id\|end_header_id\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<\s*/?\s*(?:app_command)?\s*>?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*</\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    cleaned = re.sub(
        r"(?:^|\n)\s*(?:App Update|Goal Update|Your turn!?|I'll update the app\.?|I'll update the goals page accordingly\.?|I'll update the goals page to reflect your progress\.?)\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE
    ).strip()
    return cleaned


def build_server_app_command(user_message, response, goals_state):
    lower = user_message.lower()

    if "clear" in lower and "goal" in lower:
        return '\n\n<app_command>\n{"action":"goal_clear"}\n</app_command>'

    return ""


def build_memory_context(user_id, goals_state=None):
    long_term_memories = get_long_term_memories(user_id, limit=MEMORY_FACT_LIMIT)
    context_story = get_context_story(user_id)
    time_context = get_current_time_context()

    long_term_context = "\n".join(f"- {memory}" for memory in long_term_memories)
    if goals_state:
        goals_context = json.dumps(goals_state, separators=(",", ":"))
        if len(goals_context) > GOALS_CONTEXT_MAX_CHARS:
            goals_context = goals_context[:GOALS_CONTEXT_MAX_CHARS] + "...[truncated]"
    else:
        goals_context = "No goal board state provided."

    return (
        f"Time Context:\n{time_context}\n\n"
        f"Long-Term Memory:\n{long_term_context or 'No long-term memories saved yet.'}\n\n"
        f"Contextual Story So Far:\n{context_story or 'No story has been written yet.'}\n\n"
        f"Current Goal Board State:\n{goals_context}"
    )


def build_council_context(user_id, goals_state=None):
    facts = get_long_term_memories(user_id, limit=4)
    compact = "; ".join(facts).strip()
    if not compact:
        compact = "No saved founder context."
    if len(compact) > COUNCIL_CONTEXT_MAX_CHARS:
        compact = compact[:COUNCIL_CONTEXT_MAX_CHARS] + "...[trimmed]"
    return compact


def ask_ai(user_message, user_id, goals_state=None):
    short_term_messages = get_short_term_messages(user_id, limit=CHAT_HISTORY_LIMIT)
    memory_context = build_memory_context(user_id, goals_state)

    messages = [
        {
            "role": "system",
            "content": (
                SYSTEM_PROMPT
                + f"\n\n{memory_context}"
            )
        },
        *short_term_messages,
        {"role": "user", "content": user_message}
    ]

    try:
        return call_llm(messages, max_tokens=550)
    except Exception as e:
        return f"AI request failed: {str(e)}"


def ask_goal_writer(user_message, user_id, goals_state=None):
    short_term_messages = get_short_term_messages(user_id, limit=CHAT_HISTORY_LIMIT)
    memory_context = build_memory_context(user_id, goals_state)

    messages = [
        {
            "role": "system",
            "content": (
                SYSTEM_PROMPT
                + "\n\nGOAL_WRITE_ALLOWED: You are being called by a dedicated Goals button. "
                "You may update the Goals page with exactly one hidden <app_command>."
                + f"\n\n{memory_context}"
            )
        },
        *short_term_messages,
        {"role": "user", "content": user_message}
    ]

    try:
        return call_llm(messages)
    except Exception as e:
        return f"AI request failed: {str(e)}"


def update_memory(user_id, user_message, ai_response):
    short_term_messages = get_short_term_messages(user_id, limit=CHAT_HISTORY_LIMIT)
    long_term_memories = get_long_term_memories(user_id, limit=50)
    context_story = get_context_story(user_id)

    transcript = "\n".join(
        f"{message['role']}: {message['content']}" for message in short_term_messages
    )
    long_term_context = "\n".join(f"- {memory}" for memory in long_term_memories)

    try:
        memory_response = call_llm(
            [
                {
                    "role": "system",
                    "content": """
You manage memory for an AI cofounder app.

Return ONLY valid JSON with this shape:
{
  "facts": ["current useful long-term fact", "..."],
  "story": "A first-person 400-word-or-less contextual story about everything the AI cofounder has done with this user from the start until now."
}

Rules:
- The "facts" array is the full cleaned current memory list, not just new facts.
- Keep at most 30 facts.
- Preserve existing facts that still look true and useful.
- Keep facts short, specific, and useful for helping the founder.
- Include stable facts: goals, projects, preferences, decisions, constraints, identity, deadlines, current company direction, important collaborators, and strong dislikes.
- Merge duplicates into one better fact.
- Replace outdated facts with the newest version when the recent chat changes something.
- Delete facts that are contradicted, no longer useful, vague, or only random small talk.
- Do not keep multiple versions of the same project, goal, preference, or deadline.
- Do not remember random small talk.
- The story must be under 400 words.
- The story should be written from the AI cofounder's point of view.
- The story should summarize the relationship, the work, decisions, and current direction.
"""
                },
                {
                    "role": "user",
                    "content": f"""
Existing long-term memories:
{long_term_context or "None"}

Previous story:
{context_story or "None"}

Recent chat:
{transcript}

Latest user message:
{user_message}

Latest AI response:
{ai_response}
"""
                }
            ],
            temperature=0.2,
            max_tokens=700
        )

        parsed = parse_json_response(memory_response)

        facts = parsed.get("facts")
        if facts is None:
            facts = long_term_memories + parsed.get("remember", [])

        replace_long_term_memories(user_id, facts, limit=30)

        story = parsed.get("story", "").strip()
        if story:
            save_context_story(user_id, story)
    except Exception:
        return


def should_run_memory_update(user_id, user_message):
    text = (user_message or "").strip()
    if len(text) < MEMORY_MIN_USER_CHARS:
        return False

    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute(
        "SELECT COUNT(*) FROM chat_messages WHERE user_id = ? AND role = 'user'",
        (user_id,)
    )
    count = c.fetchone()[0] or 0
    conn.close()

    if count <= 1:
        return True
    return count % MEMORY_UPDATE_INTERVAL == 0



# =========================
# ROUTES
# =========================


@app.route("/")
def home():
    return render_template("index.html")


# =========================
# AUTH ROUTES
# =========================

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


@app.route("/auth/signup", methods=["POST"])
def auth_signup():
    data = request.json or {}
    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"ok": False, "error": "Email and password are required."}), 400
    if len(password) < 8:
        return jsonify({"ok": False, "error": "Password must be at least 8 characters."}), 400

    conn = sqlite3.connect(DB)
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (name, email, _hash_password(password), datetime.now().isoformat())
        )
        conn.commit()
        user_id = str(c.lastrowid)
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"ok": False, "error": "An account with that email already exists."}), 409
    conn.close()

    session["user_id"] = user_id
    session["user_email"] = email
    return jsonify({"ok": True})


@app.route("/auth/signin", methods=["POST"])
def auth_signin():
    data = request.json or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"ok": False, "error": "Email and password are required."}), 400

    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT id, password_hash FROM users WHERE email = ?", (email,))
    row = c.fetchone()
    conn.close()

    if not row or row[1] != _hash_password(password):
        return jsonify({"ok": False, "error": "Invalid email or password."}), 401

    session["user_id"] = str(row[0])
    session["user_email"] = email
    return jsonify({"ok": True})


@app.route("/auth/signout")
def auth_signout():
    session.clear()
    return redirect(url_for("home"))


@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("home"))

    return render_template("dashboard.html")


@app.route("/api/founder/profile", methods=["POST"])
def api_founder_profile():
    if "user_id" not in session:
        return jsonify({"ok": False, "error": "Not signed in."}), 401

    data = request.json or {}
    tools = data.get("tools", [])
    if not isinstance(tools, list):
        tools = []

    cleaned_tools = []
    seen = set()
    for tool in tools:
        if not isinstance(tool, str):
            continue
        text = re.sub(r"\s+", " ", tool).strip()
        key = text.lower()
        if text and key not in seen:
            cleaned_tools.append(text)
            seen.add(key)

    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO founder_profiles (user_id, tools, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET tools = excluded.tools, updated_at = excluded.updated_at
        """,
        (session["user_id"], json.dumps(cleaned_tools[:120]), datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

    return jsonify({"ok": True, "tools": cleaned_tools[:120]})


@app.route("/api/founder/matches")
def api_founder_matches():
    if "user_id" not in session:
        return jsonify({"matches": []}), 401

    current_user_id = str(session["user_id"])

    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute(
        """
        SELECT users.id, users.name, users.email, founder_profiles.tools, founder_profiles.updated_at
        FROM founder_profiles
        JOIN users ON users.id = founder_profiles.user_id
        WHERE founder_profiles.user_id != ?
        ORDER BY founder_profiles.updated_at DESC
        LIMIT 80
        """,
        (current_user_id,)
    )
    rows = c.fetchall()
    conn.close()

    matches = []
    for user_id, name, email, tools_json, updated_at in rows:
        try:
            tools = json.loads(tools_json or "[]")
        except json.JSONDecodeError:
            tools = []

        display_name = (name or "").strip() or email.split("@")[0]
        matches.append({
            "id": str(user_id),
            "name": display_name,
            "role": "Founder on this app",
            "need": "Picked their tool DNA here, so this is a real app user.",
            "skills": tools,
            "looking": [],
            "focus": [],
            "fit": "Real person using the app. Match score is based on tool overlap and useful gaps.",
            "updatedAt": updated_at
        })

    return jsonify({"matches": matches})


@app.route("/api/memory")
def api_memory():
    user_id = session.get("user_id", "default")

    conn = sqlite3.connect(DB)
    c = conn.cursor()

    c.execute(
        "SELECT memory, created_at FROM long_term_memories WHERE user_id = ? ORDER BY id DESC LIMIT 50",
        (user_id,)
    )
    memories = [{"text": row[0], "created_at": row[1]} for row in c.fetchall()]

    c.execute("SELECT story FROM context_stories WHERE user_id = ?", (user_id,))
    story_row = c.fetchone()
    story = story_row[0] if story_row else ""

    c.execute(
        "SELECT role, content, created_at FROM chat_messages WHERE user_id = ? ORDER BY id DESC LIMIT 30",
        (user_id,)
    )
    recent = [{"role": row[0], "content": row[1], "created_at": row[2]} for row in c.fetchall()]
    conn.close()

    return jsonify({"memories": memories, "story": story, "recent": list(reversed(recent))})


@app.route("/api/memory/delete", methods=["POST"])
def api_memory_delete():
    user_id = session.get("user_id", "default")
    data = request.json or {}
    memory_text = data.get("text", "")

    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute(
        "DELETE FROM long_term_memories WHERE user_id = ? AND memory = ?",
        (user_id, memory_text)
    )
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


@app.route("/api/intro")
def api_intro():
    user_id = session.get("user_id", "default")

    # Pull context from DB
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT story FROM context_stories WHERE user_id = ?", (user_id,))
    story_row = c.fetchone()
    story = story_row[0] if story_row else ""
    c.execute(
        "SELECT memory FROM long_term_memories WHERE user_id = ? ORDER BY id DESC LIMIT 6",
        (user_id,)
    )
    facts = [r[0] for r in c.fetchall()]
    conn.close()

    context = ""
    if story:
        context += f"Context about the founder: {story}\n"
    if facts:
        context += "Key facts: " + "; ".join(facts) + "\n"

    prompt = (
        f"{context}\n"
        "Write a short, punchy greeting for the founder when they open their AI workspace. "
        "Use at most 30 words total. "
        "Make it feel personal, energetic, and grounded in their actual startup context if available. "
        "No menus. No long recap. No quotation marks. No markdown blockquotes. No 'Welcome back'. "
        "Sound like a sharp cofounder, not a chatbot. "
        "End with one direct question only if useful."
    )

    try:
        greeting = call_llm([{"role": "user", "content": prompt}])
    except Exception:
        greeting = "Let's build something worth building today. What's the one thing that moves the needle right now?"

    greeting = greeting.strip().strip('"').strip("'").strip()
    greeting = re.sub(r"^>\s*", "", greeting, flags=re.MULTILINE).strip()

    return jsonify({"greeting": greeting})


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json or {}
        message = data.get("message", "").strip()
        goals_state = data.get("goalsState")
        allow_goal_writes = bool(data.get("allowGoalWrites"))
        user_id = session.get("user_id", "default")

        if not message:
            return jsonify({"response": "Say something and I'll help.", "tools": []})

        if wants_memory_clear(message):
            clear_user_memory(user_id)
            return jsonify({
                "response": "Done. I cleared my memory, context story, and recent chat history. I’m starting fresh now.",
                "tools": ["Memory"],
                "activity": True,
                "activityTitle": "Memory cleared"
            })

        if not allow_goal_writes and user_allows_app_update(message):
            return jsonify({
                "response": "I can read Goals here, but I won’t change them from chat anymore. Use the Goals page/generate button for edits.",
                "tools": ["Goals"],
                "activity": False
            })

        if allow_goal_writes:
            response = ask_goal_writer(message, user_id, goals_state)
            response += build_server_app_command(message, response, goals_state)
            tools = ["Goals"]
        else:
            response = ask_ai(message, user_id, goals_state)
            tools = requested_app_reads(message)

        if has_app_command(response) and not allow_goal_writes:
            response = strip_goal_update(response)
            tools = [tool for tool in tools if tool != "Goals"]

        visible_response = clean_visible_response(response) or "Done."

        if not allow_goal_writes:
            save_chat_message(user_id, "user", message)
            save_chat_message(user_id, "assistant", visible_response)

            # Run memory updates less frequently to reduce token usage.
            if should_run_memory_update(user_id, message):
                threading.Thread(
                    target=update_memory,
                    args=(user_id, message, visible_response),
                    daemon=True
                ).start()

        return jsonify({"response": response, "tools": tools, "activity": False})

    except Exception as e:
        # Always return valid JSON so the frontend never crashes on .json()
        return jsonify({"response": "I hit a snag on my end. Give it a second and try again.", "tools": [], "activity": False}), 200


@app.route("/api/generate-map", methods=["POST"])
def generate_map():
    data = request.json or {}
    kind = (data.get("kind") or "launch").strip().lower()
    context = data.get("context") or {}

    prompt = f"""
You generate visual planning maps for a startup workspace UI.
Return ONLY valid JSON with this exact shape:
{{
  "title": "Short title",
  "subtitle": "One-line framing",
  "lanes": [
    {{
      "name": "Lane name",
      "nodes": [
        {{"title":"Node title","detail":"One short practical sentence"}}
      ]
    }}
  ]
}}

Rules:
- kind is "{kind}".
- Produce 3 to 5 lanes.
- Each lane has 2 to 4 nodes.
- Keep each node execution-focused and concrete.
- No markdown, no code fences, no extra keys.

Context:
{json.dumps(context, ensure_ascii=True)}
"""
    try:
        raw = call_llm(
            [{"role": "user", "content": prompt}],
            temperature=0.45,
            max_tokens=700
        )
        parsed = None
        try:
            parsed = parse_json_response(raw)
        except Exception:
            try:
                repair_prompt = (
                    "Fix this into valid JSON only. Do not add commentary.\n"
                    f"{raw}"
                )
                repaired = call_llm(
                    [{"role": "user", "content": repair_prompt}],
                    temperature=0.2,
                    max_tokens=420
                )
                parsed = parse_json_response(repaired)
            except Exception:
                parsed = None

        if not isinstance(parsed, dict):
            payload = safe_council_payload()
            COUNCIL_CACHE[idea_key] = {"ts": now_ts, "payload": payload}
            return jsonify(payload)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("lanes"), list):
            raise ValueError("Invalid map format")
        return jsonify({"ok": True, "map": parsed})
    except Exception:
        return jsonify(safe_council_payload()), 200


COUNCIL_ROLES = [
    {
        "id": "contrarian",
        "name": "The Contrarian",
        "focus": "Find blind spots, false assumptions, and downside risk."
    },
    {
        "id": "first_principles",
        "name": "First Principles",
        "focus": "Break the idea to fundamentals and root causes."
    },
    {
        "id": "executor",
        "name": "The Executor",
        "focus": "Turn strategy into immediate concrete next actions."
    },
    {
        "id": "customer_advocate",
        "name": "Customer Advocate",
        "focus": "Protect customer pain, value clarity, and usability."
    },
    {
        "id": "market_strategist",
        "name": "Market Strategist",
        "focus": "Assess positioning, moat, and go-to-market dynamics."
    }
]


@app.route("/api/council", methods=["POST"])
def council_feedback():
    try:
        data = request.json or {}
        idea = (data.get("idea") or "").strip()
        user_id = session.get("user_id", "default")
        goals_state = data.get("goalsState")
        if not idea:
            return jsonify({"ok": False, "error": "Missing idea."}), 200

        idea_key = hashlib.sha256(f"{user_id}:{idea.lower()}".encode("utf-8")).hexdigest()
        now_ts = datetime.now().timestamp()
        cached = COUNCIL_CACHE.get(idea_key)
        if cached and now_ts - cached["ts"] <= COUNCIL_CACHE_TTL_SECONDS:
            return jsonify(cached["payload"])

        memory_context = build_council_context(user_id, goals_state)
        debate_prompt = f"""
You are an AI startup council simulator. Return ONLY valid JSON:
{{
  "members":[
    {{"id":"contrarian","name":"The Contrarian","focus":"Skepticism","stance":"<=14 words","score":0-100}},
    {{"id":"first_principles","name":"First Principles","focus":"Fundamentals","stance":"<=14 words","score":0-100}},
    {{"id":"executor","name":"The Executor","focus":"Execution","stance":"<=14 words","score":0-100}},
    {{"id":"customer_advocate","name":"Customer Advocate","focus":"Customer","stance":"<=14 words","score":0-100}},
    {{"id":"market_strategist","name":"Market Strategist","focus":"Market","stance":"<=14 words","score":0-100}}
  ],
  "debate":[{{"speaker":"Member name","target":"Member/All","tone":"pushback/support/challenge","message":"<=12 words"}}],
  "metrics":{{
    "viability":0-100,
    "executionEase":0-100,
    "marketPull":0-100,
    "riskLevel":0-100
  }},
  "startup":{{
    "neededToStartUSD":0,
    "runwayMonths":0,
    "teamNeeded":["<=5 words","<=5 words"],
    "assetsNeeded":["<=6 words","<=6 words"],
    "profitMarginPct":0
  }},
  "kpis":[
    {{"name":"CAC Payback","valuePct":0}},
    {{"name":"Lead->Paid","valuePct":0}},
    {{"name":"30d Retention","valuePct":0}}
  ],
  "financialDiagram":{{
    "months":["M1","M2","M3","M4","M5","M6"],
    "revenue":[0,0,0,0,0,0],
    "costs":[0,0,0,0,0,0],
    "profit":[0,0,0,0,0,0]
  }},
  "nextSteps":["<=8 words","<=8 words","<=8 words"],
  "chatVerdict":"2 short chill sentences max."
}}

Rules:
- Use exactly the five members above.
- Debate must be 4 to 5 lines only.
- riskLevel is high when risk is high.
- Provide realistic non-zero startup + KPI estimates.

Idea:
{idea}

Founder context:
{memory_context}
"""

        raw = call_llm(
            [{"role": "user", "content": debate_prompt}],
            temperature=0.55,
            max_tokens=360
        )
        parsed = parse_json_response(raw)
        members = parsed.get("members", [])
        debate = parsed.get("debate", [])
        metrics = parsed.get("metrics", {})
        startup = parsed.get("startup", {})
        kpis = parsed.get("kpis", [])
        financial_diagram = parsed.get("financialDiagram", {})
        next_steps = parsed.get("nextSteps", [])
        chat_verdict = clean_visible_response(parsed.get("chatVerdict", "Solid concept with real upside, but we should validate demand before heavy build."))

        payload = {
            "ok": True,
            "provider": LLM_PROVIDER,
            "model": LLM_MODEL,
            "members": members,
            "debate": debate,
            "metrics": metrics,
            "startup": startup,
            "kpis": kpis[:6],
            "financialDiagram": financial_diagram,
            "nextSteps": next_steps[:5],
            "chatVerdict": chat_verdict
        }
        COUNCIL_CACHE[idea_key] = {"ts": now_ts, "payload": payload}
        if len(COUNCIL_CACHE) > 250:
            COUNCIL_CACHE.clear()
        return jsonify(payload)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 200


if __name__ == "__main__":
    app.run(debug=True)
