# Technical Design Document

## Overview

Project-Kisan is an agentic farming assistant. This overhaul fixes broken short-term memory, dead graph nodes, and non-streaming output; adds per-user long-term memory, phone-number auth, multi-chat session management, and a web frontend; migrates **all** persistence to cloud (Neon Postgres + Qdrant Cloud); and restructures the repo so `backend/` and `Database/` move under `src/`.

**Every piece of data that currently lives in SQLite or a local Qdrant instance moves to Neon Postgres or Qdrant Cloud:**

| Data | Storage | Neon Postgres table / LangGraph component |
|---|---|---|
| User accounts | Neon Postgres | `users` |
| Chat metadata | Neon Postgres | `chats` |
| Farmer locations | Neon Postgres | `farmer_locations` |
| Short-term memory (per thread) | Neon Postgres | LangGraph `AsyncPostgresSaver` |
| Long-term memory (per user) | Neon Postgres | LangGraph `AsyncPostgresStore` |
| Conversation embeddings | Qdrant Cloud | one collection per `unique_name` |

No SQLite files. No local Qdrant. The single `NEON_API` env var drives all five Postgres consumers.


## Architecture

```
Browser (HTML/CSS/JS)
  │  login/signup   ──POST──▶ /api/v1/auth/*
  │  chat (stream)  ──GET SSE─▶ /api/v1/chat/message/stream
  │  chat list/del  ──REST──▶ /api/v1/chat/*
  └──────────────────────────────────────────────────────────
                        FastAPI  (src/backend/main.py)
                            │
          ┌─────────────────┼──────────────────┐
          ▼                 ▼                  ▼
   Neon Postgres       Qdrant Cloud       LangGraph Agent
  (all relational      (per-user conv.    (src/ai_component)
   + checkpoints       embeddings,         │
   + long-term store   one collection      ├─ AsyncPostgresSaver  ──▶ Neon
   + farmer_locations) per unique_name)    └─ AsyncPostgresStore  ──▶ Neon
```

**Request flow (streaming chat):**
1. Browser sends `GET /api/v1/chat/message/stream?query=...&thread_id=...` with Bearer token.
2. FastAPI verifies JWT, extracts `unique_name` and `user_id`.
3. `chat.py` router calls `graph.astream_events(state, config, version="v2")`.
4. Filter `on_chat_model_stream` events → yield token chunks as SSE `data:` frames.
5. On graph completion emit `event: done` SSE frame.
6. `AsyncPostgresSaver` has already persisted the checkpoint to Neon mid-run.
7. `MemoryIngestionNode` writes durable facts to `AsyncPostgresStore` (Neon) and summary embeddings to Qdrant Cloud.


## Components and Interfaces

### Repository layout after restructure

```
Project-Kisan/
├── src/
│   ├── __init__.py
│   ├── backend/                  ← was backend/
│   │   ├── main.py
│   │   ├── core/
│   │   │   ├── auth.py           ← phone_number login, 15-min access token
│   │   │   └── config.py         ← adds NEON_API, QDRANT_URL, QDRANT_API
│   │   ├── routers/
│   │   │   ├── auth.py
│   │   │   ├── chat.py           ← astream_events SSE streaming
│   │   │   └── user.py           ← /nearby uses haversine SQL
│   │   ├── schemas/
│   │   │   └── schemas.py        ← UserLogin uses phone_number
│   │   └── utils/db_utils.py
│   ├── database/                 ← was Database/
│   │   ├── models.py             ← User, Chat, FarmerLocation tables
│   │   └── database.py           ← async SQLAlchemy + ChatDatabase class
│   └── ai_component/             ← unchanged location, internal fixes
│       ├── graph/
│       │   ├── graph.py          ← AsyncPostgresSaver + AsyncPostgresStore
│       │   ├── nodes.py          ← TextNode fix, CarbonFootprint "Coming soon"
│       │   └── state.py
│       └── modules/memory/
│           ├── vector_store.py   ← Qdrant Cloud client with api_key, no PII metadata
│           └── memory_manager.py ← writes to AsyncPostgresStore, strips PII
├── frontend/
│   ├── login.html
│   ├── signup.html
│   ├── chat.html
│   ├── css/style.css
│   └── js/
│       ├── api.js                ← fetch + SSE helpers, token refresh
│       ├── auth.js               ← login/signup form logic
│       └── chat.js               ← sidebar, SSE rendering, chat CRUD
└── .env  (NEON_API, QDRANT_URL, QDRANT_API, SECRET_KEY, ...)
```

### Key module interfaces

**`src/database/database.py`**
- `UserDatabase.get_user_by_phone(phone: str) → User | None`
- `UserDatabase.get_user_by_unique_name(name: str) → User | None`
- `ChatDatabase.create_chat(user_id, thread_id) → Chat`
- `ChatDatabase.list_chats(user_id) → list[Chat]` — ordered by `updated_at DESC`
- `ChatDatabase.set_chat_name(thread_id, name)` — called after first message
- `ChatDatabase.increment_message_count(thread_id)`
- `ChatDatabase.delete_chat(thread_id, user_id) → bool`
- `FarmerLocationDatabase.upsert(user_id, lat, lng, phone, district, state)`
- `FarmerLocationDatabase.search_nearby(lat, lng, radius_km, exclude_user_id) → list`

**`src/ai_component/graph/graph.py`**
- `get_graph() → CompiledGraph` — singleton, built with `AsyncPostgresSaver`
- `get_store() → AsyncPostgresStore` — singleton long-term memory store
- `stream_events(query, thread_id, collection_name) → AsyncGenerator[str, None]` — yields raw SSE lines

**`src/backend/routers/chat.py`**
- `POST /thread/create` → `{thread_id, name, created_at}`
- `GET  /threads` → `[{thread_id, name, message_count, updated_at}]`
- `GET  /thread/{thread_id}/messages` → `[{role, content}]`
- `DELETE /thread/{thread_id}` → `{ok: true}`
- `GET  /message/stream` (SSE) → token chunks + `event: done`
- `POST /message` → `{content, media_type, thread_id}`


## Data Models

### `users` table (Neon Postgres)

```sql
CREATE TABLE users (
    id            SERIAL PRIMARY KEY,
    unique_name   VARCHAR(50)  UNIQUE NOT NULL,   -- Qdrant collection name + JWT sub
    phone_number  VARCHAR(20)  UNIQUE NOT NULL,   -- login key (E.164)
    hashed_password VARCHAR(128) NOT NULL,
    full_name     VARCHAR(100),
    age           INTEGER,
    resident      VARCHAR(200),
    city          VARCHAR(100),
    district      VARCHAR(100),
    state         VARCHAR(100),
    country       VARCHAR(100),
    created_at    TIMESTAMPTZ  DEFAULT NOW(),
    updated_at    TIMESTAMPTZ  DEFAULT NOW()
);
```

`phone_number` has a UNIQUE index — it is the login credential.
`unique_name` is kept as the Qdrant collection name and JWT `sub` so downstream code is unchanged.

### `chats` table (Neon Postgres)

```sql
CREATE TABLE chats (
    id            SERIAL PRIMARY KEY,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    thread_id     VARCHAR(64) UNIQUE NOT NULL,    -- format: user_{user_id}_{uuid8}
    name          VARCHAR(50),                    -- set from first 50 chars of first message
    message_count INTEGER NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at    TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_chats_user_id ON chats(user_id);
```

### `farmer_locations` table (Neon Postgres)

```sql
CREATE TABLE farmer_locations (
    id           SERIAL PRIMARY KEY,
    user_id      INTEGER UNIQUE NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    phone_number VARCHAR(20) NOT NULL,
    district     VARCHAR(100),
    state        VARCHAR(100),
    country      VARCHAR(100),
    latitude     FLOAT,
    longitude    FLOAT,
    updated_at   TIMESTAMPTZ DEFAULT NOW()
);
```

One row per user (`user_id UNIQUE`). Upserted on profile create/update.

### LangGraph tables (auto-created by `setup()`)

`AsyncPostgresSaver.setup()` creates: `checkpoints`, `checkpoint_blobs`, `checkpoint_writes`.
`AsyncPostgresStore.setup()` creates: `store`.

No manual DDL needed — these are created at first startup.

### Qdrant Cloud collections

One collection per `unique_name`, named exactly `<unique_name>`.
Vector size 768 (Google `embedding-001`), cosine distance.
**Metadata stored per point:** `created_at`, `timestamp`, `collection`, `type` — no PII fields (no phone, name, user_id, age, address).


## Detailed Design

### 6. Cloud connection bootstrapping (`src/database/database.py`)

```python
import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

NEON_API = os.getenv("NEON_API")
if not NEON_API:
    raise RuntimeError("NEON_API environment variable is not set")

# asyncpg driver required for AsyncPostgresSaver and async SQLAlchemy
engine = create_async_engine(NEON_API, pool_pre_ping=True, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

async def init_db():
    from src.database.models import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
```

The same `NEON_API` string (with `postgresql+asyncpg://` scheme) is passed to both `AsyncPostgresSaver` and `AsyncPostgresStore` in `graph.py`.

For sync operations (passlib, legacy code) derive a psycopg2 URL:
```python
SYNC_DB_URL = NEON_API.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
```

### 7. LangGraph memory wiring (`src/ai_component/graph/graph.py`)

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

_saver: AsyncPostgresSaver | None = None
_store: AsyncPostgresStore | None = None

async def get_saver() -> AsyncPostgresSaver:
    global _saver
    if _saver is None:
        _saver = AsyncPostgresSaver.from_conn_string(os.getenv("NEON_API"))
        await _saver.setup()   # creates checkpoint tables if not present
    return _saver

async def get_store() -> AsyncPostgresStore:
    global _store
    if _store is None:
        _store = AsyncPostgresStore.from_conn_string(os.getenv("NEON_API"))
        await _store.setup()   # creates store table if not present
    return _store

async def get_graph():
    saver = await get_saver()
    store = await get_store()
    # graph_builder.compile(checkpointer=saver, store=store)
    ...
```

**Long-term memory namespace per user:**
```python
namespace = ("long_term", user_unique_name)   # e.g. ("long_term", "alice123")
await store.aput(namespace, key=str(uuid4()), value={"summary": summary_text})
results = await store.asearch(namespace, query=query, limit=10)
```

Each user's memories are completely isolated by namespace. No user can read another's.

### 8. Token streaming fix (`src/backend/routers/chat.py`)

Current code streams per-node output (one blob per node). The fix uses `astream_events`:

```python
from fastapi.responses import StreamingResponse

async def token_stream(query, thread_id, collection_name):
    graph = await get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    state  = {"messages": [{"role": "user", "content": query}],
               "collection_name": collection_name, "workflow": "GeneralNode"}

    async for event in graph.astream_events(state, config=config, version="v2"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            token = chunk.content if hasattr(chunk, "content") else ""
            if token:
                yield f"data: {json.dumps({'content': token, 'type': 'token'})}\n\n"
        elif kind == "on_chain_end" and event.get("name") == "LangGraph":
            yield f"event: done\ndata: {{}}\n\n"
            return

@router.get("/message/stream")
async def stream_message(query: str, thread_id: str, current_user=Depends(verify_token)):
    if not thread_id.startswith(f"user_{current_user['id']}_"):
        raise HTTPException(403, "Access denied")
    return StreamingResponse(
        token_stream(query, thread_id, current_user["unique_name"]),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
```

### 9. Farmer-location search (replaces Qdrant metadata scan)

The `user.py` router `/nearby` endpoint now runs a haversine SQL query directly on `farmer_locations`:

```python
HAVERSINE_SQL = """
SELECT fl.user_id, fl.phone_number, fl.district, fl.state, fl.country,
       (6371 * acos(
           cos(radians(:lat)) * cos(radians(fl.latitude)) *
           cos(radians(fl.longitude) - radians(:lng)) +
           sin(radians(:lat)) * sin(radians(fl.latitude))
       )) AS distance_km
FROM farmer_locations fl
WHERE fl.user_id != :exclude_id
  AND fl.latitude IS NOT NULL
  AND fl.longitude IS NOT NULL
HAVING distance_km <= :radius
ORDER BY distance_km ASC
LIMIT 100
"""
```

On user register/update, `FarmerLocationDatabase.upsert()` writes one row. Location data never goes into Qdrant metadata.

### 10. Dead nodes resolution

| Node | Action | Reason |
|---|---|---|
| `TextNode` | Fix: return `{"output": last_ai_message.content}` | Was returning `{}` — loses the response |
| `CarbonFootprintNode` | "Coming soon" message | Carbon data APIs require a new API key |
| `VoiceNode` | Keep as-is (needs `CARTESIA_API_KEY`) | Existing key — works if key provided |
| `ImageNode` | Keep as-is (needs `TOGETHER_API_KEY`) | Existing key — works if key provided |

`TextNode` fix:
```python
@staticmethod
async def TextNode(state: AICompanionState) -> dict:
    last_ai = next((m for m in reversed(state["messages"])
                    if isinstance(m, AIMessage)), None)
    return {"output": last_ai.content if last_ai else ""}
```

`CarbonFootprintNode` fix:
```python
@staticmethod
async def CarbonFootprintNode(state: AICompanionState) -> dict:
    msg = ("Carbon footprint estimation is coming soon. "
           "This feature will calculate your farm's carbon output "
           "based on crop type, fertilizer use, and irrigation.")
    return {"messages": [AIMessage(content=msg)]}
```

### 11. Qdrant Cloud client (`src/ai_component/modules/memory/vector_store.py`)

```python
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API = os.getenv("QDRANT_API")
if not QDRANT_URL or not QDRANT_API:
    raise RuntimeError("QDRANT_URL and QDRANT_API must be set")

self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API, prefer_grpc=False)
```

PII stripped from metadata — only timing/type fields allowed:
```python
metadata = {
    "created_at": datetime.now().isoformat(),
    "timestamp":  datetime.now().timestamp(),
    "collection": collection_name,
    "type":       additional_metadata.get("type", "conversation_summary")
    # NO phone_number, user_name, user_id, user_age, user_address
}
```


### 12. Chat session management

`ChatDatabase` (in `src/database/database.py`) handles all chat CRUD:

```python
class ChatDatabase:
    async def create_chat(self, user_id: int) -> Chat:
        thread_id = f"user_{user_id}_{uuid.uuid4().hex[:8]}"
        chat = Chat(user_id=user_id, thread_id=thread_id)
        # persist to Neon via AsyncSessionLocal
        return chat

    async def list_chats(self, user_id: int) -> list[Chat]:
        # SELECT * FROM chats WHERE user_id=? ORDER BY updated_at DESC

    async def set_chat_name(self, thread_id: str, first_message: str):
        name = (first_message.strip() or "New Chat")[:50]
        # UPDATE chats SET name=? WHERE thread_id=?

    async def increment_message_count(self, thread_id: str):
        # UPDATE chats SET message_count=message_count+1, updated_at=NOW()

    async def delete_chat(self, thread_id: str, user_id: int) -> bool:
        # DELETE FROM chats WHERE thread_id=? AND user_id=?
        # Also delete checkpoints: saver.adelete({"configurable":{"thread_id":thread_id}})
        return rows_deleted > 0
```

Chat naming happens in `chat.py` router after the first successful message response:
```python
chat_count = await chat_db.get_message_count(thread_id)
if chat_count == 0:
    await chat_db.set_chat_name(thread_id, message.query)
await chat_db.increment_message_count(thread_id)
```

### 13. Frontend design

Three HTML pages, no build step, vanilla JS:

**`login.html`** — phone number + password form. On success stores `access_token` + `refresh_token` in `localStorage`, redirects to `chat.html`.

**`signup.html`** — phone + unique_name + password + optional fields. On success auto-logs in.

**`chat.html`** layout:
```
┌─────────────────────────────────────────────────┐
│  Project-Kisan          [New Chat]  [Logout]    │
├──────────────┬──────────────────────────────────┤
│  Chat list   │  Messages area                   │
│  (sidebar)   │                                  │
│  ─────────   │  User: ...                       │
│  Chat 1      │  AI:   ... (streams token-by-    │
│  Chat 2  🗑  │         token as SSE arrives)    │
│  ...         │                                  │
│              ├──────────────────────────────────┤
│              │  [input box]        [Send]        │
└──────────────┴──────────────────────────────────┘
```

`api.js` key functions:
- `streamMessage(query, threadId, onToken, onDone, onError)` — opens SSE via `EventSource` or `fetch` with `ReadableStream`, calls `onToken(chunk)` per `data:` frame, `onDone()` on `event: done`.
- `refreshTokenIfNeeded()` — checks token expiry, calls `POST /api/v1/auth/refresh` automatically.

### 14. Auth wiring summary

| Field | Registration | Login | JWT `sub` |
|---|---|---|---|
| `phone_number` | Required, E.164 | Login key | — |
| `unique_name` | Required | — | ✓ (collection_name) |
| `password` | 8–128 chars | Verified | — |

Access token: 15-minute expiry. Refresh token: 7-day expiry. Both signed with `SECRET_KEY` (HS256).


## Correctness Properties

### Property 1: Per-user memory isolation
Every `AsyncPostgresStore` read/write uses `namespace = ("long_term", unique_name)`. Cross-user leakage is structurally impossible — a different namespace returns a different set of rows.
**Validates: Requirements 4.2, 4.3, 4.4**

### Property 2: Thread ownership
`thread_id` is `user_{user_id}_{uuid8}`. Every API endpoint validates the prefix before acting, and the `chats` table enforces a `user_id` FK so DB-level orphan threads cannot exist.
**Validates: Requirements 7.1, 7.5**

### Property 3: Idempotent store setup
Both `AsyncPostgresSaver.setup()` and `AsyncPostgresStore.setup()` use `CREATE TABLE IF NOT EXISTS` — safe to call on every startup without data loss.
**Validates: Requirements 1.1, 9.9**

### Property 4: No PII in Qdrant
`ingest_data()` only ever writes four allowed metadata keys (`created_at`, `timestamp`, `collection`, `type`). The memory manager no longer passes phone, name, user_id, age, or address fields.
**Validates: Requirements 5.2**

### Property 5: Farmer location uniqueness
`farmer_locations.user_id` has a UNIQUE constraint — exactly one row per farmer, always replaced on update.
**Validates: Requirements 5.1, 5.5**

### Property 6: TextNode passthrough correctness
The fixed node reads the last `AIMessage` from state. Because the `add_messages` reducer accumulates messages in order, the correct AI response is always the most recent `AIMessage`.
**Validates: Requirements 2.5, 11.4**

## Error Handling

| Failure point | Behaviour |
|---|---|
| `NEON_API` not set | `RuntimeError` at import time — server won't start |
| `QDRANT_URL`/`QDRANT_API` not set | `RuntimeError` at `LongTermMemory.__init__` |
| Postgres connection timeout (30 s) | `asyncpg.TimeoutError` caught in lifespan; logged; component marked unavailable; server starts without that component |
| Qdrant Cloud unreachable | `QdrantException` caught in `ingest_data`/`search_in_collection`; logged; returns empty/False |
| LangGraph checkpoint write failure | Propagated as HTTP 500 with descriptive detail |
| SSE streaming error | `yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"` then generator returns |
| Chat not found / wrong owner | HTTP 403 (not 404, to avoid leaking existence) |
| Refresh token invalid/expired | HTTP 401 `"Invalid or expired refresh token"` |
| Phone already registered | HTTP 400 `"Phone number already registered"` |
| Phone format invalid | HTTP 422 (Pydantic validation) with E.164 hint |

## Testing Strategy

Since no test framework currently exists this section records what **should** be covered once tests are added:

- **Unit:** `validate_e164()`, `ChatDatabase.set_chat_name()` truncation, `FarmerLocationDatabase.upsert()` uniqueness, `TextNode` passthrough, metadata PII stripping in `ingest_data`.
- **Integration:** Startup with valid/missing env vars; `AsyncPostgresSaver` round-trip (write checkpoint, reload, verify message history); `AsyncPostgresStore` namespace isolation (two users, cross-read returns empty); SSE endpoint delivers `on_chat_model_stream` chunks before `event: done`.
- **E2E:** Register → login → create chat → send message (streaming) → reload chat → delete chat → confirm 403 on re-access.
- **Migration:** Run on fresh Neon DB — all five table groups created automatically on first startup, no manual DDL.

