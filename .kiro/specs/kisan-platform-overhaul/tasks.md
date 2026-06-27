# Implementation Plan: Project-Kisan Platform Overhaul

## Overview

12 tasks in dependency order. Start with the repo restructure and cloud wiring (Tasks 1–2) since everything else depends on the new paths and Neon connection. Then fix memory (3–5), auth (6), sessions (7), streaming (8), dead nodes (9), farmer search (10), frontend (11), and finally dependency cleanup (12). Each task maps to one or more requirements from `requirements.md`.

## Tasks

- [x] 1. Repo restructure — move backend/ and Database/ into src/
  - Move `backend/` → `src/backend/` and `Database/` → `src/database/` using smart file moves
  - Add `__init__.py` to `src/`, `src/backend/`, `src/database/` and all sub-packages
  - Update all imports from `backend.*` → `src.backend.*` and `Database.*` → `src.database.*`
  - Remove all `sys.path.append` hacks; replace with proper package-relative imports
  - Update `pyproject.toml` so `src` is the package root
  - Verify `python -m src.backend.main` starts without import errors
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

- [x] 2. Cloud database wiring — Neon Postgres for ALL data
  - Update `.env_example` with `NEON_API`, `QDRANT_URL`, `QDRANT_API`; remove old sqlite/local defaults
  - Rewrite `src/database/database.py` to use `create_async_engine` with `NEON_API`; raise `RuntimeError` if missing
  - Update `src/database/models.py`: add `phone_number UNIQUE` index, `latitude`/`longitude` to `User`, add `Chat` table, add `FarmerLocation` table
  - Update `src/backend/core/config.py` to load `NEON_API`, `QDRANT_URL`, `QDRANT_API`
  - Update `src/backend/main.py` lifespan to call `init_db()` on Neon then setup LangGraph components
  - _Requirements: 9.1, 9.2, 9.3, 9.5, 9.6, 9.9_

- [x] 3. Short-term memory — replace AsyncSqliteSaver with AsyncPostgresSaver on Neon
  - Install `langgraph-checkpoint-postgres` dependency
  - Replace `AsyncSqliteSaver` with `AsyncPostgresSaver.from_conn_string(NEON_API)` in `src/ai_component/graph/graph.py`
  - Call `.setup()` on first use to auto-create checkpoint tables on Neon
  - Remove all local SQLite file path references (`data2/chat_history.db`)
  - Update `delete_thread()` to use LangGraph async delete API instead of raw SQL
  - Test: two messages on same thread_id — second sees first in history
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 9.2_

- [x] 4. Long-term memory — AsyncPostgresStore on Neon with per-user namespaces
  - Wire `AsyncPostgresStore.from_conn_string(NEON_API)` and call `.setup()` on first use
  - Rewrite `memory_manager.py` to write via `store.aput(("long_term", unique_name), key, {"summary": text})`
  - Before each node, call `store.asearch(("long_term", unique_name), query, limit=10)` and inject results into system context
  - Verify namespace isolation: user A memories are not returned for user B
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 9.3_

- [x] 5. Qdrant Cloud migration and PII removal from embeddings
  - Update `src/ai_component/modules/memory/vector_store.py`: use `QdrantClient(url=QDRANT_URL, api_key=QDRANT_API)`; raise `RuntimeError` if either missing
  - Strip PII from `ingest_data()` metadata — only allow `created_at`, `timestamp`, `collection`, `type`
  - Update `memory_manager.py` to not pass `user_phone`, `user_name`, `user_id`, `user_age`, `user_address` to `ingest_data()`
  - Test: ingest a record, verify stored metadata contains no PII fields
  - _Requirements: 5.2, 9.4, 9.7_

- [x] 6. Phone-number-based authentication
  - Update `UserLogin` schema to use `phone_number` with E.164 validator; `UserRegister` keeps both `unique_name` and `phone_number`
  - Add `get_user_by_phone()` method to `UserDatabase`
  - Update login endpoint to look up by `phone_number`; keep JWT `sub = unique_name`
  - Update register endpoint to check phone uniqueness; return 400 if duplicate; return identical error for wrong phone vs wrong password
  - Set access token expiry to 15 min, refresh token to 7 days
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8_

- [x] 7. Multi-chat session management with auto-naming
  - Create `ChatDatabase` class with `create_chat`, `list_chats`, `set_chat_name`, `increment_message_count`, `delete_chat` methods
  - Update `POST /thread/create` to insert into `chats` table
  - Update `GET /threads` to query `chats` table ordered by `updated_at DESC` instead of scanning LangGraph checkpoints
  - After first message response, call `set_chat_name(thread_id, query[:50])` and `increment_message_count(thread_id)`
  - Update `DELETE /thread/{thread_id}` to delete from `chats` table AND LangGraph checkpoints
  - Update `GET /thread/{thread_id}/messages` to load from `AsyncPostgresSaver` state
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8_

- [x] 8. Token-by-token SSE streaming
  - Replace per-node `astream` with `graph.astream_events(..., version="v2")`
  - Filter `on_chat_model_stream` events and yield individual token chunks as `data: {"content": token, "type": "token"}\n\n`
  - Emit `event: done\ndata: {}\n\n` when graph execution ends
  - Add 30-second stall timeout: if no token emitted, emit `event: error` and close generator
  - Add `X-Accel-Buffering: no` and `Cache-Control: no-cache` response headers
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

- [x] 9. Fix dead graph nodes
  - Fix `TextNode`: return `{"output": last_ai_message.content}` instead of `{}`
  - Fix `CarbonFootprintNode`: return a "Coming soon" `AIMessage`
  - Add graceful fallback to `VoiceNode` and `ImageNode` when API key is absent
  - Add node-status comment block at top of `nodes.py` listing each node's status
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8_

- [x] 10. Farmer-location search redesign
  - Add `FarmerLocationDatabase` with `upsert()` and `search_nearby()` (haversine SQL) methods
  - Call `upsert()` on user register and profile update
  - Update `GET /user/nearby` to use `search_nearby()` instead of Qdrant cross-collection scan
  - Remove Qdrant `search_across_collections()` from nearby-farmer flow
  - _Requirements: 5.1, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9_

- [x] 11. Frontend — login, signup, and streaming chat UI
  - Create `frontend/login.html` — phone + password form, stores tokens in localStorage, redirects to chat
  - Create `frontend/signup.html` — phone + unique_name + password + optional fields
  - Create `frontend/chat.html` — sidebar with chat list + delete buttons + New Chat; main message area; input box
  - Create `frontend/js/api.js` — `streamMessage()` via `fetch` ReadableStream SSE; `refreshTokenIfNeeded()`; CRUD helpers
  - Create `frontend/js/auth.js` — form handlers, token storage, redirect logic
  - Create `frontend/js/chat.js` — load chat list, load messages on select, render streaming tokens, delete with confirmation
  - Create `frontend/css/style.css` — responsive sidebar + main layout, message bubbles, loading indicator
  - Serve `frontend/` as FastAPI static files
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 8.10, 8.11_

- [x] 12. Dependency and packaging cleanup
  - Add to `pyproject.toml`/`requirements.txt`: `langgraph-checkpoint-postgres`, `asyncpg`, `sqlalchemy[asyncio]`
  - Remove `aiosqlite` dependency
  - Pin all new deps to exact versions
  - Run install in venv and confirm no conflicts
  - _Requirements: 9.5, 10.4_

## Task Dependency Graph

```json
{
  "tasks": [
    {"id": 1, "name": "Repo restructure", "deps": []},
    {"id": 2, "name": "Neon Postgres wiring", "deps": [1]},
    {"id": 3, "name": "Short-term memory (AsyncPostgresSaver)", "deps": [2]},
    {"id": 4, "name": "Long-term memory (AsyncPostgresStore)", "deps": [2]},
    {"id": 5, "name": "Qdrant Cloud migration + PII strip", "deps": [2]},
    {"id": 6, "name": "Phone-number auth", "deps": [2]},
    {"id": 7, "name": "Multi-chat session management", "deps": [6]},
    {"id": 8, "name": "Token-by-token SSE streaming", "deps": [7]},
    {"id": 9, "name": "Fix dead graph nodes", "deps": [1]},
    {"id": 10, "name": "Farmer-location search redesign", "deps": [2]},
    {"id": 11, "name": "Frontend", "deps": [7, 8]},
    {"id": 12, "name": "Dependency and packaging cleanup", "deps": [1]}
  ],
  "waves": [
    {"wave": 1, "tasks": [1, 12]},
    {"wave": 2, "tasks": [2, 9]},
    {"wave": 3, "tasks": [3, 4, 5, 6, 10]},
    {"wave": 4, "tasks": [7]},
    {"wave": 5, "tasks": [8]},
    {"wave": 6, "tasks": [11]}
  ]
}
```

## Notes

- `NEON_API` must use the `postgresql+asyncpg://` scheme prefix. The Neon console gives a standard `postgresql://` URL — replace `postgresql://` with `postgresql+asyncpg://` before pasting into `.env`.
- LangGraph's `AsyncPostgresSaver` and `AsyncPostgresStore` create their own tables automatically on first `.setup()` call. No manual DDL needed on Neon.
- JWT `sub` stays as `unique_name` throughout — this is the Qdrant collection name and user identifier inside the agent. Only the login *input* changes to `phone_number`.
- `CarbonFootprintNode` is marked "Coming soon" because no free carbon-data API is available without a new key. It can be upgraded later.
- The frontend uses vanilla HTML/CSS/JS — no build step, no npm. FastAPI serves it directly as static files.
