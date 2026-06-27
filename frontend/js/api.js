/**
 * api.js — HTTP / SSE helpers for Project-Kisan frontend
 *
 * Key exports:
 *   refreshTokenIfNeeded()
 *   login(phoneNumber, password)
 *   register(data)
 *   logout()
 *   createThread()
 *   getThreads()
 *   getMessages(threadId)
 *   deleteThread(threadId)
 *   streamMessage(query, threadId, onToken, onDone, onError)
 */

const API_BASE = '/api/v1';

// ---------------------------------------------------------------------------
// Token storage helpers
// ---------------------------------------------------------------------------

/** Decode the base64url JWT payload and return the parsed object. */
function decodeJwtPayload(token) {
  try {
    const base64Url = token.split('.')[1];
    if (!base64Url) return null;
    // base64url → base64
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
    const json = atob(base64.padEnd(base64.length + (4 - base64.length % 4) % 4, '='));
    return JSON.parse(json);
  } catch {
    return null;
  }
}

/** Store tokens after a successful auth response. */
function storeTokens(data) {
  localStorage.setItem('access_token', data.access_token);
  localStorage.setItem('refresh_token', data.refresh_token);

  // Compute expiry timestamp (ms) from JWT exp claim
  const payload = decodeJwtPayload(data.access_token);
  if (payload && payload.exp) {
    // exp is seconds since epoch
    localStorage.setItem('expires_at', String(payload.exp * 1000));
  } else if (data.expires_in) {
    localStorage.setItem('expires_at', String(Date.now() + data.expires_in * 1000));
  }
}

function clearTokens() {
  localStorage.removeItem('access_token');
  localStorage.removeItem('refresh_token');
  localStorage.removeItem('expires_at');
}

function getAccessToken() {
  return localStorage.getItem('access_token');
}

function getRefreshToken() {
  return localStorage.getItem('refresh_token');
}

function getExpiresAt() {
  const v = localStorage.getItem('expires_at');
  return v ? parseInt(v, 10) : 0;
}

/** Returns true if a valid access token exists and the user is considered logged in. */
export function isAuthenticated() {
  return !!getAccessToken();
}

// ---------------------------------------------------------------------------
// Token refresh
// ---------------------------------------------------------------------------

let _refreshPromise = null; // deduplicate concurrent refresh calls

/**
 * If the access token expires within 60 seconds, refresh it.
 * Returns true if refresh was performed (or not needed), false on failure.
 */
export async function refreshTokenIfNeeded() {
  const expiresAt = getExpiresAt();
  const nowPlusBuffer = Date.now() + 60_000; // 60-second buffer

  // Token still has more than 60 s remaining — nothing to do
  if (expiresAt > nowPlusBuffer) return true;

  const refreshToken = getRefreshToken();
  if (!refreshToken) return false;

  // Deduplicate: if a refresh is already in-flight, wait for it
  if (_refreshPromise) return _refreshPromise;

  _refreshPromise = (async () => {
    try {
      const res = await fetch(`${API_BASE}/auth/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: refreshToken }),
      });

      if (!res.ok) {
        clearTokens();
        return false;
      }

      const data = await res.json();
      storeTokens(data);
      return true;
    } catch {
      return false;
    } finally {
      _refreshPromise = null;
    }
  })();

  return _refreshPromise;
}

/**
 * Extracts a human-readable error message from a FastAPI error response body.
 * FastAPI 422 responses return detail as an array of validation error objects.
 */
function extractDetail(body, fallback) {
  if (!body) return fallback;
  if (typeof body.detail === 'string') return body.detail;
  if (Array.isArray(body.detail) && body.detail.length > 0) {
    // Each item has { loc: [...], msg: "...", type: "..." }
    return body.detail.map(e => e.msg).join('; ');
  }
  return body.message || fallback;
}

// ---------------------------------------------------------------------------
// Core fetch wrapper
// ---------------------------------------------------------------------------

/**
 * Authenticated fetch — automatically refreshes token if needed.
 * Throws an Error whose message contains the backend detail string on non-2xx.
 */
async function apiFetch(path, options = {}) {
  await refreshTokenIfNeeded();

  const token = getAccessToken();
  const headers = {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...(options.headers || {}),
  };

  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = extractDetail(body, detail);
    } catch { /* ignore parse error */ }
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }

  // 204 No Content
  if (res.status === 204) return null;

  return res.json();
}

// ---------------------------------------------------------------------------
// Auth endpoints
// ---------------------------------------------------------------------------

/**
 * POST /auth/login
 * Returns the full token response and stores tokens in localStorage.
 */
export async function login(phoneNumber, password) {
  const data = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ phone_number: phoneNumber, password }),
  }).then(async (res) => {
    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try {
        const body = await res.json();
        detail = extractDetail(body, detail);
      } catch { /* ignore */ }
      const err = new Error(detail);
      err.status = res.status;
      throw err;
    }
    return res.json();
  });

  storeTokens(data);
  return data;
}

/**
 * POST /auth/register
 * Returns the full token response and stores tokens in localStorage.
 */
export async function register(formData) {
  const data = await fetch(`${API_BASE}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(formData),
  }).then(async (res) => {
    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try {
        const body = await res.json();
        detail = extractDetail(body, detail);
      } catch { /* ignore */ }
      const err = new Error(detail);
      err.status = res.status;
      throw err;
    }
    return res.json();
  });

  storeTokens(data);
  return data;
}

/** POST /auth/logout — clears local tokens regardless of server response. */
export async function logout() {
  try {
    await apiFetch('/auth/logout', { method: 'POST' });
  } catch { /* best-effort */ }
  clearTokens();
}

// ---------------------------------------------------------------------------
// Chat / thread CRUD
// ---------------------------------------------------------------------------

/** POST /chat/thread/create → { thread_id, thread_name, ... } */
export async function createThread() {
  return apiFetch('/chat/thread/create', {
    method: 'POST',
    body: JSON.stringify({}),
  });
}

/** GET /chat/threads → { threads: [...], total, limit, offset, has_more } */
export async function getThreads({ limit = 50, offset = 0 } = {}) {
  return apiFetch(`/chat/threads?limit=${limit}&offset=${offset}`);
}

/** GET /chat/thread/{threadId}/messages → { messages: [...] } */
export async function getMessages(threadId) {
  return apiFetch(`/chat/thread/${encodeURIComponent(threadId)}/messages`);
}

/** DELETE /chat/thread/{threadId} → { message: "..." } */
export async function deleteThread(threadId) {
  return apiFetch(`/chat/thread/${encodeURIComponent(threadId)}`, {
    method: 'DELETE',
  });
}

// ---------------------------------------------------------------------------
// SSE streaming
// ---------------------------------------------------------------------------

/**
 * streamMessage — streams a chat response token by token via SSE.
 *
 * Uses fetch + ReadableStream rather than EventSource so we can attach
 * Authorization headers (EventSource does not support custom headers).
 *
 * SSE protocol used by the backend:
 *   data: {"content": "token", "type": "token"}   ← normal token chunk
 *   event: done\ndata: {}                          ← stream complete
 *   event: error\ndata: {"detail": "..."}          ← explicit backend error
 *
 * Requirements:
 *   8.5  — render each token as it arrives
 *   8.10 — explicit error event stops rendering; connection loss does NOT
 *
 * @param {string}   query     User's message text
 * @param {string}   threadId  Active thread identifier
 * @param {Function} onToken   Called with each token string as it arrives
 * @param {Function} onDone    Called when stream completes normally
 * @param {Function} onError   Called with an Error when an explicit SSE error arrives
 * @returns {Function} abort — call to cancel the in-flight request
 */
export function streamMessage(query, threadId, onToken, onDone, onError) {
  const controller = new AbortController();

  (async () => {
    // Ensure token is fresh before opening stream
    const ok = await refreshTokenIfNeeded();
    if (!ok) {
      onError(new Error('Session expired. Please log in again.'));
      return;
    }

    const token = getAccessToken();
    const url = `${API_BASE}/chat/message/stream?` +
      `query=${encodeURIComponent(query)}` +
      `&thread_id=${encodeURIComponent(threadId)}` +
      `&workflow=GeneralNode`;

    let res;
    try {
      res = await fetch(url, {
        method: 'GET',
        headers: {
          Authorization: `Bearer ${token}`,
          Accept: 'text/event-stream',
        },
        signal: controller.signal,
      });
    } catch (fetchErr) {
      // Network / connection errors (Req 8.10: do NOT call onError for connection loss)
      if (fetchErr.name !== 'AbortError') {
        // Connection lost — silently stop; do not show error per requirement 8.10
        onDone();
      }
      return;
    }

    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try {
        const body = await res.json();
        detail = extractDetail(body, detail);
      } catch { /* ignore */ }
      onError(new Error(detail));
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          // Stream closed by server without explicit done event — treat as complete
          onDone();
          break;
        }

        buffer += decoder.decode(value, { stream: true });

        // Process complete SSE messages (delimited by blank line \n\n)
        const parts = buffer.split('\n\n');
        buffer = parts.pop(); // keep incomplete trailing portion

        for (const part of parts) {
          const lines = part.split('\n');
          let eventName = 'message';
          let dataLine = '';

          for (const line of lines) {
            if (line.startsWith('event:')) {
              eventName = line.slice(6).trim();
            } else if (line.startsWith('data:')) {
              dataLine = line.slice(5).trim();
            }
          }

          if (eventName === 'done') {
            onDone();
            reader.cancel();
            return;
          }

          if (eventName === 'error') {
            // Explicit server error — stop rendering (Req 8.10)
            let detail = 'Streaming error';
            try {
              const parsed = JSON.parse(dataLine);
              detail = parsed.detail || detail;
            } catch { /* ignore */ }
            onError(new Error(detail));
            reader.cancel();
            return;
          }

          // Normal data frame
          if (dataLine) {
            try {
              const parsed = JSON.parse(dataLine);
              if (parsed.content) {
                onToken(parsed.content);
              }
            } catch { /* ignore malformed frames */ }
          }
        }
      }
    } catch (readErr) {
      if (readErr.name !== 'AbortError') {
        // Read error / connection drop — per Req 8.10, do NOT trigger error UI
        onDone();
      }
    }
  })();

  return () => controller.abort();
}
