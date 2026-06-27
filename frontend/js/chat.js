/**
 * chat.js — Sidebar, message rendering, SSE streaming, chat CRUD
 *
 * Requirements covered:
 *   8.3  — chat screen with sidebar
 *   8.4  — load message history on select
 *   8.5  — render streaming tokens as they arrive
 *   8.6  — delete with confirmation; remove from sidebar on success
 *   8.7  — New Chat creates thread, makes it active
 *   8.9  — failed deletion keeps chat in sidebar, shows error
 *   8.10 — explicit SSE error stops rendering; connection loss does not
 *   8.11 — history load failure shows error, retains current chat
 */

import {
  isAuthenticated,
  logout,
  createThread,
  getThreads,
  getMessages,
  deleteThread,
  streamMessage,
} from './api.js';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let threads = [];           // [{thread_id, thread_name, message_count, updated_at}]
let activeThreadId = null;  // currently selected thread
let abortStream = null;     // cancel function returned by streamMessage()
let isStreaming = false;    // true while SSE stream is in progress

// Pagination state
const PAGE_SIZE = 50;
let currentOffset = 0;
let totalThreads = 0;
let hasMore = false;
let isLoadingMore = false;

// ---------------------------------------------------------------------------
// DOM refs (populated after DOMContentLoaded)
// ---------------------------------------------------------------------------

let chatListEl, sidebarEmptyEl, sidebarErrorEl;
let messagesAreaEl, chatWelcomeEl, historyErrorEl;
let inputTextarea, sendBtn;
let newChatBtn, logoutBtn, themeToggleBtn;
let deleteDialog, deleteDialogMsg, deleteCancelBtn, deleteConfirmBtn;
let sidebarToggleBtn, sidebarBackdropEl, sidebarEl;
let loadMoreBtn;

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  if (!isAuthenticated()) {
    window.location.href = '/login.html';
    return;
  }

  // Bind DOM refs
  chatListEl        = document.getElementById('chat-list');
  sidebarEmptyEl    = document.getElementById('sidebar-empty');
  sidebarErrorEl    = document.getElementById('sidebar-error');
  messagesAreaEl    = document.getElementById('messages-area');
  chatWelcomeEl     = document.getElementById('chat-welcome');
  historyErrorEl    = document.getElementById('history-error');
  inputTextarea     = document.getElementById('message-input');
  sendBtn           = document.getElementById('send-btn');
  newChatBtn        = document.getElementById('new-chat-sidebar-btn');
  logoutBtn         = document.getElementById('logout-btn');
  themeToggleBtn    = document.getElementById('theme-toggle-btn');
  deleteDialog      = document.getElementById('delete-dialog');
  deleteDialogMsg   = document.getElementById('delete-dialog-name');
  deleteCancelBtn   = document.getElementById('delete-cancel-btn');
  deleteConfirmBtn  = document.getElementById('delete-confirm-btn');
  sidebarToggleBtn  = document.getElementById('sidebar-toggle');
  sidebarBackdropEl = document.getElementById('sidebar-backdrop');
  sidebarEl         = document.getElementById('sidebar');
  loadMoreBtn       = document.getElementById('load-more-btn');

  // Event listeners
  newChatBtn.addEventListener('click', handleNewChat);
  logoutBtn.addEventListener('click', handleLogout);
  sendBtn.addEventListener('click', handleSend);
  deleteCancelBtn.addEventListener('click', closeDeleteDialog);
  sidebarToggleBtn?.addEventListener('click', openSidebar);
  sidebarBackdropEl?.addEventListener('click', closeSidebar);
  loadMoreBtn?.addEventListener('click', loadMoreThreads);

  // Dark mode toggle
  initTheme();
  themeToggleBtn?.addEventListener('click', toggleTheme);

  // Suggestion chips on welcome screen
  document.querySelectorAll('.chip[data-query]').forEach((chip) => {
    chip.addEventListener('click', async () => {
      // If no active thread, create one first
      if (!activeThreadId) {
        await handleNewChat();
      }
      if (activeThreadId) {
        inputTextarea.value = chip.dataset.query;
        handleSend();
      }
    });
  });

  inputTextarea.addEventListener('keydown', (e) => {
    // Ctrl+Enter or Cmd+Enter to send
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  // Auto-grow textarea
  inputTextarea.addEventListener('input', () => {
    inputTextarea.style.height = 'auto';
    inputTextarea.style.height = Math.min(inputTextarea.scrollHeight, 140) + 'px';
  });

  loadThreadList();
});

// ---------------------------------------------------------------------------
// Sidebar toggle (mobile)
// ---------------------------------------------------------------------------

function openSidebar() {
  sidebarEl?.classList.add('open');
  sidebarBackdropEl?.classList.add('open');
  sidebarToggleBtn?.setAttribute('aria-expanded', 'true');
}

function closeSidebar() {
  sidebarEl?.classList.remove('open');
  sidebarBackdropEl?.classList.remove('open');
  sidebarToggleBtn?.setAttribute('aria-expanded', 'false');
}

// ---------------------------------------------------------------------------
// Thread list
// ---------------------------------------------------------------------------

async function loadThreadList() {
  hideSidebarError();
  currentOffset = 0;
  threads = [];
  try {
    const data = await getThreads({ limit: PAGE_SIZE, offset: 0 });
    threads = data.threads || [];
    totalThreads = data.total ?? threads.length;
    hasMore = data.has_more ?? false;
    currentOffset = threads.length;
    renderThreadList();
  } catch (err) {
    showSidebarError('Could not load chats. ' + (err.message || ''));
  }
}

async function loadMoreThreads() {
  if (isLoadingMore || !hasMore) return;
  isLoadingMore = true;
  loadMoreBtn.disabled = true;
  loadMoreBtn.textContent = 'Loading…';

  try {
    const data = await getThreads({ limit: PAGE_SIZE, offset: currentOffset });
    const newThreads = data.threads || [];
    threads = [...threads, ...newThreads];
    totalThreads = data.total ?? threads.length;
    hasMore = data.has_more ?? false;
    currentOffset += newThreads.length;

    // Append only the new items to avoid full re-render
    for (const thread of newThreads) {
      chatListEl.appendChild(buildThreadItem(thread));
    }
    updateLoadMoreVisibility();
    if (activeThreadId) highlightThread(activeThreadId);
  } catch (err) {
    showSidebarError('Could not load more chats. ' + (err.message || ''));
  } finally {
    isLoadingMore = false;
    loadMoreBtn.disabled = false;
    loadMoreBtn.textContent = 'Load more chats';
  }
}

function updateLoadMoreVisibility() {
  if (loadMoreBtn) {
    loadMoreBtn.style.display = hasMore ? 'block' : 'none';
  }
}

function renderThreadList() {
  chatListEl.innerHTML = '';

  if (threads.length === 0) {
    sidebarEmptyEl.style.display = 'block';
    updateLoadMoreVisibility();
    return;
  }

  sidebarEmptyEl.style.display = 'none';

  for (const thread of threads) {
    const item = buildThreadItem(thread);
    chatListEl.appendChild(item);
  }

  updateLoadMoreVisibility();

  // Re-highlight active thread if still in list
  if (activeThreadId) {
    highlightThread(activeThreadId);
  }
}

function buildThreadItem(thread) {
  const item = document.createElement('div');
  item.className = 'chat-item';
  item.dataset.threadId = thread.thread_id;
  item.setAttribute('role', 'button');
  item.setAttribute('tabindex', '0');
  item.setAttribute('aria-label', `Chat: ${thread.thread_name || 'Untitled'}`);

  const name = document.createElement('span');
  name.className = 'chat-item-name';
  name.textContent = thread.thread_name || 'New Chat';

  const meta = document.createElement('span');
  meta.className = 'chat-item-meta';
  meta.textContent = thread.message_count != null ? `${thread.message_count} msg` : '';

  const delBtn = document.createElement('button');
  delBtn.className = 'btn-icon';
  delBtn.innerHTML = '🗑';
  delBtn.setAttribute('aria-label', `Delete chat: ${thread.thread_name || 'Untitled'}`);
  delBtn.setAttribute('title', 'Delete chat');
  delBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    openDeleteDialog(thread);
  });

  item.appendChild(name);
  item.appendChild(meta);
  item.appendChild(delBtn);

  item.addEventListener('click', () => selectThread(thread.thread_id));
  item.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      selectThread(thread.thread_id);
    }
  });

  return item;
}

function highlightThread(threadId) {
  document.querySelectorAll('.chat-item').forEach((el) => {
    el.classList.toggle('active', el.dataset.threadId === threadId);
  });
}

// ---------------------------------------------------------------------------
// Thread selection
// ---------------------------------------------------------------------------

async function selectThread(threadId) {
  // Abort any ongoing stream before switching
  if (isStreaming && abortStream) {
    abortStream();
    abortStream = null;
    isStreaming = false;
  }

  activeThreadId = threadId;
  highlightThread(threadId);
  closeSidebar();

  // Show messages area, hide welcome
  chatWelcomeEl.style.display = 'none';
  messagesAreaEl.style.display = 'flex';
  hideHistoryError();

  clearMessages();
  showLoadingDots(); // temporary indicator

  try {
    const data = await getMessages(threadId);
    hideLoadingDots();
    renderHistory(data.messages || []);
  } catch (err) {
    // Req 8.11 — keep current chat displayed, show error
    hideLoadingDots();
    showHistoryError('Could not load message history. ' + (err.message || ''));
  }

  setInputEnabled(true);
}

function renderHistory(messages) {
  clearMessages();
  for (const msg of messages) {
    appendMessage(msg.role === 'user' ? 'user' : 'assistant', msg.content);
  }
  scrollToBottom();
}

// ---------------------------------------------------------------------------
// New Chat
// ---------------------------------------------------------------------------

async function handleNewChat() {
  // Abort ongoing stream
  if (isStreaming && abortStream) {
    abortStream();
    abortStream = null;
    isStreaming = false;
  }

  try {
    const thread = await createThread();
    // Prepend to local list and re-render
    threads.unshift({
      thread_id:     thread.thread_id,
      thread_name:   thread.thread_name || 'New Chat',
      message_count: 0,
      updated_at:    new Date().toISOString(),
    });
    renderThreadList();
    await selectThread(thread.thread_id);
  } catch (err) {
    showSidebarError('Could not create new chat. ' + (err.message || ''));
  }
}

// ---------------------------------------------------------------------------
// Delete dialog
// ---------------------------------------------------------------------------

let _pendingDeleteThread = null;

function openDeleteDialog(thread) {
  _pendingDeleteThread = thread;
  deleteDialogMsg.textContent = thread.thread_name || 'this chat';
  deleteDialog.classList.add('open');
  deleteDialog.setAttribute('aria-hidden', 'false');
  deleteConfirmBtn.focus();

  deleteConfirmBtn.onclick = () => confirmDelete(thread);
}

function closeDeleteDialog() {
  deleteDialog.classList.remove('open');
  deleteDialog.setAttribute('aria-hidden', 'true');
  _pendingDeleteThread = null;
}

async function confirmDelete(thread) {
  closeDeleteDialog();

  try {
    await deleteThread(thread.thread_id);

    // Req 8.6 — remove from sidebar on confirmed deletion
    threads = threads.filter((t) => t.thread_id !== thread.thread_id);
    renderThreadList();

    // If the deleted chat was active, go back to welcome screen
    if (activeThreadId === thread.thread_id) {
      activeThreadId = null;
      clearMessages();
      messagesAreaEl.style.display = 'none';
      chatWelcomeEl.style.display = '';
      setInputEnabled(false);
    }
  } catch (err) {
    // Req 8.9 — keep chat in sidebar, show error
    showSidebarError('Could not delete chat. ' + (err.message || ''));
  }
}

// ---------------------------------------------------------------------------
// Message send / streaming
// ---------------------------------------------------------------------------

async function handleSend() {
  if (!activeThreadId) return;
  if (isStreaming) return;

  const text = inputTextarea.value.trim();
  if (!text) return;

  inputTextarea.value = '';
  inputTextarea.style.height = 'auto';
  setInputEnabled(false);

  // Render user message immediately
  appendMessage('user', text);
  scrollToBottom();

  // Show loading dots while waiting for first token
  const loadingEl = showLoadingDots();

  // Create the AI message bubble that will be filled by streaming
  let aiBubble = null;
  let firstToken = true;

  isStreaming = true;

  abortStream = streamMessage(
    text,
    activeThreadId,
    // onToken — Req 8.5
    (token) => {
      if (firstToken) {
        firstToken = false;
        if (loadingEl) loadingEl.remove();
        aiBubble = appendStreamingMessage();
        scrollToBottom();
      }
      appendToken(aiBubble, token);
      scrollToBottom();
    },
    // onDone
    () => {
      isStreaming = false;
      abortStream = null;
      if (loadingEl) loadingEl.remove();
      if (aiBubble) finalizeStreamingMessage(aiBubble);
      setInputEnabled(true);
      inputTextarea.focus();
      scrollToBottom();

      // Refresh thread list to update message count / name
      loadThreadList();
    },
    // onError — Req 8.10 explicit error only
    (err) => {
      isStreaming = false;
      abortStream = null;
      if (loadingEl) loadingEl.remove();
      if (aiBubble) finalizeStreamingMessage(aiBubble);
      appendErrorMessage('Response failed: ' + (err.message || 'Unknown error'));
      setInputEnabled(true);
      inputTextarea.focus();
      scrollToBottom();
    }
  );
}

// ---------------------------------------------------------------------------
// Message DOM helpers
// ---------------------------------------------------------------------------

function clearMessages() {
  messagesAreaEl.innerHTML = '';
}

function appendMessage(role, content) {
  const msg = document.createElement('div');
  msg.className = `message ${role}`;

  const label = document.createElement('div');
  label.className = 'message-label';
  label.textContent = role === 'user' ? 'You' : 'Kisan AI';

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';
  bubble.textContent = content;

  msg.appendChild(label);
  msg.appendChild(bubble);
  messagesAreaEl.appendChild(msg);
  return msg;
}

/** Creates an empty AI message bubble that tokens will be appended to. */
function appendStreamingMessage() {
  const msg = document.createElement('div');
  msg.className = 'message assistant';

  const label = document.createElement('div');
  label.className = 'message-label';
  label.textContent = 'Kisan AI';

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble streaming-cursor';
  bubble.textContent = '';

  msg.appendChild(label);
  msg.appendChild(bubble);
  messagesAreaEl.appendChild(msg);
  return bubble; // return bubble so caller can append tokens to it
}

/** Append a token chunk to a streaming bubble. */
function appendToken(bubbleEl, token) {
  bubbleEl.textContent += token;
}

/** Remove the streaming cursor class once stream is done. */
function finalizeStreamingMessage(bubbleEl) {
  bubbleEl.classList.remove('streaming-cursor');
}

/** Insert a red error chip in the message area. */
function appendErrorMessage(text) {
  const el = document.createElement('div');
  el.className = 'message-error';
  el.setAttribute('role', 'alert');
  el.textContent = text;
  messagesAreaEl.appendChild(el);
}

/** Insert and return the loading dots element. */
function showLoadingDots() {
  const wrapper = document.createElement('div');
  wrapper.className = 'message assistant';
  wrapper.id = 'loading-dots-wrapper';

  const label = document.createElement('div');
  label.className = 'message-label';
  label.textContent = 'Kisan AI';

  const dots = document.createElement('div');
  dots.className = 'loading-dots';
  dots.setAttribute('aria-label', 'Loading response');
  dots.setAttribute('role', 'status');
  dots.innerHTML = '<span></span><span></span><span></span>';

  wrapper.appendChild(label);
  wrapper.appendChild(dots);
  messagesAreaEl.appendChild(wrapper);
  scrollToBottom();
  return wrapper;
}

function hideLoadingDots() {
  document.getElementById('loading-dots-wrapper')?.remove();
}

function scrollToBottom() {
  messagesAreaEl.scrollTop = messagesAreaEl.scrollHeight;
}

// ---------------------------------------------------------------------------
// Error banners
// ---------------------------------------------------------------------------

function showSidebarError(msg) {
  sidebarErrorEl.textContent = msg;
  sidebarErrorEl.style.display = 'block';
  sidebarErrorEl.setAttribute('role', 'alert');
}

function hideSidebarError() {
  sidebarErrorEl.style.display = 'none';
  sidebarErrorEl.textContent = '';
}

function showHistoryError(msg) {
  historyErrorEl.textContent = msg;
  historyErrorEl.style.display = 'block';
  historyErrorEl.setAttribute('role', 'alert');
}

function hideHistoryError() {
  historyErrorEl.style.display = 'none';
  historyErrorEl.textContent = '';
}

// ---------------------------------------------------------------------------
// Input state
// ---------------------------------------------------------------------------

function setInputEnabled(enabled) {
  inputTextarea.disabled = !enabled;
  sendBtn.disabled = !enabled;
  if (!enabled) {
    sendBtn.setAttribute('aria-disabled', 'true');
  } else {
    sendBtn.removeAttribute('aria-disabled');
  }
}

// ---------------------------------------------------------------------------
// Theme (dark / light mode)
// ---------------------------------------------------------------------------

function initTheme() {
  const saved = localStorage.getItem('kisan-theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const theme = saved || (prefersDark ? 'dark' : 'light');
  applyTheme(theme);
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  if (themeToggleBtn) {
    themeToggleBtn.textContent = theme === 'dark' ? '☀️' : '🌙';
    themeToggleBtn.setAttribute('aria-label', theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
  }
  localStorage.setItem('kisan-theme', theme);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme') || 'light';
  applyTheme(current === 'dark' ? 'light' : 'dark');
}

// ---------------------------------------------------------------------------
// Logout
// ---------------------------------------------------------------------------

async function handleLogout() {
  await logout();
  window.location.href = '/login.html';
}
