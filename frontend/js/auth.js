/**
 * auth.js — Form handlers, token storage, and redirect logic
 *
 * Handles login.html and signup.html form submissions.
 * Requirements: 8.1, 8.2, 8.8
 */

import { login, register, isAuthenticated } from './api.js';

// Apply saved theme immediately so auth pages match user preference
(function initTheme() {
  const saved = localStorage.getItem('kisan-theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const theme = saved || (prefersDark ? 'dark' : 'light');
  document.documentElement.setAttribute('data-theme', theme);
})();

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/** Show an error alert with a given message. */
function showError(alertEl, message) {
  alertEl.textContent = message;
  alertEl.classList.add('visible');
  alertEl.setAttribute('role', 'alert');
}

/** Hide the error alert. */
function hideError(alertEl) {
  alertEl.textContent = '';
  alertEl.classList.remove('visible');
}

/** Set button to loading / disabled state. */
function setLoading(btn, loading) {
  btn.disabled = loading;
  if (loading) {
    btn.dataset.originalText = btn.textContent;
    btn.textContent = 'Please wait…';
  } else {
    btn.textContent = btn.dataset.originalText || btn.textContent;
  }
}

// ---------------------------------------------------------------------------
// Login page (login.html)
// ---------------------------------------------------------------------------

function initLogin() {
  const form = document.getElementById('login-form');
  if (!form) return;

  // Redirect already-authenticated users
  if (isAuthenticated()) {
    window.location.href = '/chat.html';
    return;
  }

  const alertEl = document.getElementById('login-error');
  const submitBtn = document.getElementById('login-btn');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    hideError(alertEl);

    const phoneNumber = form.phone_number.value.trim();
    const password = form.password.value;

    if (!phoneNumber || !password) {
      showError(alertEl, 'Please enter your phone number and password.');
      return;
    }

    setLoading(submitBtn, true);

    try {
      await login(phoneNumber, password);
      // Successful login — redirect to chat
      window.location.href = '/chat.html';
    } catch (err) {
      // Req 8.8 — show the specific reason from the backend
      showError(alertEl, err.message || 'Login failed. Please try again.');
    } finally {
      setLoading(submitBtn, false);
    }
  });
}

// ---------------------------------------------------------------------------
// Signup page (signup.html)
// ---------------------------------------------------------------------------

function initSignup() {
  const form = document.getElementById('signup-form');
  if (!form) return;

  // Redirect already-authenticated users
  if (isAuthenticated()) {
    window.location.href = '/chat.html';
    return;
  }

  const alertEl = document.getElementById('signup-error');
  const submitBtn = document.getElementById('signup-btn');

  // Optional fields toggle
  const toggleBtn = document.getElementById('optional-toggle-btn');
  const optionalSection = document.getElementById('optional-fields');
  if (toggleBtn && optionalSection) {
    toggleBtn.addEventListener('click', () => {
      const isHidden = optionalSection.hidden;
      optionalSection.hidden = !isHidden;
      toggleBtn.textContent = isHidden ? 'Hide optional fields ▲' : 'Show optional fields ▼';
      toggleBtn.setAttribute('aria-expanded', String(isHidden));
    });
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    hideError(alertEl);

    const phoneNumber = form.phone_number.value.trim();
    const uniqueName  = form.unique_name.value.trim();
    const password    = form.password.value;
    const confirmPwd  = form.confirm_password.value;

    // Basic client-side validation
    if (!phoneNumber || !uniqueName || !password) {
      showError(alertEl, 'Phone number, username, and password are required.');
      return;
    }

    if (password !== confirmPwd) {
      showError(alertEl, 'Passwords do not match.');
      return;
    }

    if (password.length < 8) {
      showError(alertEl, 'Password must be at least 8 characters.');
      return;
    }

    // Validate username format — must match backend validator
    if (!/^[a-zA-Z0-9_-]+$/.test(uniqueName)) {
      showError(alertEl, 'Username can only contain letters, numbers, hyphens and underscores. No spaces, @ or special characters.');
      return;
    }

    // Build payload — include optional fields only if provided
    const payload = {
      phone_number: phoneNumber,
      unique_name:  uniqueName,
      password:     password,
    };

    const optionalFields = ['full_name', 'age', 'resident', 'city', 'district', 'state', 'country'];
    for (const field of optionalFields) {
      const el = form.elements[field];
      if (el && el.value.trim()) {
        payload[field] = field === 'age' ? parseInt(el.value.trim(), 10) : el.value.trim();
      }
    }

    // Location fields (lat/lng) if provided
    const latEl = form.elements['latitude'];
    const lngEl = form.elements['longitude'];
    if (latEl && latEl.value.trim() && lngEl && lngEl.value.trim()) {
      payload.latitude  = parseFloat(latEl.value.trim());
      payload.longitude = parseFloat(lngEl.value.trim());
    }

    setLoading(submitBtn, true);

    try {
      await register(payload);
      // Auto-logged in — redirect to chat
      window.location.href = '/chat.html';
    } catch (err) {
      // Req 8.8 — show specific error (e.g. "Phone number already registered")
      showError(alertEl, err.message || 'Registration failed. Please try again.');
    } finally {
      setLoading(submitBtn, false);
    }
  });
}

// ---------------------------------------------------------------------------
// Bootstrap — run the right init depending on which page is loaded
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  initLogin();
  initSignup();
});
