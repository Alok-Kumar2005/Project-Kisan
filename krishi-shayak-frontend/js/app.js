// Global variables
let currentUser = null;
let currentThreadId = null;
let userThreads = [];
let isLoading = false;

// API Configuration
const API_BASE_URL = 'http://localhost:8000/api/v1';

// DOM Elements
const heroSection = document.getElementById('hero-section');
const chatContainer = document.getElementById('chat-container');
const loginModal = document.getElementById('login-modal');
const registerModal = document.getElementById('register-modal');
const chatMessages = document.getElementById('chat-messages');
const messageInput = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const threadsList = document.getElementById('threads-list');
const chatSidebar = document.getElementById('chat-sidebar');

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

// Initialize application
function initializeApp() {
    // Check if user is already logged in
    const token = localStorage.getItem('access_token');
    const userData = localStorage.getItem('user_data');
    
    if (token && userData) {
        currentUser = JSON.parse(userData);
        showChatInterface();
        loadUserThreads();
    }
    
    // Add event listeners
    setupEventListeners();
}

// Setup event listeners
function setupEventListeners() {
    // Close modal when clicking outside
    window.addEventListener('click', function(event) {
        if (event.target === loginModal) {
            closeModal('login-modal');
        }
        if (event.target === registerModal) {
            closeModal('register-modal');
        }
    });
    
    // Handle Enter key in message input
    messageInput?.addEventListener('keypress', function(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendMessage(event);
        }
    });
}

// API call utility function
async function apiCall(endpoint, options = {}) {
    const token = localStorage.getItem('access_token');
    
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
            ...(token && { 'Authorization': `Bearer ${token}` })
        }
    };
    
    const finalOptions = {
        ...defaultOptions,
        ...options,
        headers: {
            ...defaultOptions.headers,
            ...options.headers
        }
    };
    
    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, finalOptions);
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'An error occurred' }));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API call error:', error);
        throw error;
    }
}

// Show toast notification
function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    const toastMessage = document.getElementById('toast-message');
    
    toastMessage.textContent = message;
    toast.className = `toast ${type}`;
    toast.classList.add('show');
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// Modal functions
function showLogin() {
    loginModal.style.display = 'block';
    document.body.style.overflow = 'hidden';
}

function showRegister() {
    registerModal.style.display = 'block';
    document.body.style.overflow = 'hidden';
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    modal.style.display = 'none';
    document.body.style.overflow = 'auto';
}

function switchToRegister() {
    closeModal('login-modal');
    showRegister();
}

function switchToLogin() {
    closeModal('register-modal');
    showLogin();
}

// Handle login form submission
async function handleLogin(event) {
    event.preventDefault();
    
    const loginBtn = document.getElementById('login-btn');
    const btnText = loginBtn.querySelector('.btn-text');
    const btnLoader = loginBtn.querySelector('.btn-loader');
    
    if (isLoading) return;
    
    isLoading = true;
    loginBtn.classList.add('loading');
    loginBtn.disabled = true;
    
    try {
        const formData = new FormData(event.target);
        const loginData = {
            unique_name: formData.get('unique_name'),
            password: formData.get('password')
        };
        
        const response = await apiCall('/auth/login', {
            method: 'POST',
            body: JSON.stringify(loginData)
        });
        
        // Store tokens and user data
        localStorage.setItem('access_token', response.access_token);
        localStorage.setItem('refresh_token', response.refresh_token);
        localStorage.setItem('user_data', JSON.stringify(response.user));
        
        currentUser = response.user;
        
        showToast('Login successful!', 'success');
        closeModal('login-modal');
        showChatInterface();
        loadUserThreads();
        
    } catch (error) {
        showToast(error.message, 'error');
    } finally {
        isLoading = false;
        loginBtn.classList.remove('loading');
        loginBtn.disabled = false;
    }
}

// Handle register form submission
async function handleRegister(event) {
    event.preventDefault();
    
    const registerBtn = document.getElementById('register-btn');
    
    if (isLoading) return;
    
    isLoading = true;
    registerBtn.classList.add('loading');
    registerBtn.disabled = true;
    
    try {
        const formData = new FormData(event.target);
        const registerData = {
            name: formData.get('name'),
            unique_name: formData.get('unique_name'),
            mobile_number: formData.get('mobile_number'),
            email: formData.get('email') || null,
            district: formData.get('district'),
            state: formData.get('state'),
            country: formData.get('country'),
            date_of_birth: formData.get('date_of_birth') || null,
            password: formData.get('password')
        };
        
        // Remove empty string values
        Object.keys(registerData).forEach(key => {
            if (registerData[key] === '') {
                registerData[key] = null;
            }
        });
        
        const response = await apiCall('/auth/register', {
            method: 'POST',
            body: JSON.stringify(registerData)
        });
        
        // Store tokens and user data
        localStorage.setItem('access_token', response.access_token);
        localStorage.setItem('refresh_token', response.refresh_token);
        localStorage.setItem('user_data', JSON.stringify(response.user));
        
        currentUser = response.user;
        
        showToast('Registration successful! Welcome to Krishi Shayak!', 'success');
        closeModal('register-modal');
        showChatInterface();
        loadUserThreads();
        
    } catch (error) {
        showToast(error.message, 'error');
    } finally {
        isLoading = false;
        registerBtn.classList.remove('loading');
        registerBtn.disabled = false;
    }
}

// Show chat interface
function showChatInterface() {
    heroSection.style.display = 'none';
    chatContainer.style.display = 'flex';
    
    // Update user info in sidebar
    document.getElementById('user-name').textContent = currentUser.name;
    document.getElementById('user-location').textContent = 
        `${currentUser.district}, ${currentUser.state}`;
    
    // Update navbar
    const navMenu = document.getElementById('nav-menu');
    navMenu.innerHTML = `
        <button class="nav-btn" onclick="logout()">Logout</button>
    `;
}

// Load user threads
async function loadUserThreads() {
    try {
        const response = await apiCall('/chat/threads');
        userThreads = response.threads || [];
        renderThreads();
    } catch (error) {
        console.error('Error loading threads:', error);
        showToast('Error loading chat history', 'error');
    }
}

// Render threads in sidebar
function renderThreads() {
    threadsList.innerHTML = '';
    
    if (userThreads.length === 0) {
        threadsList.innerHTML = `
            <div style="text-align: center; padding: 2rem; color: #6b7280;">
                <p>No conversations yet.</p>
                <p style="font-size: 0.9rem; margin-top: 0.5rem;">Start a new chat to get farming advice!</p>
            </div>
        `;
        return;
    }
    
    userThreads.forEach(thread => {
        const threadElement = document.createElement('div');
        threadElement.className = `thread-item ${thread.thread_id === currentThreadId ? 'active' : ''}`;
        threadElement.onclick = () => selectThread(thread.thread_id);
        
        threadElement.innerHTML = `
            <div class="thread-name">${thread.thread_name || 'New Conversation'}</div>
            <div class="thread-preview">${thread.message_count} messages</div>
        `;
        
        threadsList.appendChild(threadElement);
    });
}

// Create new thread
async function createNewThread() {
    try {
        const response = await apiCall('/chat/thread/create', {
            method: 'POST',
            body: JSON.stringify({
                thread_name: 'New Conversation'
            })
        });
        
        currentThreadId = response.thread_id;
        await loadUserThreads();
        clearChatMessages();
        showWelcomeMessage();
        
        // Show delete button
        document.getElementById('delete-thread-btn').style.display = 'block';
        
        showToast('New conversation started!', 'success');
        
    } catch (error) {
        console.error('Error creating thread:', error);
        showToast('Error creating new conversation', 'error');
    }
}

// Select thread
async function selectThread(threadId) {
    currentThreadId = threadId;
    
    // Update active thread in UI
    document.querySelectorAll('.thread-item').forEach(item => {
        item.classList.remove('active');
    });
    
    event.target.closest('.thread-item').classList.add('active');
    
    // Load thread messages
    await loadThreadMessages(threadId);
    
    // Show delete button
    document.getElementById('delete-thread-btn').style.display = 'block';
}

// Load thread messages
async function loadThreadMessages(threadId) {
    try {
        const response = await apiCall(`/chat/thread/${threadId}/messages`);
        
        clearChatMessages();
        
        if (response.messages && response.messages.length > 0) {
            response.messages.forEach(message => {
                addMessageToChat(message.content, message.role, message.timestamp);
            });
        } else {
            showWelcomeMessage();
        }
        
        // Scroll to bottom
        scrollToBottom();
        
    } catch (error) {
        console.error('Error loading thread messages:', error);
        showToast('Error loading conversation', 'error');
    }
}

// Delete current thread
async function deleteCurrentThread() {
    if (!currentThreadId) return;
    
    if (!confirm('Are you sure you want to delete this conversation?')) {
        return;
    }
    
    try {
        await apiCall(`/chat/thread/${currentThreadId}`, {
            method: 'DELETE'
        });
        
        // Remove from local threads list
        userThreads = userThreads.filter(thread => thread.thread_id !== currentThreadId);
        
        // Clear current thread
        currentThreadId = null;
        clearChatMessages();
        showWelcomeMessage();
        
        // Re-render threads
        renderThreads();
        
        // Hide delete button
        document.getElementById('delete-thread-btn').style.display = 'none';
        
        showToast('Conversation deleted', 'success');
        
    } catch (error) {
        console.error('Error deleting thread:', error);
        showToast('Error deleting conversation', 'error');
    }
}

// Clear chat messages
function clearChatMessages() {
    chatMessages.innerHTML = '';
}

// Show welcome message
function showWelcomeMessage() {
    chatMessages.innerHTML = `
        <div class="welcome-message fade-in">
            <div class="logo-circle large">
                <div class="logo-top"></div>
                <div class="logo-bottom"></div>
                <div class="wheat-icon">🌾</div>
            </div>
            <h3>Welcome to Krishi Shayak!</h3>
            <p>I'm here to help you with agricultural guidance, crop management, and farming solutions.</p>
            <div class="example-questions">
                <h4>Try asking:</h4>
                <button class="example-btn" onclick="sendExampleMessage('What crops grow best in my region?')">
                    What crops grow best in my region?
                </button>
                <button class="example-btn" onclick="sendExampleMessage('How do I prevent crop diseases?')">
                    How do I prevent crop diseases?
                </button>
                <button class="example-btn" onclick="sendExampleMessage('What are current market prices?')">
                    What are current market prices?
                </button>
            </div>
        </div>
    `;
}

// Send example message
function sendExampleMessage(message) {
    messageInput.value = message;
    const event = { preventDefault: () => {} };
    sendMessage(event);
}

// Send message
async function sendMessage(event) {
    event.preventDefault();
    
    const message = messageInput.value.trim();
    if (!message || isLoading) return;
    
    // If no thread is selected, create a new one
    if (!currentThreadId) {
        await createNewThread();
    }
    
    // Clear input and disable send button
    messageInput.value = '';
    isLoading = true;
    sendBtn.disabled = true;
    
    // Add user message to chat
    addMessageToChat(message, 'user');
    
    // Add typing indicator
    const typingId = addTypingIndicator();
    
    try {
        // Send message to API
        const response = await apiCall('/chat/message', {
            method: 'POST',
            body: JSON.stringify({
                query: message,
                thread_id: currentThreadId,
                workflow: 'GeneralNode'
            })
        });
        
        // Remove typing indicator
        removeTypingIndicator(typingId);
        
        // Add assistant response
        if (response.content) {
            addMessageToChat(response.content, 'assistant');
        } else {
            addMessageToChat('Sorry, I couldn\'t process your request. Please try again.', 'assistant');
        }
        
        // Reload threads to update message count
        await loadUserThreads();
        
    } catch (error) {
        removeTypingIndicator(typingId);
        console.error('Error sending message:', error);
        addMessageToChat('Sorry, there was an error processing your message. Please try again.', 'assistant');
        showToast('Error sending message', 'error');
    } finally {
        isLoading = false;
        sendBtn.disabled = false;
        messageInput.focus();
    }
}

// Add message to chat
function addMessageToChat(content, role, timestamp = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role} slide-up`;
    
    const avatar = role === 'user' ? '👤' : '🌾';
    const time = timestamp ? new Date(timestamp).toLocaleTimeString() : new Date().toLocaleTimeString();
    
    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            ${content}
            <div class="message-time">${time}</div>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// Add typing indicator
function addTypingIndicator() {
    const typingDiv = document.createElement('div');
    const typingId = 'typing-' + Date.now();
    typingDiv.id = typingId;
    typingDiv.className = 'message assistant';
    
    typingDiv.innerHTML = `
        <div class="message-avatar">🌾</div>
        <div class="message-content">
            <div class="typing-indicator">
                <span>Krishi Shayak is thinking</span>
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(typingDiv);
    scrollToBottom();
    
    return typingId;
}

// Remove typing indicator
function removeTypingIndicator(typingId) {
    const typingElement = document.getElementById(typingId);
    if (typingElement) {
        typingElement.remove();
    }
}

// Scroll to bottom of chat
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Mobile functions
function toggleSidebar() {
    chatSidebar.classList.toggle('open');
}

function toggleMobileMenu() {
    const navMenu = document.getElementById('nav-menu');
    navMenu.classList.toggle('active');
}

// Logout function
function logout() {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('user_data');
    
    currentUser = null;
    currentThreadId = null;
    userThreads = [];
    
    chatContainer.style.display = 'none';
    heroSection.style.display = 'block';
    
    // Reset navbar
    const navMenu = document.getElementById('nav-menu');
    navMenu.innerHTML = `
        <button class="nav-btn" onclick="showLogin()">Login</button>
        <button class="nav-btn primary" onclick="showRegister()">Register</button>
    `;
    
    showToast('Logged out successfully', 'success');
}

// Handle token refresh (optional - for automatic token refresh)
async function refreshToken() {
    try {
        const refreshToken = localStorage.getItem('refresh_token');
        if (!refreshToken) {
            throw new Error('No refresh token available');
        }
        
        const response = await apiCall('/auth/refresh', {
            method: 'POST',
            body: JSON.stringify({
                refresh_token: refreshToken
            })
        });
        
        localStorage.setItem('access_token', response.access_token);
        localStorage.setItem('refresh_token', response.refresh_token);
        
        return response.access_token;
        
    } catch (error) {
        console.error('Error refreshing token:', error);
        // If refresh fails, logout user
        logout();
        throw error;
    }
}

// Error handling for expired tokens
window.addEventListener('unhandledrejection', function(event) {
    if (event.reason.message && event.reason.message.includes('401')) {
        console.log('Token expired, attempting refresh...');
        refreshToken().catch(() => {
            showToast('Session expired. Please login again.', 'error');
        });
    }
});

// Service Worker Registration (optional - for offline functionality)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/sw.js')
            .then(function(registration) {
                console.log('ServiceWorker registration successful');
            })
            .catch(function(err) {
                console.log('ServiceWorker registration failed');
            });
    });
}