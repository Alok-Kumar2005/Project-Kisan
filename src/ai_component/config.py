# =============================================================================
# Model selection
# =============================================================================
# Change this one value to switch every node, chain, and memory manager
# between providers.  Supported values: "groq" | "gemini"
default_model = "groq"

# =============================================================================
# Gemini settings
# =============================================================================
gemini_model_name = "gemini-1.5-flash"
gemini_model_kwargs = {
    "temperature": 0.2,
    "top_p": 0.95,
    "max_output_tokens": 512,
    "top_k": 40,
}

# =============================================================================
# Groq settings
# =============================================================================
groq_model_name = "llama-3.3-70b-versatile"
groq_model_kwargs = {
    "temperature": 0.2,
    "max_tokens": 512,
}

# =============================================================================
# Tavily
# =============================================================================
max_result = 2

# =============================================================================
# Weather
# =============================================================================
DEFAULT_FORECAST_COUNT = 40
DEFAULT_DAYS = 5

# =============================================================================
# Image generation (Together API)
# =============================================================================
image_model  = "black-forest-labs/FLUX.1-kontext-dev"
image_width  = 256
image_height = 192
steps        = 38
image_url    = (
    "https://imgs.search.brave.com/sYprGgU1Zl3qYed4XU19fDsWpBVFDv0RvNKEe-FAdaQ"
    "/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly90My5mdGNkbi5uZXQvanBnLzAzLzk1LzQ4Lzg2"
    "LzM2MF9GXzM5NTQ4ODY4M19DZnhwYlphM2hlMXlnVFpYSGRTcEhVdlp5cUw0c3YyNi5qcGc"
)

# =============================================================================
# Memory / vector store
# =============================================================================
top_collection_search = 3
top_database_search   = 10
