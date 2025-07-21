gemini_model_name = "gemini-1.5-flash"
gemini_model_kwargs = {
    "temperature": 0.2,
    "top_p": 0.95,
    "max_output_tokens": 512,
    "top_k": 40,
}

groq_model_name = "gemma2-9b-it"
groq_model_kwargs = {
    "temperature": 0.2,
    # "top_p": 0.95,
    "max_tokens": 512
}

## tavily config
max_result = 2

## weather config
DEFAULT_FORECAST_COUNT = 40
DEFAULT_DAYS = 5

## image_model
image_model="black-forest-labs/FLUX.1-kontext-dev"
image_width = 256
image_height = 192
steps = 38
image_url="https://imgs.search.brave.com/sYprGgU1Zl3qYed4XU19fDsWpBVFDv0RvNKEe-FAdaQ/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly90My5m/dGNkbi5uZXQvanBn/LzAzLzk1LzQ4Lzg2/LzM2MF9GXzM5NTQ4/ODY4M19DZnhwYlph/M2hlMXlnVFpYSGRT/cEhVdlp5cUw0c3Yy/di5qcGc"


### memory config
top_collection_search = 3
top_database_search = 10