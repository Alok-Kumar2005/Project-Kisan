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