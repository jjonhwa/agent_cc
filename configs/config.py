import os
from dotenv import load_dotenv

load_dotenv()

# App Names
TEXT_APP_NAME = os.getenv("TEXT_APP_NAME", "Text Labeling")
MULTIMODAL_APP_NAME = os.getenv("MULTIMODAL_APP_NAME", "Multimodal Labeling")
TIME_SERIES_APP_NAME = os.getenv("TIME_SERIES_APP_NAME", "Time Series")
RISK_APP_NAME = os.getenv("RISK_APP_NAME", "Risk")

# Inference Defaults
LABEL_CANDIDATES = os.getenv("LABEL_CANDIDATES", "True,False,Neutral").split(",")
FIXED_FIELDS = os.getenv("FIXED_FIELDS", "판정근거,근거문장,자동라벨").split(",")
HIGHLIGHT_FIELD = os.getenv("HIGHLIGHT_FIELD", "근거문장")
PAGE_SIZE = int(os.getenv("PAGE_SIZE", 1))
IS_EDITABLE = os.getenv("IS_EDITABLE", "True").lower() == "true"
IS_DOWNLOAD = os.getenv("IS_DOWNLOAD", "True").lower() == "true"
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", 32))
MAX_LEN = int(os.getenv("MAX_LEN", 8192))
TEMP = float(os.getenv("TEMPERATURE", 0))
MAX_RETRY = int(os.getenv("MAX_RETRY", 3))

# Default Model Names
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "")
VLLM_API_URL = os.getenv("VLLM_API_URL", "http://localhost:8000/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 100))

# Predefined Model Lists
OPENAI_MODELS = [
    "gpt-5.2",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5",
    "gpt-4.1"
]

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
]
