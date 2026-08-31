import os

# Keep tests fully offline and deterministic: no network, no API key needed.
os.environ.setdefault("VECTOR_BACKEND", "numpy")
os.environ.setdefault("LLM_PROVIDER", "custom")  # avoids requiring a real key
