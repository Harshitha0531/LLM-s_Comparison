from transformers import pipeline

# =========================
# LOAD MODELS ONCE
# =========================

model1 = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    framework="pt"
)

model2 = pipeline(
    "text-generation",
    model="microsoft/phi-2",
    framework="pt"
)
