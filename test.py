# Use a pipeline as a high-level helper
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import pipeline

pipe = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")