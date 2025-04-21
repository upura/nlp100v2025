from transformers import pipeline
from pprint import pprint

pipe = pipeline(
    "fill-mask", model="answerdotai/ModernBERT-base", device="cpu", top_k=10
)

input_text = "The movie was full of [MASK]."
results = pipe(input_text)
pprint(results)
