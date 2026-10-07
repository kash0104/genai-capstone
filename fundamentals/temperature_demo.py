"""Watch temperature reshape a next-token distribution.

Run: uv run python -m fundamentals.temperature_demo
"""
from fundamentals.maths import softmax

VOCAB = ["Paris", "London", "Rome", "pizza", "the"]
LOGITS = [6.0, 3.5, 3.0, 1.0, 0.5]  # made-up scores for "The capital of France is ..."

for t in (0.2, 0.7, 1.0, 2.0):
    print(f"\ntemperature = {t}")
    for word, p in zip(VOCAB, softmax(LOGITS, temperature=t)):
        print(f"  {word:<7} {p:6.1%} {'█' * round(p * 40)}")
