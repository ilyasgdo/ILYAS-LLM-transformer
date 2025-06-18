import random

# Charge toutes les lignes
with open("data/cleaned.txt", "r", encoding="utf-8") as f:
    lines = [l for l in f if l.strip()]

random.shuffle(lines)
n = len(lines)
train, valid, test = (
    lines[: int(0.8*n)],
    lines[int(0.8*n) : int(0.9*n)],
    lines[int(0.9*n) :],
)

for name, chunk in [("train", train), ("valid", valid), ("test", test)]:
    with open(f"data/{name}.txt", "w", encoding="utf-8") as f:
        f.writelines(chunk)
