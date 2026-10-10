"""Measure retrieval hit-rate and answer keyword accuracy on eval.json."""
import json

from rag import answer, retrieve

cases = json.load(open("eval.json", encoding="utf-8"))
hit = acc = 0
for c in cases:
    docs, _ = retrieve(c["question"])
    hit += any(c["keyword"].lower() in d.lower() for d in docs)
    text, _ = answer(c["question"])
    acc += c["keyword"].lower() in text.lower()
n = len(cases)
print(f"Retrieval hit-rate: {hit}/{n} = {hit/n:.0%}")
print(f"Answer accuracy:    {acc}/{n} = {acc/n:.0%}")
