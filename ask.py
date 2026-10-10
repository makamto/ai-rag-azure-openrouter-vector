import sys

from rag import answer

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or input("Question: ")
    text, metas = answer(q)
    print(text)
    print("\nSources:", sorted({f"{m['source']}#{m['chunk']}" for m in metas}))
