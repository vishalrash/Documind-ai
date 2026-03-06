from retrieval import search

question = "What are transformers?"

results = search(question)

for r in results:
    print(r)