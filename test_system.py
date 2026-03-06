from retrieval import search
from answer_generator import generate_answer

# Ask a question
question = "Who won the FIFA World Cup 2022?"

# Retrieve relevant chunks
results = search(question)

# Generate answer
response = generate_answer(results)

# Print output
print("Answer:\n", response["answer"])
print("\nConfidence:", response["confidence"])

print("\nSources:")
for s in response["sources"]:
    print("Document:", s["document"])
    print("Snippet:", s["snippet"])
    print("Score:", s["score"])
    print("-----")