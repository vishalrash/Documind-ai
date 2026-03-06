from langchain_community.llms import Ollama

llm = Ollama(model="tinyllama:chat")

def generate_llm_answer(context, question):

    prompt = f"""
You are a document assistant.

You MUST follow these rules:
1. Answer ONLY using the provided context.
2. Do NOT use outside knowledge.
3. If the answer is not present in the context, respond exactly with:
   "No relevant information found in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response