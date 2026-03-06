def generate_answer(results):

    # If no results
    if not results:
        return {
            "answer": "I could not find this in the provided documents. Can you share the relevant document?",
            "sources": [],
            "confidence": "low"
        }

    # Combine top snippets
    combined_text = ""

    for r in results:
        combined_text += r["snippet"] + " "

    # Create short answer
    answer = combined_text[:300]

    # Determine confidence
    best_score = results[0]["score"]

    if best_score < 0.3:
        confidence = "high"
    elif best_score < 0.7:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "answer": answer,
        "sources": results,
        "confidence": confidence
    }