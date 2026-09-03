from langchain_core.messages import HumanMessage
from app.src.graph import graph

TEST_QUERIES = {
    "relevant": [
        "what are the symptoms of depression",
        "tell me about high blood pressure",
    ],
    "irrelevant": [
        "who invented alternating current",
    ],
    "ambiguous": [
        "what should I do about it",
    ],
}

def run_query(query: str):
    result = graph.invoke({
        "messages": [HumanMessage(content=query)],
        "retry_count": 0,
    })
    return result["messages"][-1]

if __name__ == "__main__":
    for category, queries in TEST_QUERIES.items():
        for q in queries:
            print(f"\n{'='*60}")
            print(f"[{category.upper()}] QUERY: {q}")
            print('='*60)
            final_message = run_query(q)
            print("FINAL ANSWER:", final_message.content)