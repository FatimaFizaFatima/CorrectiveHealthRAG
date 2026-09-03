from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from app.src.state import AgentState
from app.src.retriever import build_retriever_tool
from app.src.node.decide import generate_query_or_response
from app.src.node.grade import grade_documents
from app.src.node.rewrite import rewrite_question
from app.src.node.generate import generate_answer

retriever_tool = build_retriever_tool()
MAX_RETRIES = 2

workflow = StateGraph(AgentState)

workflow.add_node("generate_query_or_response", generate_query_or_response)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("generate_answer", generate_answer)

workflow.add_edge(START, "generate_query_or_response")

workflow.add_conditional_edges(
    "generate_query_or_response",
    tools_condition,
    {"tools": "retrieve", END: END},
)

def route_after_grading(state: AgentState) -> str:
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "generate_answer"   
    return grade_documents(state)   

workflow.add_conditional_edges(
    "retrieve",
    route_after_grading,
    {"generate_answer": "generate_answer", "rewrite_question": "rewrite_question"},
)

workflow.add_edge("rewrite_question", "generate_query_or_response")
workflow.add_edge("generate_answer", END)

graph = workflow.compile()

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    test_queries = [
        "what are the symptoms of depression",     
        "who invented alternating current",          
        "tell me about high blood pressure",          
    ]

    for q in test_queries:
        print(f"\n{'='*60}\nQUERY: {q}\n{'='*60}")
        result = graph.invoke({
            "messages": [HumanMessage(content=q)],
            "retry_count": 0,
        })
        final_answer = result["messages"][-1]
        print("FINAL ANSWER:", final_answer.content)