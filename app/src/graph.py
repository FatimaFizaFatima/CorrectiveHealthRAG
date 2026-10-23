from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from app.src.state import AgentState
from app.src.retriever import build_retriever_tool
from app.src.node.approval import approve_sensitive_tools, has_sensitive_call
from app.src.node.decide import make_decide_node
from app.src.node.grade import grade_documents
from app.src.node.rewrite import rewrite_question
from app.src.node.generate import generate_answer

RAG_TOOL_NAME = "search_health_docs"
MAX_RETRIES = 2

retriever_tool = build_retriever_tool()


def route_after_decide(state: AgentState) -> str:
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None)
    if not tool_calls:
        return END
    if has_sensitive_call(tool_calls):
        return "approval"
    return "tools"


def route_after_tools(state: AgentState) -> str:
    ai_message = next(
        m for m in reversed(state["messages"]) if m.type == "ai" and m.tool_calls
    )
    called = {call["name"] for call in ai_message.tool_calls}

    if called != {RAG_TOOL_NAME}:
        return "generate_query_or_response"

    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "generate_answer"
    return grade_documents(state)


def build_graph(mcp_tools=None, checkpointer=None):
    all_tools = [retriever_tool] + list(mcp_tools or [])

    workflow = StateGraph(AgentState)

    workflow.add_node("generate_query_or_response", make_decide_node(all_tools))
    workflow.add_node("approval", approve_sensitive_tools)
    workflow.add_node("tools", ToolNode(all_tools, handle_tool_errors=True))
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate_answer", generate_answer)

    workflow.add_edge(START, "generate_query_or_response")

    workflow.add_conditional_edges(
        "generate_query_or_response",
        route_after_decide,
        {"tools": "tools", "approval": "approval", END: END},
    )

    workflow.add_conditional_edges(
        "tools",
        route_after_tools,
        {
            "generate_query_or_response": "generate_query_or_response",
            "generate_answer": "generate_answer",
            "rewrite_question": "rewrite_question",
        },
    )

    workflow.add_edge("rewrite_question", "generate_query_or_response")
    workflow.add_edge("generate_answer", END)

    return workflow.compile(checkpointer=checkpointer)


graph = build_graph()