from langchain_core.messages import ToolMessage
from langgraph.types import Command, interrupt

from app.src.state import AgentState

# Tools that change or delete data. They need a human decision before they run.
SENSITIVE_TOOLS = {"delete_document"}


def has_sensitive_call(tool_calls) -> bool:
    return any(call["name"] in SENSITIVE_TOOLS for call in tool_calls)


def approve_sensitive_tools(state: AgentState):
    ai_message = state["messages"][-1]
    pending = [
        {"tool": call["name"], "args": call["args"]}
        for call in ai_message.tool_calls
        if call["name"] in SENSITIVE_TOOLS
    ]

    decision = interrupt({"type": "approval_required", "actions": pending})

    if isinstance(decision, dict) and decision.get("approved") is True:
        return Command(goto="tools")

    rejected = [
        ToolMessage(
            content="The user rejected this action. It was not executed.",
            tool_call_id=call["id"],
            name=call["name"],
            status="error",
        )
        for call in ai_message.tool_calls
    ]
    return Command(goto="generate_query_or_response", update={"messages": rejected})