from langchain_core.messages import SystemMessage, AIMessage

from app.config import get_llm
from app.src.state import AgentState
from app.src.utils.extract_content import extract_text

SYSTEM_PROMPT = (
    "You are an assistant with access to tools.\n\n"
    "Rules:\n"
    "- For questions about depression or hypertension (symptoms, causes, "
    "diagnosis, treatment), call search_health_docs.\n"
    "- For any other request, check the descriptions of the other available "
    "tools. If one fits, call it.\n"
    "- If no tool fits (general knowledge, casual chat), do NOT call a tool. "
    "Reply briefly and say you can only help with depression and hypertension "
    "information and the connected tools.\n"
    "- When a tool result arrives, answer using only that result.\n"
    "- If a tool returns an error or finds nothing, say so plainly. Do not guess."
)


def make_decide_node(tools):
    model_with_tools = get_llm(temperature=0).bind_tools(tools)

    def generate_query_or_response(state: AgentState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = model_with_tools.invoke(messages)

        if response.tool_calls:
            return {"messages": [response]}

        cleaned_text = extract_text(response.content).strip()
        return {"messages": [AIMessage(content=cleaned_text)]}

    return generate_query_or_response