from app.config import get_llm
from app.src.retriever import build_retriever_tool
from app.src.state import AgentState
from app.src.utils.extract_content import extract_text
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

retriever_tool = build_retriever_tool()
model = get_llm(temperature=0)
model_with_tools = model.bind_tools([retriever_tool])

SYSTEM_PROMPT = (
    "You are a health information assistant. You have access to a tool called "
    "search_health_docs that searches official documents about depression and "
    "hypertension ONLY.\n\n"
    "Rules:\n"
    "- If the user's question is about depression, hypertension, their symptoms, "
    "causes, diagnosis, or treatment, call search_health_docs.\n"
    "- If the question is unrelated to these two topics (e.g. general knowledge, "
    "other medical conditions, casual conversation), do NOT call the tool. "
    "Instead, respond directly and briefly explain this assistant only covers "
    "depression and hypertension information."
)

def generate_query_or_response(state: AgentState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state['messages']
    response = model_with_tools.invoke(messages)

    if response.tool_calls:
        return {'messages': [response]}

    cleaned_text = extract_text(response.content).strip()
    return {'messages': [AIMessage(content=cleaned_text)]}