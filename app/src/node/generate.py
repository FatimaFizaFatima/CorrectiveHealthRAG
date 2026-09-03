from app.config import get_llm
from app.src.state import AgentState
from app.src.logger import log_node
from app.src.utils.extract_content import extract_text, get_latest_tool_messages
from langchain_core.messages import AIMessage
import re


model = get_llm(temperature=0.3)

GENERATE_PROMPT = (
    "You are a health information assistant. Answer the question using ONLY "
    "the context below. If the context does not contain enough information "
    "to answer, say so clearly instead of guessing.\n\n"
    "Question: {question}\n\n"
    "Context:\n{context}\n\n"
    "Answer:"
)

@log_node
def generate_answer(state: AgentState):
    question = state["messages"][0].content

    context = "\n\n".join(m.content for m in get_latest_tool_messages(state["messages"]))

    filename = re.findall(r"PDF Name: (\w+)", context)

    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = model.invoke([{"role": "user", "content": prompt}])

    cleaned_text = extract_text(response.content).strip()

    return {"messages": [AIMessage(content=cleaned_text)], "filename": filename}