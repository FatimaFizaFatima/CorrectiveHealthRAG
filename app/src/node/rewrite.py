from langchain_core.messages import HumanMessage
from app.config import get_llm
from app.src.state import AgentState
from app.src.logger import log_node
from app.src.utils.extract_content import extract_text

model = get_llm(temperature=0)

REWRITE_PROMPT = (
    "Here is the original question:\n\n{question}\n\n"
    "Here is the most recent rephrased attempt, which still failed to retrieve relevant documents:\n\n{previous_attempt}\n\n"
    "Formulate a NEW question that approaches this topic from a different angle than the previous attempt "
    "(for example: different terminology, a narrower or broader focus, or a different aspect of the topic).\n"
    "Only return the single, precise question itself — no explanation, options, or extra text."
)

@log_node
def rewrite_question(state: AgentState):
    human_questions = [m.content for m in state["messages"] if m.type == "human"]
    original_question = human_questions[0]
    previous_attempt = human_questions[-1]

    prompt = REWRITE_PROMPT.format(question=original_question, previous_attempt=previous_attempt)

    response = model.invoke([{"role": "user", "content": prompt}])
    rewritten_text = extract_text(response.content).strip()

    return {
        "messages": [HumanMessage(content=rewritten_text)],
        "retry_count": state.get("retry_count", 0) + 1,
    }

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage as HM

    fake_state_retry_1 = {
        "messages": [HM(content="what causes high blood pressure")],
        "retry_count": 0,
    }
    fake_state_retry_2 = {
        "messages": [HM(content="what causes high blood pressure")],
        "retry_count": 1,
    }

    result_1 = rewrite_question(fake_state_retry_1)
    result_2 = rewrite_question(fake_state_retry_2)

    print("\n--- Retry 1 rewrite ---")
    print(result_1["messages"][-1].content)

    print("\n--- Retry 2 rewrite ---")
    print(result_2["messages"][-1].content)

    print("\n--- Are they identical? ---")
    print(result_1["messages"][-1].content == result_2["messages"][-1].content)