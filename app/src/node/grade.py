from pydantic import BaseModel, Field
from app.config import get_llm
from app.src.state import AgentState
from app.src.logger import log_node

class GradeDocuments(BaseModel):
    """Grade the relevance of retrieved documents to the user's question."""
    binary_score: str = Field(description="Relevance score: 'yes' if relevant, 'no' if not")

model = get_llm(temperature=0)
grader_model = model.with_structured_output(GradeDocuments)

GRADE_PROMPT = (
    "You are grading whether a retrieved document is relevant to a user's question.\n\n"
    "Question: {question}\n\n"
    "Retrieved document:\n{context}\n\n"
    "If the document contains information that helps answer the question, "
    "grade it as relevant. Give a binary score 'yes' or 'no'."
)

@log_node
def grade_documents(state: AgentState) -> str:
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    result = grader_model.invoke([{"role": "user", "content": prompt}])

    if result.binary_score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"



if __name__ == "__main__":
    from langchain_core.messages import HumanMessage, ToolMessage

    state_relevant = {
        "messages": [
            HumanMessage(content="what are the symptoms of depression"),
            ToolMessage(content="Common symptoms of depression include persistent sadness, loss of interest, fatigue...", tool_call_id="1"),
        ],
        "retry_count": 0,
    }
    print("Relevant test:", grade_documents(state_relevant))

    state_irrelevant = {
        "messages": [
            HumanMessage(content="what are the symptoms of depression"),
            ToolMessage(content="Hypertension is diagnosed using blood pressure measurements above 140/90 mmHg.", tool_call_id="2"),
        ],
        "retry_count": 0,
    }
    print("Irrelevant test:", grade_documents(state_irrelevant))    