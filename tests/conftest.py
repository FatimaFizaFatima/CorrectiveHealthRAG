"""Shared test setup.

These tests do not need Gemini, PostgreSQL, the embedding model, or a running
MCP server. The LLM is a scripted fake and the MCP tools are small fake tools.
"""
import os

# Must be set before the app is imported. This keeps tests away from the real DB.
os.environ.setdefault("GOOGLE_API_KEY", "test-key")
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool, ToolException, tool
from langgraph.checkpoint.memory import MemorySaver

import app.src.retriever as retriever_module


@tool
def search_health_docs(query: str) -> str:
    """Search official health information about depression and hypertension."""
    return "Content: Depression symptoms include sadness. \n PDF Name: depression"


# Replace the real vector store (slow, needs the embedding model) with a fake tool.
# This must happen before app.src.graph is imported.
retriever_module.build_retriever_tool = lambda: search_health_docs

import app.src.graph as graph_module  # noqa: E402
from app.src.graph import build_graph  # noqa: E402


# ---------------------------------------------------------------- fake LLM

PLAN = {
    "depression": ("search_health_docs", {"query": "depression"}),
    "refund": ("search_documents", {"query": "refund"}),
    "document 999": ("get_document", {"doc_id": 999}),
    "bad args": ("get_document", {"doc_id": "abc"}),
    "document 1": ("get_document", {"doc_id": 1}),
    "usd": ("get_exchange_rate", {"base": "USD", "target": "EUR"}),
    "delete document 2": ("delete_document", {"doc_id": 2}),
    "delete document 3": ("delete_document", {"doc_id": 3}),
    "ghost": ("no_such_tool", {}),
}


class ScriptedLLM(BaseChatModel):
    """Calls the tool listed in PLAN for a keyword, and echoes tool results."""

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        last = messages[-1]
        if last.type == "tool":
            message = AIMessage(content=f"DONE: {last.content}")
        else:
            text = str(last.content).lower()
            message = AIMessage(content="No tool needed.")
            for keyword, (name, args) in PLAN.items():
                if keyword in text:
                    message = AIMessage(
                        content="",
                        tool_calls=[
                            {"name": name, "args": args, "id": "call-1", "type": "tool_call"}
                        ],
                    )
                    break
        return ChatResult(generations=[ChatGeneration(message=message)])


# ---------------------------------------------------------------- fixtures

@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def executed():
    """Records every MCP tool that really ran."""
    return []


@pytest.fixture
def grade_log():
    """Records every time the document grader ran."""
    return []


@pytest.fixture
def mcp_tools(executed):
    def search_documents(query: str) -> str:
        """Search company documents by keyword and return the best matches."""
        executed.append("search_documents")
        return '{"query": "refund", "count": 1}'

    def get_document(doc_id: int) -> str:
        """Fetch the full text of one document by its id."""
        executed.append("get_document")
        if doc_id == 999:
            raise ToolException("No document found with id 999")
        return '{"id": 1, "title": "Refund policy"}'

    def get_exchange_rate(base: str, target: str) -> str:
        """Get the latest official exchange rate between two currencies."""
        executed.append("get_exchange_rate")
        return '{"base": "USD", "target": "EUR", "rate": 0.9}'

    def delete_document(doc_id: int) -> str:
        """Permanently delete one document by its id. This cannot be undone."""
        executed.append("delete_document")
        return '{"deleted": true}'

    return [
        StructuredTool.from_function(f)
        for f in (search_documents, get_document, get_exchange_rate, delete_document)
    ]


@pytest.fixture(autouse=True)
def fake_llm_and_nodes(monkeypatch, grade_log):
    monkeypatch.setattr(
        "app.src.node.decide.get_llm", lambda temperature=0: ScriptedLLM()
    )

    def fake_grade(state):
        grade_log.append("graded")
        return "generate_answer"

    def fake_generate(state):
        return {"messages": [AIMessage(content="RAG ANSWER")], "filename": ["depression"]}

    monkeypatch.setattr(graph_module, "grade_documents", fake_grade)
    monkeypatch.setattr(graph_module, "generate_answer", fake_generate)


@pytest.fixture
def agent(mcp_tools):
    """Graph with MCP tools and an approval checkpointer."""
    return build_graph(mcp_tools, MemorySaver())


def config(thread_id: str = "t1") -> dict:
    return {"configurable": {"thread_id": thread_id}}