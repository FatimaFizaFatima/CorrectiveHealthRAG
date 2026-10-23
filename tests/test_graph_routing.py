"""Tool selection, routing, and error handling inside the graph."""
import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.src.graph import build_graph
from tests.conftest import config


async def ask(graph, question, thread_id="t1"):
    state = {"messages": [HumanMessage(content=question)], "retry_count": 0}
    return await graph.ainvoke(state, config(thread_id))


def tool_messages(result):
    return [m for m in result["messages"] if m.type == "tool"]


@pytest.mark.anyio
async def test_health_question_uses_rag_and_is_graded(agent, grade_log):
    result = await ask(agent, "what are the symptoms of depression")
    assert grade_log == ["graded"]
    assert result["messages"][-1].content == "RAG ANSWER"


@pytest.mark.anyio
async def test_mcp_result_skips_grading(agent, grade_log, executed):
    result = await ask(agent, "What is our refund policy?")
    assert executed == ["search_documents"]
    assert grade_log == []
    assert result["messages"][-1].content.startswith("DONE:")


@pytest.mark.anyio
async def test_external_api_tool_runs(agent, executed):
    result = await ask(agent, "How much is 50 USD in EUR?")
    assert executed == ["get_exchange_rate"]
    assert "0.9" in result["messages"][-1].content


@pytest.mark.anyio
async def test_no_tool_is_used_for_unrelated_question(agent, executed, grade_log):
    result = await ask(agent, "who invented alternating current")
    assert executed == []
    assert grade_log == []
    assert result["messages"][-1].content == "No tool needed."


@pytest.mark.anyio
async def test_tool_error_is_returned_to_the_agent(agent):
    result = await ask(agent, "Give me document 999.")
    tool_msg = tool_messages(result)[0]
    assert tool_msg.status == "error"
    assert "No document found" in tool_msg.content
    assert "No document found" in result["messages"][-1].content


@pytest.mark.anyio
async def test_invalid_arguments_do_not_run_the_tool(agent, executed):
    result = await ask(agent, "bad args please")
    assert tool_messages(result)[0].status == "error"
    assert executed == []


@pytest.mark.anyio
async def test_unknown_tool_is_handled_without_crash(agent, executed):
    result = await ask(agent, "ghost tool please")
    tool_msg = tool_messages(result)[0]
    assert tool_msg.status == "error"
    assert "not a valid tool" in tool_msg.content
    assert executed == []


@pytest.mark.anyio
async def test_mcp_tool_is_unavailable_when_not_discovered(executed):
    """Same as the app running without the MCP server."""
    rag_only = build_graph(None, MemorySaver())
    result = await ask(rag_only, "What is our refund policy?")
    assert tool_messages(result)[0].status == "error"
    assert executed == []