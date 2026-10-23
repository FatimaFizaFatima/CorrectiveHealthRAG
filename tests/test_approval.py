"""Human approval for sensitive tools."""
import pytest
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from tests.conftest import config


async def start(graph, question, thread_id="t1"):
    state = {"messages": [HumanMessage(content=question)], "retry_count": 0}
    return await graph.ainvoke(state, config(thread_id))


async def decide(graph, decision, thread_id="t1"):
    return await graph.ainvoke(Command(resume=decision), config(thread_id))


@pytest.mark.anyio
async def test_sensitive_tool_pauses_before_running(agent, executed):
    result = await start(agent, "Delete document 2")
    interrupt = result["__interrupt__"][0].value
    assert interrupt["actions"] == [
        {"tool": "delete_document", "args": {"doc_id": 2}}
    ]
    assert executed == []


@pytest.mark.anyio
async def test_approve_runs_the_tool_once(agent, executed):
    await start(agent, "Delete document 2")
    result = await decide(agent, {"approved": True})
    assert executed == ["delete_document"]
    assert "__interrupt__" not in result
    assert result["messages"][-1].content.startswith("DONE:")


@pytest.mark.anyio
async def test_reject_never_runs_the_tool(agent, executed):
    await start(agent, "Delete document 2")
    result = await decide(agent, {"approved": False})
    assert executed == []
    assert "rejected" in result["messages"][-1].content


@pytest.mark.anyio
@pytest.mark.parametrize("bad_decision", ["yes", "true", {}, {"approved": "true"}])
async def test_unclear_decision_is_treated_as_reject(agent, executed, bad_decision):
    await start(agent, "Delete document 2")
    await decide(agent, bad_decision)
    assert executed == []


@pytest.mark.anyio
async def test_safe_tools_never_ask_for_approval(agent, executed):
    result = await start(agent, "What is our refund policy?")
    assert "__interrupt__" not in result
    assert executed == ["search_documents"]


@pytest.mark.anyio
async def test_approval_is_separate_for_each_thread(agent, executed):
    await start(agent, "Delete document 2", thread_id="A")
    await start(agent, "Delete document 3", thread_id="B")

    await decide(agent, {"approved": True}, thread_id="A")
    assert executed == ["delete_document"]

    state_b = await agent.aget_state(config("B"))
    assert any(task.interrupts for task in state_b.tasks)