"""FastAPI endpoints, startup discovery, and connection failure."""
import pytest
from fastapi.testclient import TestClient

from app.db.session import get_db
from app.main import app


class FakeAdapter:
    def __init__(self, tools):
        self.tools = tools

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def list_tools(self):
        return self.tools


class FakeDB:
    def __init__(self):
        self.saved = []

    def add(self, row):
        self.saved.append(row)

    def commit(self):
        pass


@pytest.fixture
def db():
    return FakeDB()


@pytest.fixture
def client(monkeypatch, mcp_tools, db):
    monkeypatch.setattr("app.main.get_mcp_adapter", lambda: FakeAdapter(mcp_tools))
    app.dependency_overrides[get_db] = lambda: db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def offline_client(monkeypatch, db):
    def unreachable():
        raise ConnectionError("MCP server is down")

    monkeypatch.setattr("app.main.get_mcp_adapter", unreachable)
    app.dependency_overrides[get_db] = lambda: db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def ask(client, question):
    return client.post("/corrective-health/", json={"question": question})


def approve(client, thread_id, approved):
    return client.post(
        "/corrective-health/approve",
        json={"thread_id": thread_id, "approved": approved},
    )


def test_rag_question_returns_answer_and_tool(client):
    body = ask(client, "what are the symptoms of depression").json()
    assert body["answer"] == "RAG ANSWER"
    assert body["tools_used"] == ["search_health_docs"]
    assert body["requires_approval"] is False


def test_mcp_tool_is_used_through_the_api(client):
    body = ask(client, "What is our refund policy?").json()
    assert body["tools_used"] == ["search_documents"]


def test_empty_question_is_rejected(client):
    assert ask(client, "   ").status_code == 422


def test_delete_needs_approval_then_runs(client, executed):
    body = ask(client, "Delete document 2").json()
    assert body["requires_approval"] is True
    assert body["pending_actions"] == [
        {"tool": "delete_document", "args": {"doc_id": 2}}
    ]
    assert executed == []

    final = approve(client, body["thread_id"], True).json()
    assert final["requires_approval"] is False
    assert executed == ["delete_document"]


def test_rejected_delete_does_not_run(client, executed):
    body = ask(client, "Delete document 2").json()
    final = approve(client, body["thread_id"], False).json()
    assert "rejected" in final["answer"]
    assert executed == []


def test_approve_unknown_thread_returns_404(client):
    assert approve(client, "does-not-exist", True).status_code == 404


def test_approve_twice_returns_404(client, executed):
    body = ask(client, "Delete document 2").json()
    assert approve(client, body["thread_id"], True).status_code == 200
    assert approve(client, body["thread_id"], True).status_code == 404
    assert executed == ["delete_document"]


def test_pending_request_is_logged_only_after_it_finishes(client, db):
    body = ask(client, "Delete document 2").json()
    assert db.saved == []
    approve(client, body["thread_id"], True)
    assert len(db.saved) == 1


def test_app_starts_and_serves_rag_when_mcp_server_is_down(offline_client):
    body = ask(offline_client, "what are the symptoms of depression").json()
    assert body["answer"] == "RAG ANSWER"


def test_mcp_tools_are_unavailable_when_mcp_server_is_down(offline_client, executed):
    response = ask(offline_client, "What is our refund policy?")
    assert response.status_code == 200
    assert "not a valid tool" in response.json()["answer"]
    assert executed == []