"""Tests against a REAL running MCP server.

Start the MCP server first (port 8000). If it is not running, the live tests
are skipped. The delete tool is never called here, so no data is changed.
"""
import socket
from urllib.parse import urlparse

import pytest

from app.src.mcp_client import MCP_URL, get_mcp_adapter


def server_is_running() -> bool:
    parsed = urlparse(MCP_URL)
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=1):
            return True
    except OSError:
        return False


live = pytest.mark.skipif(not server_is_running(), reason="MCP server is not running")


@pytest.fixture
def anyio_backend():
    return "asyncio"


@live
@pytest.mark.anyio
async def test_discovers_at_least_three_tools():
    async with get_mcp_adapter() as adapter:
        names = {t.name for t in await adapter.list_tools()}
    assert {"search_documents", "get_document", "get_exchange_rate"} <= names
    assert len(names) >= 3


@live
@pytest.mark.anyio
async def test_sensitive_tool_is_exposed_by_the_server():
    async with get_mcp_adapter() as adapter:
        names = {t.name for t in await adapter.list_tools()}
    assert "delete_document" in names


@live
@pytest.mark.anyio
async def test_successful_tool_call():
    async with get_mcp_adapter() as adapter:
        tools = {t.name: t for t in await adapter.list_tools()}
        result = await tools["get_document"].ainvoke({"doc_id": 1})
    assert "Refund policy" in str(result)


@live
@pytest.mark.anyio
async def test_missing_document_is_an_error():
    async with get_mcp_adapter() as adapter:
        tools = {t.name: t for t in await adapter.list_tools()}
        result = await tools["get_document"].ainvoke({"doc_id": 999})
    text = str(result)
    assert "Error executing tool" in text
    assert "No document found" in text


@live
@pytest.mark.anyio
async def test_invalid_input_is_rejected_by_the_server():
    async with get_mcp_adapter() as adapter:
        tools = {t.name: t for t in await adapter.list_tools()}
        result = await tools["get_document"].ainvoke({"doc_id": 0})
    text = str(result)
    assert "Error executing tool" in text
    assert "greater than or equal to 1" in text


@live
@pytest.mark.anyio
async def test_connection_without_a_token_is_refused(monkeypatch):
    monkeypatch.delenv("MCP_TOKEN", raising=False)
    with pytest.raises(Exception):
        async with get_mcp_adapter() as adapter:
            await adapter.list_tools()


@live
@pytest.mark.anyio
async def test_connection_with_a_wrong_token_is_refused():
    with pytest.raises(Exception):
        async with get_mcp_adapter(token="wrong-token") as adapter:
            await adapter.list_tools()


@pytest.mark.anyio
async def test_unreachable_server_raises_a_connection_error(monkeypatch):
    monkeypatch.setattr("app.src.mcp_client.MCP_URL", "http://localhost:9/mcp")
    with pytest.raises(Exception):
        async with get_mcp_adapter() as adapter:
            await adapter.list_tools()