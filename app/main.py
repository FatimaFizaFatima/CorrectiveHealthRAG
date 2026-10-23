from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.memory import MemorySaver

from app.router.corrective_health_router import router as corrective_health_router
from app.src.graph import build_graph
from app.src.logger import logger
from app.src.mcp_client import get_mcp_adapter


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpointer = MemorySaver()

    async with AsyncExitStack() as stack:
        try:
            adapter = await stack.enter_async_context(get_mcp_adapter())
            mcp_tools = await adapter.list_tools()
            logger.info("MCP tools discovered: %s", [t.name for t in mcp_tools])
            app.state.graph = build_graph(mcp_tools, checkpointer)
        except Exception as exc:
            logger.warning(
                "MCP server unavailable or access refused (%s). Running with the health RAG tool only.",
                exc,
            )
            app.state.graph = build_graph(None, checkpointer)
        yield


app = FastAPI(lifespan=lifespan)

app.include_router(corrective_health_router)