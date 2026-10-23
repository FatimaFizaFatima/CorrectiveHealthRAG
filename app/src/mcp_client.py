import os
import warnings

from dotenv import load_dotenv
from fastmcp import Client
from langchain_core._api import LangChainBetaWarning

warnings.filterwarnings("ignore", category=LangChainBetaWarning)

from langchain.mcp import MCPAdapter  

load_dotenv()

MCP_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")


def get_mcp_adapter(token: str | None = None) -> MCPAdapter:
    """Build the MCP adapter. The token is sent as 'Authorization: Bearer <token>'."""
    token = token or os.getenv("MCP_TOKEN") or None
    return MCPAdapter(Client(MCP_URL, auth=token))