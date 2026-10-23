# MCP vs Direct API / Tool Integration

This note compares two ways to give an AI agent a tool. It uses examples from this project.

## The two approaches

**Direct integration.** The tool is code inside the agent app. In LangChain it is a function with the `@tool` decorator. The agent imports it and calls it in the same process.

```python
@tool
def get_exchange_rate(base: str, target: str) -> str:
    """Get the latest exchange rate."""
    ...  # calls the exchange rate API from inside the agent
```

**MCP integration.** The tool lives in a separate MCP server. The agent connects over HTTP, asks the server which tools exist, and calls them. The agent code contains no tool code and no tool names.

```python
async with MCPAdapter(Client(url, auth=token)) as adapter:
    mcp_tools = await adapter.list_tools()
```

## Comparison

| Topic | Direct tool | MCP tool |
|---|---|---|
| Where the code lives | Inside the agent project | In a separate server project |
| Adding a tool | Edit the agent code and redeploy the agent | Add it to the server. The agent finds it after a restart |
| Reuse | One agent or framework. Another agent must copy the code | Any MCP client can use it: LangGraph, other frameworks, desktop apps |
| Discovery | Fixed list in code | Dynamic. The agent asks the server at startup |
| Security boundary | The tool runs with the agent's permissions. The agent can do anything the code allows | The server checks a token and scopes for every call. The agent only gets what its token allows |
| Ownership | Same team and repository | Another team can own, version, and deploy the tools |
| Testing | Tested together with the agent | The server is tested alone (28 tests). The agent is tested with fake tools |
| Speed | A function call | An extra network call per tool call |
| Setup effort | Low | Higher: a second service, tokens, ports, discovery at startup |
| Failure modes | Tool code errors | Same, plus server down, refused token, and connection errors |

## What we saw in this project

* **Server-side permission.** With the reader token, the agent could not delete a document even after a human approved. The server refused it. A direct tool has no second boundary: if the agent code can call it, it runs.
* **Safe fallback.** When the MCP server was down or refused the token, the app still started and answered health questions. The MCP tools were gone, and the agent said it could not help with them.
* **No tool names in the agent.** The agent found the four tools at startup. The decide prompt only says to use a tool whose description fits.
* **Extra work.** We handled the MCP result format (a list of content blocks) and had to keep the connection open for the app lifetime. We also fixed a routing problem: an exchange rate result must not be graded as a health document.
* **Young tooling.** The `langchain.mcp` adapter is marked beta. The older `langchain-mcp-adapters` package did not work with the `mcp` 2.3 SDK we used.

## Why the health retriever stays a direct tool

`search_health_docs` is tightly tied to this agent. It uses the same vector store, runs on every health question, and no other agent needs it. A network hop would add cost and no benefit. The company documents, exchange rate, and delete tools are different. Other agents could share them, and delete needs its own permission check.

## When to use which

**Use a direct tool when:**

* The tool is small and used by one agent
* It depends on the agent's own data or state
* Speed matters and nobody else needs it

**Use MCP when:**

* Several agents, frameworks, or teams need the same tools
* The tools are owned or deployed separately from the agent
* You need a server-side security boundary, such as scopes on sensitive actions
* You want to add or change tools without redeploying the agent

**Use both**, as this project does: direct for the agent's core logic, MCP for shared and sensitive capabilities.

## Summary

MCP gives a reusable integration layer with its own security boundary. The price is a second service, more setup, and more failure cases. Direct tools are simpler and faster, but each one is tied to one agent and runs with that agent's full permissions.