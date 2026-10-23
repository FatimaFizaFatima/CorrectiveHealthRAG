import asyncio
import warnings

from langchain_core.messages import HumanMessage

from app.src.mcp_client import get_mcp_adapter
from app.src.graph import build_graph
from app.src.utils.extract_content import extract_text

warnings.filterwarnings("ignore")

QUESTIONS = [
    "what are the symptoms of depression",  
    "What is our refund policy?",            
    "Show me the full text of document 3.",  
    "How much is 50 USD in EUR?",            
    "Give me document 999.",                 
    "who invented alternating current",      
]


async def main():
    async with get_mcp_adapter() as adapter:
        mcp_tools = await adapter.list_tools()
        print("Discovered:", [t.name for t in mcp_tools])

        graph = build_graph(mcp_tools)

        for q in QUESTIONS:
            print("\n" + "=" * 60)
            print("USER:", q)
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content=q)], "retry_count": 0}
            )
            for m in result["messages"]:
                if m.type == "ai" and m.tool_calls:
                    for c in m.tool_calls:
                        print("  -> tool call:", c["name"], c["args"])
                elif m.type == "tool":
                    print(f"  <- tool result ({m.status}):", extract_text(m.content)[:150])
            print("AGENT:", extract_text(result["messages"][-1].content))


asyncio.run(main())