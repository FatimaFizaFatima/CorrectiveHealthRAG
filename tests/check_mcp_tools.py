import asyncio

from app.src.mcp_client import get_mcp_adapter


async def main():
    async with get_mcp_adapter() as adapter:
        tools = await adapter.list_tools()
        for t in tools:
            print(t.name, "-", t.description.strip().splitlines()[0])

        by_name = {t.name: t for t in tools}
        print(await by_name["get_document"].ainvoke({"doc_id": 1}))


asyncio.run(main())