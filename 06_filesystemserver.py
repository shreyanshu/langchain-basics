import asyncio
# python -m uv tool install duckduckgo-mcp-server
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent

import config


async def main():

    print("=" * 60)
    print("DuckDuckGo MCP + LangChain Agent")
    print("=" * 60)

    # Launch the MCP server
    import sys

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "uv",
            "tool",
            "run",
            "duckduckgo-mcp-server",
        ],
    )

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            print("Initializing MCP Session...")
            await session.initialize()

            print("Loading MCP tools...")

            tools = await load_mcp_tools(session)

            print("\nAvailable Tools")
            print("-" * 40)

            for tool in tools:
                print(tool.name)

            llm = config.get_llm(temperature=0)

            agent = create_agent(
                model=llm,
                tools=tools,
                system_prompt="""
You are a helpful research assistant.

Use the available DuckDuckGo tools whenever the
question requires current information or web pages.

Do not guess.

Always cite what you find.
"""
            )

            while True:

                query = input("\nAsk a question (or 'quit'): ")

                if query.lower() == "quit":
                    break

                print("\nThinking...\n")

                response = await agent.ainvoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": query
                            }
                        ]
                    }
                )

                print("=" * 60)
                print(response["messages"][-1].content)
                print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())