import asyncio
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent

# Import config helper to get config-driven LLM
import config

# Setup parameters to launch our local MCP server (04_mcp_server.py)
# We run it using the same Python interpreter (sys.executable) to ensure it uses
# the same virtual environment/packages.
server_params = StdioServerParameters(
    command=sys.executable,
    args=["04_mcp_server.py"],
    env=None # Inherits parent environment variables
)

async def main():
    print("=== LangChain Demo 05: MCP Server Client Agent ===\n")
    print("Launching MCP Server process and establishing Stdio connection...")
    
    # 1. Establish connection with the MCP server
    async with stdio_client(server_params) as (read, write):
        # 2. Establish a Client Session over the stdio transport channel
        async with ClientSession(read, write) as session:
            # 3. Initialize the session. This negotiates protocol details and fetches server capabilities.
            await session.initialize()
            print("[OK] MCP Server connection initialized.")
            
            # 4. Dynamically load tools from the MCP server
            # load_mcp_tools automatically reads the tools exposed by the MCP server session
            # and converts them into standard LangChain BaseTool objects.
            print("\nDiscovering and loading tools from MCP server...")
            mcp_tools = await load_mcp_tools(session)
            
            print(f"[OK] Loaded {len(mcp_tools)} tools from server:")
            for tool in mcp_tools:
                print(f" - {tool.name}: {tool.description.strip()}")
                
            # 5. Initialize the configuration-driven LLM
            llm = config.get_llm(temperature=0.0)
            
            # 6. Build a standard LangChain Tool-Calling Agent
            agent = create_agent(
                model=llm,
                tools=mcp_tools,
                system_prompt=(
                    "You are a customer assistant with access to tools hosted on a "
                    "Model Context Protocol (MCP) server. "
                    "Use the available tools whenever necessary."
                ),
            )
            
            # 8. Run queries requiring MCP tools
            
            # Query A: Requires get_system_uptime tool
            query_a = "How is our local cluster health looking and what is its uptime?"
            response = await agent.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": query_a,
                        }
                    ]
                }
            )

            print(response["messages"][-1].content)
            # Query B: Requires fetch_customer_tier tool
            query_b = "I have a customer requesting a discount. Their customer ID is CUST-001. Can you look up their loyalty tier?"
            print(f"Sending Query B: '{query_b}'")
            print("--- Agent Execution Start ---")
            response = await agent.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": query_b,
                        }
                    ]
                }
            )

            print("--- Agent Execution End ---")
            print(f"\nFinal Response:\n{response["messages"][-1].content}\n")

if __name__ == "__main__":
    # The MCP python client uses asyncio to manage standard I/O channels asynchronously.
    # Therefore, we run our main entrypoint within an asyncio event loop.
    asyncio.run(main())
