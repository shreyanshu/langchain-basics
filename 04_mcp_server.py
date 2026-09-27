import sys
from fastmcp import FastMCP

# Step 1: Initialize the FastMCP server
# We name it "NebulaCoreServer". FastMCP automatically handles registering tools,
# formatting arguments, and handling the standard input/output communication protocol.
mcp = FastMCP("NebulaCoreServer")

# Step 2: Define Tools exposed by the MCP Server
# Note that we use the `@mcp.tool()` decorator from FastMCP.
# These tools are hosted *on the server* and will be discovered dynamically by the client.

@mcp.tool()
def fetch_customer_tier(customer_id: str) -> str:
    """
    Retrieve the premium loyalty membership tier for a given customer ID.
    Use this tool to find out what discounts or loyalty programs a customer belongs to.
    """
    # A simple mock database of customer membership tiers
    db = {
        "CUST-001": "Gold (15% discount eligible)",
        "CUST-002": "Silver (10% discount eligible)",
        "CUST-003": "Bronze (5% discount eligible)",
    }
    
    cust_id_upper = customer_id.upper()
    if cust_id_upper in db:
        return f"Customer {cust_id_upper} is in the {db[cust_id_upper]} tier."
    else:
        return f"Customer {cust_id_upper} is a new client with 'Standard' tier status (no special discounts)."

@mcp.tool()
def get_system_uptime() -> str:
    """
    Retrieve the current status and uptime of the local NebulaCore cluster.
    Use this tool if the user asks about system health, server status, or cluster uptime.
    """
    return "Cluster status: ACTIVE. Uptime: 45 days, 12 hours. Average load: 0.42. Nodes online: 8/8."

if __name__ == "__main__":
    # When this script is run directly, FastMCP boots up in stdio mode.
    # Stdio transport allows a parent process (like our client script) to launch this script 
    # as a subprocess and exchange JSON-RPC packets via stdin/stdout.
    # Note: We print debug information to stderr so it does not interfere with the stdio JSON-RPC transport!
    print("Starting NebulaCore MCP Server on Stdio transport...", file=sys.stderr)
    mcp.run()
