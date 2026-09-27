from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_agent

# Import config helper to get config-driven LLM
import config

# Step 1: Define Custom Tools using the `@tool` decorator
# The docstring and typing of these functions are critical! LangChain uses them
# to generate the JSON Schema that is sent to the LLM so it knows when and how to call the tools.

@tool
def get_product_stock(product_name: str) -> str:
    """
    Check the current warehouse stock level for a given product name.
    Use this tool whenever a user asks if an item is available or in stock.
    """
    # Mock stock database
    database = {
        "helix-core": {"stock": 14, "price": 250.00},
        "nebulaos-pro": {"stock": 0, "price": 99.99},
        "alpha-grid-time": {"stock": 5, "price": 1200.00}
    }
    
    prod_lower = product_name.lower()
    if prod_lower in database:
        item = database[prod_lower]
        status = "in stock" if item["stock"] > 0 else "out of stock"
        return f"Product '{product_name}' is {status}. We have {item['stock']} units left in the warehouse. Regular Price: ${item['price']}."
    else:
        return f"Product '{product_name}' was not found in our database."

@tool
def calculate_discounted_price(price: float, discount_percent: float) -> float:
    """
    Calculate the final price of an item after applying a percentage discount.
    Use this tool to calculate savings or final sales prices.
    """
    discount_amount = price * (discount_percent / 100.0)
    final_price = price - discount_amount
    return round(final_price, 2)

def main():
    print("=== LangChain Demo 03: Custom Tools and Agents ===\n")
    
    # Initialize our config-driven LLM
    # Note: Tool calling requires a model capable of tool binding/function calling.
    # gemini-1.5-flash and claude-3-5-sonnet both natively support tool/function calling.
    llm = config.get_llm(temperature=0.0) # We set temperature=0.0 for deterministic tool execution
    
    # Bundle the tools in a list
    tools = [get_product_stock, calculate_discounted_price]
    
    print("Tools defined and loaded:")
    for t in tools:
        print(f" - {t.name}: {t.description.strip()}")
        
    # Step 2: Build the Agent Prompt
    # Tool-calling prompts require a placeholder 'agent_scratchpad' where LangChain
    # will insert intermediate tool call outputs and assistant reasoning steps.
    prompt = "You are a helpful customer service assistant for NebulaCorp. Answer questions using the tools provided. Be clear and polite."

    
    # Step 3: Create the Tool-Calling Agent
    # This agent handles the logic of determining which tool to call based on the LLM's instructions.
    agent = create_agent(model=llm, tools=tools, system_prompt=prompt)
    
    # Step 5: Execute the Agent with a complex multi-tool request
    query = "Hello! I want to buy the 'Helix-Core' chip. Can you check if it is currently in stock, and if it is, what would the final price be if you give me a 15% discount?"

    
    print(f"\nSending Query to Agent: '{query}'")
    print("\n--- Agent Execution Start ---")
    response = agent.invoke({ "messages": [
            {
                "role": "user",
                "content": query
            }
        ]})
    print("--- Agent Execution End ---\n")
    
    print("--- Agent Final Response ---")
    for m in response["messages"]:
        print(type(m).__name__, ":", m.content)

if __name__ == "__main__":
    main()
