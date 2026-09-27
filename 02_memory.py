from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from pywin.framework.toolmenu import tools

# Import config helper to get config-driven LLM
import config

# A dictionary to persist session histories in-memory.
# In a web application, this could be backed by a Redis database or SQL database.
session_histories = {}

def get_session_history(session_id: str):
    """
    Retrieve or create a new ChatMessageHistory instance for a given session ID.
    This function is passed to RunnableWithMessageHistory to fetch the history dynamically.
    """
    if session_id not in session_histories:
        session_histories[session_id] = InMemoryChatMessageHistory()
    return session_histories[session_id]

def main():
    print("=== LangChain Demo 02: Conversational Memory ===\n")
    
    # Initialize our config-driven LLM
    llm = config.get_llm(temperature=0.7) # Slightly higher temperature for chat variety
    
    # Step 1: Create the Chat Prompt Template
    # We use MessagesPlaceholder to specify where the history (list of prior messages) 
    # should be injected inside the prompt.
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly chatbot. Answer questions concisely and reference past facts if asked."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # Step 2: Build the Core Chain
    # We pipe the prompt to the LLM (LangChain Expression Language or LCEL)
    chain = prompt | llm
    
    # Step 3: Wrap the Chain with Message History Management
    # RunnableWithMessageHistory intercepts the inputs, retrieves history for the session, 
    # inserts it into 'history', runs the chain, and writes the model response back to the history.
    conversational_chain = RunnableWithMessageHistory(
        runnable=chain,
        tools = [],
        get_session_history=get_session_history,
        input_messages_key="input",      # Key in the input dict mapped to current prompt
        history_messages_key="history"   # Key in the prompt template mapped to history placeholder
    )
    
    # Step 4: Multi-Turn Conversation (Session A)
    session_id_a = "user_session_123"
    config_a = {"configurable": {"session_id": session_id_a}}
    
    print(f"--- Starting Session A (ID: {session_id_a}) ---")
    
    # Turn 1
    input_1 = "Hi! My name is Sahil. I enjoy building things with LLMs."
    print(f"\nUser: {input_1}")
    response_1 = conversational_chain.invoke({"input": input_1}, config=config_a)
    print(f"Bot : {response_1.content}")
    
    # Turn 2
    input_2 = "I'm looking to learn LangChain today."
    print(f"\nUser: {input_2}")
    response_2 = conversational_chain.invoke({"input": input_2}, config=config_a)
    print(f"Bot : {response_2.content}")
    
    # Turn 3 (Requires Memory)
    input_3 = "What is my name? And what did I tell you I like to do?"
    print(f"\nUser: {input_3}")
    response_3 = conversational_chain.invoke({"input": input_3}, config=config_a)
    print(f"Bot : {response_3.content}")
    
    # Step 5: Session Isolation Demo (Session B)
    # We start a conversation under a new session ID. The model should have no memory of Session A.
    session_id_b = "user_session_456"
    config_b = {"configurable": {"session_id": session_id_b}}
    
    print(f"\n--- Starting Session B (ID: {session_id_b}) - Session Isolation Test ---")
    
    input_b1 = "Do you know what my name is or what I plan to learn today?"
    print(f"User: {input_b1}")
    response_b1 = conversational_chain.invoke({"input": input_b1}, config=config_b)
    print(f"Bot : {response_b1.content}")
    
    # Verify contents of the raw message history store
    print("\n--- Raw Session Histories stored in memory ---")
    for sid, history in session_histories.items():
        print(f"Session {sid}: {len(history.messages)} message(s) recorded.")

if __name__ == "__main__":
    main()
