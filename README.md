# LangChain Basics: A Hands-On Guide

This project is a beginner-friendly, structured tutorial demonstrating the core concepts of the **LangChain** ecosystem. It focuses on the primary pillars of modern LLM application development:
1. **Knowledge Base (KB) / RAG Integration**: Ingesting, chunking, embedding, indexing, and retrieving custom data.
2. **Conversational Memory**: Storing and injecting history for multi-turn chats.
3. **Custom Tools**: Defining Python functions and giving them to agents as tools.
4. **Model Context Protocol (MCP) Integration**: Connecting LangChain agents to standard MCP servers (built with `FastMCP` and wired with `langchain-mcp-adapters`).

The project is fully **config-driven** and supports both **Google Gemini** (using API keys) and **AWS Bedrock** (using IAM credentials).

---

## Project Structure

- `requirements.txt`: Specifies the Python library dependencies.
- `.env.example`: A template file for your local environment configuration.
- `config.py`: A helper utility that dynamically sets up the LLM and Embeddings based on settings in `.env`.
- `verify_env.py`: Checks python dependencies and tests LLM/embeddings connections to verify credentials.
- `01_kb_rag.py`: Implements a simple Retrieval-Augmented Generation (RAG) pipeline.
- `02_memory.py`: Implements multi-turn chat memory and session-based isolation.
- `03_tools.py`: Demonstrates defining custom python tools and executing a tool-calling agent.
- `04_mcp_server.py`: A local Model Context Protocol (MCP) server exposing tools.
- `05_mcp_client.py`: A LangChain agent that dynamically connects to the MCP server and invokes its tools.

---

## 🛠️ Step 1: Installation & Configuration

### 1. Set Up Environment Variables
First, copy the example environment template into a `.env` file:
```bash
cp .env.example .env
```
Open `.env` and set your desired `LLM_PROVIDER` (`google` or `bedrock`) along with the corresponding credentials:

*   **Google Gemini**:
    Set `LLM_PROVIDER=google` and enter your `GOOGLE_API_KEY`.
*   **AWS Bedrock**:
    Set `LLM_PROVIDER=bedrock` and fill in `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, and the specific model IDs.

### 2. Verify Your Environment
Run the verification script. It will automatically check for, and offer to install, any missing dependencies, and run test invocations for both the LLM and Embedding models:
```bash
python verify_env.py
```
Ensure you see the `=== [SUCCESS] Environment is fully configured and ready! ===` message before proceeding.

---

## 📚 Step 2: Exploring the Demos

### 1. Knowledge Base (RAG) Integration
RAG extends the LLM's knowledge by retrieving relevant documents from an external source and adding them to the context window.
*   **Concepts Shown**:
    *   `TextLoader`: Loads raw documents.
    *   `RecursiveCharacterTextSplitter`: Chunks data to fit embedding and context windows.
    *   `InMemoryVectorStore` & `Embeddings`: Translates text chunks to high-dimensional vectors and stores them.
    *   `create_retrieval_chain`: Ties the retriever and the generator prompt together.
*   **To Run**:
    ```bash
    python 01_kb_rag.py
    ```

### 2. Conversational Memory
LLM APIs are stateless by default. To create a chatbot, we must feed prior messages back into the conversation context.
*   **Concepts Shown**:
    *   `MessagesPlaceholder`: Reserves a spot in the prompt template for history.
    *   `InMemoryChatMessageHistory`: Stores list of messages in memory.
    *   `RunnableWithMessageHistory`: Automatically fetches, inserts, and saves conversation messages.
    *   **Session Isolation**: Demonstrates how two different sessions maintain completely separate memories.
*   **To Run**:
    ```bash
    python 02_memory.py
    ```

### 3. Custom Tools & Agents
Agents use LLMs to reason about a task and decide which actions (tools) to execute, in what order.
*   **Concepts Shown**:
    *   `@tool` decorator: Annotates simple Python functions. LangChain converts their docstrings and type hints into JSON Schema tool definitions for the LLM.
    *   `create_tool_calling_agent`: Binds tools to a compatible model.
    *   `AgentExecutor`: Manages the iterative loop of "LLM decides -> execute tool -> return output -> LLM decides".
*   **To Run**:
    ```bash
    python 03_tools.py
    ```

### 4. Model Context Protocol (MCP) Server & Client
MCP is an open standard that allows clients (like LangChain applications) to discover and execute tools hosted on separate servers.
*   **Concepts Shown**:
    *   `FastMCP`: An easy framework to build local/remote MCP servers.
    *   `stdio_client` / `ClientSession`: Standard MCP transport protocol client.
    *   `load_mcp_tools`: Automatically fetches tool definitions from the server session and translates them into LangChain-compatible tools.
*   **How it works**:
    1. `04_mcp_server.py` defines a server exposing `fetch_customer_tier` and `get_system_uptime` tools.
    2. `05_mcp_client.py` spawns `04_mcp_server.py` as a background subprocess via standard I/O channels.
    3. The client fetches the schemas, converts them to LangChain tools, binds them to a local agent, and asks queries requiring those tools.
*   **To Run**:
    ```bash
    python 05_mcp_client.py
    ```

---

## 💡 Key Design Patterns Used

*   **LCEL (LangChain Expression Language)**: Allows composing prompts, models, and output parsers using the pipe operator (`prompt | llm`).
*   **Tool Calling**: Employs modern tool-calling APIs (native to Gemini and Claude) instead of legacy parser regex (ReAct).
*   **Configurability**: Keeps environment parameters decoupled from business logic using `config.py` and `python-dotenv`.
