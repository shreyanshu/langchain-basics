import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

def get_llm_provider():
    """Get the active LLM provider from environment variables."""
    return os.getenv("LLM_PROVIDER", "google").lower()

def get_llm(temperature=0.0):
    """
    Initialize and return the LLM based on the configured provider.
    Supports Google Gemini ('google') and AWS Bedrock ('bedrock').
    """
    provider = get_llm_provider()
    
    if provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("[ERROR] GOOGLE_API_KEY environment variable is not set.", file=sys.stderr)
            print("Please create a .env file and set GOOGLE_API_KEY, or run verify_env.py for assistance.", file=sys.stderr)
            sys.exit(1)
            
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            # Use gemini-1.5-flash as the default lightweight, fast model
            return ChatGoogleGenerativeAI(
                model="gemini-3.1-flash-lite",
                temperature=temperature,
                google_api_key=api_key
            )
        except ImportError:
            print("[ERROR] langchain-google-genai is not installed. Run 'pip install langchain-google-genai'", file=sys.stderr)
            sys.exit(1)
            
    elif provider == "bedrock":
        # Boto3 will automatically pick up AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
        # AWS_SESSION_TOKEN, and AWS_DEFAULT_REGION from the environment if they are set.
        model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
        region_name = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        
        try:
            from langchain_aws import ChatBedrock
            return ChatBedrock(
                model_id=model_id,
                region_name=region_name,
                model_kwargs={"temperature": temperature}
            )
        except ImportError:
            print("[ERROR] langchain-aws is not installed. Run 'pip install langchain-aws boto3'", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"[ERROR] Unsupported LLM_PROVIDER: '{provider}'. Must be 'google' or 'bedrock'.", file=sys.stderr)
        sys.exit(1)

def get_embeddings():
    """
    Initialize and return the embedding model based on the configured provider.
    """
    provider = get_llm_provider()
    
    if provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("[ERROR] GOOGLE_API_KEY environment variable is not set for embeddings.", file=sys.stderr)
            sys.exit(1)
            
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            # text-embedding-004 is Google's default embedding model
            return GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-2-preview",
                google_api_key=api_key
            )
        except ImportError:
            print("[ERROR] langchain-google-genai is not installed.", file=sys.stderr)
            sys.exit(1)
            
    elif provider == "bedrock":
        model_id = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v1")
        region_name = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        
        try:
            from langchain_aws import BedrockEmbeddings
            return BedrockEmbeddings(
                model_id=model_id,
                region_name=region_name
            )
        except ImportError:
            print("[ERROR] langchain-aws is not installed.", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"[ERROR] Unsupported LLM_PROVIDER: '{provider}' for embeddings.", file=sys.stderr)
        sys.exit(1)

def get_vector_store(documents, embeddings):
    """
    Initialize and populate a vector store based on the active configuration.
    Supports local in-memory store ('in_memory') and remote cloud store ('pinecone').
    """
    provider = os.getenv("VECTOR_STORE_PROVIDER", "in_memory").lower()
    
    if provider == "in_memory":
        try:
            from langchain_core.vectorstores import InMemoryVectorStore
            return InMemoryVectorStore.from_documents(documents, embeddings)
        except ImportError as e:
            print(f"[ERROR] Failed to load InMemoryVectorStore: {e}", file=sys.stderr)
            sys.exit(1)
            
    elif provider == "pinecone":
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME")
        
        if not api_key or api_key == "your_pinecone_api_key_here":
            print("[ERROR] PINECONE_API_KEY environment variable is not set.", file=sys.stderr)
            print("Please edit your .env file with your actual Pinecone API key.", file=sys.stderr)
            sys.exit(1)
            
        if not index_name or index_name == "your_pinecone_index_name_here":
            print("[ERROR] PINECONE_INDEX_NAME environment variable is not set.", file=sys.stderr)
            sys.exit(1)
            
        try:
            from langchain_pinecone import PineconeVectorStore
            # Binds to the Pinecone index and uploads document embeddings.
            # Make sure your Pinecone index has been pre-created with correct dimensions:
            # - Google text-embedding-004: 768 dimensions
            # - Titan Embeddings: 1536 (default) or 1024 dimensions
            print(f"[INFO] Uploading embeddings to Pinecone index: '{index_name}'...")
            return PineconeVectorStore.from_documents(
                documents=documents,
                embedding=embeddings,
                index_name=index_name,
                pinecone_api_key=api_key
            )
        except ImportError:
            print("[ERROR] langchain-pinecone or pinecone-client is not installed. Run 'pip install langchain-pinecone'", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"[ERROR] Unsupported VECTOR_STORE_PROVIDER: '{provider}'. Must be 'in_memory' or 'pinecone'.", file=sys.stderr)
        sys.exit(1)

