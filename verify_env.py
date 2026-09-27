import os
import sys
import subprocess
import shutil

def run_pip_install():
    """Attempt to install requirements using pip."""
    print("Installing missing dependencies from requirements.txt...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("\n[SUCCESS] Dependencies installed successfully.\n")
        return True
    except Exception as e:
        print(f"\n[ERROR] Failed to install dependencies: {e}", file=sys.stderr)
        print("Please run manually: pip install -r requirements.txt", file=sys.stderr)
        return False

def verify_packages():
    """Verify that required packages are installed."""
    required = [
        ("dotenv", "python-dotenv"),
        ("langchain", "langchain"),
        ("langchain_core", "langchain-core"),
        ("langchain_community", "langchain-community"),
        ("mcp", "mcp"),
        ("fastmcp", "fastmcp"),
        ("langchain_mcp_adapters", "langchain-mcp-adapters"),
    ]
    
    # Add provider-specific checks based on selection
    # We will load dotenv first if possible
    try:
        import dotenv
        dotenv.load_dotenv()
    except ImportError:
        pass
        
    provider = os.getenv("LLM_PROVIDER", "google").lower()
    if provider == "google":
        required.append(("langchain_google_genai", "langchain-google-genai"))
    elif provider == "bedrock":
        required.append(("langchain_aws", "langchain-aws"))
        required.append(("boto3", "boto3"))
        
    missing = []
    for module_name, pip_name in required:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pip_name)
            
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        success = run_pip_install()
        if not success:
            sys.exit(1)
    else:
        print("[OK] All required Python packages are installed.")

def main():
    print("=== LangChain Basics: Environment Verification ===\n")
    
    # 1. Check for .env file
    if not os.path.exists(".env"):
        print("[WARNING] .env file not found.")
        if os.path.exists(".env.example"):
            print("Copying .env.example to .env...")
            shutil.copy(".env.example", ".env")
            print("[OK] Created .env file. Please open it and configure your API keys/credentials.")
        else:
            print("[ERROR] .env.example not found. Cannot create default .env.")
            sys.exit(1)
            
    # 2. Check packages
    verify_packages()
    
    # Reload environment after potential installations/dotenv setup
    import dotenv
    dotenv.load_dotenv()
    
    # Import our config helper
    try:
        import config
    except ImportError:
        print("[ERROR] config.py not found in current directory.", file=sys.stderr)
        sys.exit(1)
        
    provider = config.get_llm_provider()
    print(f"[INFO] Using LLM Provider: {provider.upper()}")
    
    # 3. Check credentials
    if provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            print("\n[ERROR] GOOGLE_API_KEY is not set or still has the placeholder value in .env.")
            print("Please edit the .env file with your actual Google Gemini API key.")
            sys.exit(1)
        else:
            print("[OK] GOOGLE_API_KEY environment variable detected.")
            
    elif provider == "bedrock":
        aws_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
        if not aws_key or aws_key == "your_aws_access_key_here" or not aws_secret:
            print("\n[ERROR] AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY is not set in .env.")
            print("Please edit the .env file with your actual AWS credentials.")
            sys.exit(1)
        else:
            print("[OK] AWS Bedrock credentials detected.")
            
    # 4. Test LLM invocation
    print("\nAttempting to connect to LLM and send a test prompt...")
    try:
        llm = config.get_llm()
        response = llm.invoke("Say 'Hello LangChain!'")
        # In langchain, response is a BaseMessage (usually AIMessage)
        # Content is the text.
        print(f"[OK] LLM Connection successful!")
        print(f"    Response: {response.content.strip()}")
    except Exception as e:
        print(f"\n[ERROR] LLM invocation failed: {e}", file=sys.stderr)
        print("Please check your API key/credentials, network connection, or Bedrock model access permissions.", file=sys.stderr)
        sys.exit(1)
        
    # 5. Test Embeddings
    print("\nAttempting to generate test embeddings...")
    try:
        embeddings = config.get_embeddings()
        vector = embeddings.embed_query("Hello LangChain")
        print(f"[OK] Embedding generation successful! (Vector size: {len(vector)})")
    except Exception as e:
        print(f"\n[ERROR] Embedding generation failed: {e}", file=sys.stderr)
        print("Please check your API key/credentials or model access configuration.", file=sys.stderr)
        sys.exit(1)
        
    print("\n=== [SUCCESS] Environment is fully configured and ready! ===")

if __name__ == "__main__":
    main()
