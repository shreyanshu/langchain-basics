import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# Import config helper to get config-driven LLM and Embeddings
import config

def create_sample_kb_file(filepath):
    """Creates a sample text file acting as our local knowledge base."""
    kb_content = """
========================================
NEBULACORP INTERNAL PRODUCT DOCUMENTATION
========================================

1. Project Helix: Quantum Fusion Reactor
- Description: Project Helix is NebulaCorp's next-generation clean energy reactor.
- Launch Date: March 15, 2026.
- Operating Efficiency: 98.4%.
- Cooling System: Closed-loop liquid neon circulation system.
- Lead Scientist: Dr. Elena Vance.
- Location: Sector 4 Research Facility, Geneva.

2. NebulaOS Operating System
- Description: A decentralized, real-time operating system designed for edge devices.
- Kernel Version: 4.12-LTS.
- Security Protocol: Zero-Trust cryptographic verification on every instruction.
- Lead Architect: Marcus Finch.

3. Alpha-Grid Supercomputer
- Description: NebulaCorp's flagship quantum-classical hybrid supercomputer.
- Processing Speed: 500 Petaflops (classical equivalent) + 120 logical qubits.
- Main Purpose: Climate simulation and molecular modeling for drug discovery.
"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(kb_content.strip())
    print(f"[INFO] Created sample knowledge base file: {filepath}")

def main():
    print("=== LangChain Demo 01: Knowledge Base (RAG) Integration ===\n")
    
    # Ensure active LLM/Embeddings configuration is loaded
    llm = config.get_llm(temperature=0.1)
    embeddings = config.get_embeddings()
    
    # Step 1: Create and Load the Knowledge Base Document
    kb_file = "kb_data.txt"
    if not os.path.exists(kb_file):
        create_sample_kb_file(kb_file)
        
    print(f"Loading document: {kb_file}...")
    loader = TextLoader(kb_file, encoding="utf-8")
    docs = loader.load()
    print(f"Loaded {len(docs)} document(s).")
    
    # Step 2: Split the Documents into Chunks
    # Text splitting is crucial because LLMs have context limits, and embedding shorter, 
    # semantically focused chunks leads to better retrieval matches.
    print("\nSplitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Split original document into {len(chunks)} smaller chunks.")
    
    # Step 3: Embed Chunks and Index in Vector Store
    # We use InMemoryVectorStore here as a simple, zero-setup option.
    # In production, you would use a persistent store like Pinecone, PgVector, Chroma, or Qdrant.
    print("\nGenerating embeddings and indexing chunks in vector store...")
    vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
    print("[OK] Vector store initialized and populated.")
    
    # Step 4: Create the Retriever
    # A retriever is an interface that returns documents based on an unstructured query.
    # 'k=2' means it will retrieve the top 2 most similar document chunks.
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    # Step 5: Construct the RAG Chain
    # We define a prompt that instructs the LLM to only answer based on the retrieved context.
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # combine_docs_chain stuffed the retrieved document texts into the {context} variable of our prompt.
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # retrieval_chain binds the retriever and the Q&A chain:
    # 1. Takes the user 'input' query.
    # 2. Uses the retriever to fetch relevant documents and populates 'context'.
    # 3. Passes both to the QA chain, returning the output under 'answer'.
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # Step 6: Query the RAG System
    query = "Who is the lead scientist of Project Helix and what cools it?"
    print(f"\nQuerying: '{query}'")
    
    response = rag_chain.invoke({"input": query})
    
    print("\n--- RAG Response ---")
    print(response["answer"])
    
    print("\n--- Retrieved Source Documents ---")
    for i, doc in enumerate(response["context"]):
        print(f"\n[Source Chunk {i+1}]:")
        print(doc.page_content.strip())
        print("-" * 30)

if __name__ == "__main__":
    main()
