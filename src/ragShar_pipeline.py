import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = os.path.join(BASE_DIR, "data", "Sharleen-about.pdf")  # Updated filename
DB_DIR = os.path.join(BASE_DIR, "chroma_db_storage")

def initialize_rag():
    """Initialize or load the RAG system."""
    print("🚀 Initializing RAG system...")
    
    # Check if PDF exists
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: File not found at {PDF_PATH}")
        print(f"📁 Please place your PDF in: {os.path.join(BASE_DIR, 'data')}")
        return None
    
    # Load or create vector store
    embeddings = OllamaEmbeddings(model="llama3.1")
    
    if os.path.exists(DB_DIR):
        print("📂 Loading existing database...")
        vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        print("✅ Database loaded!")
    else:
        print("📄 Loading PDF...")
        loader = PyPDFLoader(PDF_PATH)
        data = loader.load()
        print(f"   ✓ {len(data)} pages loaded")
        
        print("✂️  Chunking text...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(data)
        print(f"   ✓ {len(chunks)} chunks created")
        
        print("💾 Creating vector database...")
        vector_db = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings,
            persist_directory=DB_DIR
        )
        vector_db.persist()
        print("✅ Database created and saved!")
    
    # Initialize LLM
    llm = Ollama(model="llama3.1", temperature=0.1)
    
    # Create prompt
    # reasoning-focused one
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior talent analyst conducting a professional competency assessment.

    CONTEXT (candidate information):
    {context}

    TASK: Analyze if this person would be suitable for the specified role.

    YOUR ANALYSIS MUST:
    1. Extract RELEVANT evidence from the context
    2. REASON about how their skills/experience transfer to the role
    3. IDENTIFY gaps between their background and role requirements
    4. Make a REASONED judgment, not just "I don't know"

    FORMAT YOUR RESPONSE:
    📊 EVIDENCE FOUND:
    • [Specific fact from documents that relates to role]

    🧠 REASONING:
    • [How this evidence connects to role requirements]

    ✅ CONCLUSION:
    • [Clear, evidence-based judgment]

    If no direct evidence exists, use this format:
    ⚠️ LIMITED EVIDENCE: [What IS known about them]
    🔄 TRANSFERABLE SKILLS: [How existing skills might apply]
    ❓ GAPS: [What we don't know]
    📋 RECOMMENDATION: [Conditional assessment]

    Never say "I cannot find this information" without attempting analysis.
    """),
        ("human", "Based on the document, can {input}?")
    ])
    
    # Create chain
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(
        vector_db.as_retriever(search_kwargs={"k": 3}), 
        qa_chain
    )
    
    print("✅ RAG system ready!\n")
    return rag_chain, vector_db

def ask_question(rag_chain, question):
    """Ask a single question and get answer with sources."""
    print(f"\n❓ Question: {question}")
    print("💭 Thinking...")
    
    response = rag_chain.invoke({"input": question})
    
    print("\n📝 Answer:")
    print("-" * 40)
    print(response["answer"])
    print("-" * 40)
    
    # Show sources
    print("\n📚 Sources used:")
    for i, doc in enumerate(response["context"][:3]):
        page = doc.metadata.get('page', 'N/A')
        preview = doc.page_content[:150].replace('\n', ' ') + "..."
        print(f"   [{i+1}] Page {page}: {preview}")
    
    return response

def interactive_mode(rag_chain):
    """Interactive Q&A session."""
    print("\n" + "=" * 60)
    print("🤖 INTERACTIVE Q&A MODE")
    print("=" * 60)
    print("Type 'quit' to exit, 'sources' to see all documents, 'help' for commands")
    print("-" * 60)
    
    while True:
        question = input("\n💬 Your question: ").strip()
        
        if question.lower() == 'quit':
            print("👋 Goodbye!")
            break
        elif question.lower() == 'sources':
            print("\n📚 Document loaded:", os.path.basename(PDF_PATH))
            continue
        elif question.lower() == 'help':
            print("\nCommands:")
            print("  quit    - Exit the program")
            print("  sources - Show document info")
            print("  help    - Show this message")
            print("\nAsk any question about the document!")
            continue
        elif question == "":
            continue
        
        ask_question(rag_chain, question)

def test_mode(rag_chain):
    """Run some test questions automatically."""
    print("\n" + "=" * 60)
    print("🧪 RUNNING TEST QUESTIONS")
    print("=" * 60)
    
    test_questions = [
        "Summarize the key information about Sharleen.",
        "What are Sharleen's skills or expertise?",
        "What is Sharleen's educational background?",
        "What projects has Sharleen worked on?",
        "Tell me about Sharleen's professional experience.",
    ]
    
    for q in test_questions:
        ask_question(rag_chain, q)
        print("\n" + "-" * 60)

def main():
    print("=" * 60)
    print("📄 PERSONAL RAG SYSTEM - SHARLEEN'S DOCUMENTS")
    print("=" * 60)
    
    # Initialize the system
    result = initialize_rag()
    if result is None:
        return
    
    rag_chain, vector_db = result
    
    # Choose mode
    print("\nChoose mode:")
    print("1. 🧪 Test mode (run preset questions)")
    print("2. 💬 Interactive mode (ask your own questions)")
    print("3. ❌ Exit")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        test_mode(rag_chain)
        # After tests, offer interactive
        again = input("\nWant to ask more questions? (y/n): ").strip().lower()
        if again == 'y':
            interactive_mode(rag_chain)
    elif choice == '2':
        interactive_mode(rag_chain)
    else:
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()