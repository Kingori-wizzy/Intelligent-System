import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from transformers import pipeline

# Load environment variables (kept for consistency)
load_dotenv()

# Step 1: Collect all text files from data/
data_dir = "data"
texts = []

for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".txt"):
            with open(os.path.join(root, file), "r", encoding="utf-8", errors="ignore") as f:
                texts.append(f.read())

# Step 2: Split into chunks
splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=50)
docs = []
for text in texts:
    docs.extend(splitter.split_text(text))

# Step 3: Local HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(docs, embeddings)

# Step 4: Local HuggingFace LLM (Flan-T5 base for richer answers)
hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", tokenizer="google/flan-t5-base")
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Step 5: Interactive Q&A loop
print("📚 Interactive RAG System Ready! Type your questions below.")
print("Type 'exit' to quit.\n")

while True:
    query = input("Q: ")
    if query.lower() == "exit":
        print("👋 Exiting interactive mode.")
        break

    # Retrieve relevant docs
    docs = vectorstore.similarity_search(query, k=2)

    # Generate answer
    answer = llm.invoke(f"Based on these notes: {docs}, answer the question: {query}")
    print("A:", answer)
