import os
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables (Make sure OPENAI_API_KEY is in your .env)
# This points to the .env file in the parent directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# Configuration paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "talentscout_resumes"

def ingest_resumes():
    print(f"📂 Scanning directory: {DATA_DIR}")
    
    # 1. Load PDFs
    # PyPDFDirectoryLoader extracts text and adds metadata (like file name/page number)
    loader = PyPDFDirectoryLoader(DATA_DIR)
    documents = loader.load()
    
    if not documents:
        print("⚠️ No PDFs found in the 'data' directory. Please add some resumes!")
        return

    print(f"📄 Loaded {len(documents)} document pages.")

    # 2. Chunk the Text (The "Secret Sauce")
    # We split by paragraphs/sentences to maintain context.
    # The 'overlap' ensures a list of skills isn't accidentally cut in half.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ Split documents into {len(chunks)} contextual chunks.")

    # 3. Create Embeddings & Store in ChromaDB
    print("🧠 Embedding chunks and saving to ChromaDB... (This may take a moment)")
    
    # Using OpenAI's optimized embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small") 

    # Generate the vectors and save them locally
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )
    
    print(f"✅ Success! Vector database securely saved to: {CHROMA_DIR}")

if __name__ == "__main__":
    # Ensure the data directory exists before we try to read from it
    os.makedirs(DATA_DIR, exist_ok=True)
    ingest_resumes()