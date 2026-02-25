import os
import json
from dotenv import load_dotenv

# LangChain imports
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "talentscout_resumes"

def get_vector_db():
    """Loads the existing Chroma database."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

def rank_candidates(job_description, top_k=5):
    """Searches the DB and uses an LLM to rank the best matches."""
    print("🔍 Searching the vector database for matching resume chunks...")
    
    try:
        db = get_vector_db()
    except Exception as e:
        return {"error": f"Could not load database. Did you run ingest.py? Error: {e}"}
    
    # 1. Retrieve the most relevant resume chunks
    # We use "mmr" (Maximal Marginal Relevance) to ensure we get diverse results,
    # not just the same paragraph repeated 10 times.
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 10})
    docs = retriever.invoke(job_description)
    
    if not docs:
        return {"error": "No relevant resume data found in the database."}

    # 2. Combine the retrieved chunks into a single context block
    # We include the 'source' metadata so the AI knows which file it belongs to
    context = "\n\n---\n\n".join([
        f"Source File: {d.metadata.get('source', 'Unknown')}\nContent: {d.page_content}" 
        for d in docs
    ])

    print("🧠 Analyzing candidates with LLM...")

    context = "\n\n---\n\n".join([
        f"Source File: {d.metadata.get('source', 'Unknown')}\nContent: {d.page_content}" 
        for d in docs
    ])

    print("\n🧐 DEBUG: Here is the raw text retrieved from ChromaDB:")
    print(context[:500] + "...\n") # Print the first 500 characters to verify
    
    print("🧠 Analyzing candidates with LLM...")
    
    # 3. Use LLM to analyze and rank the findings
    # gpt-4o-mini is perfect here: it's incredibly fast, cheap, and smart enough for extraction
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) 
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert technical recruiter. 
        Evaluate EVERY candidate found in the provided resume excerpts against the job description. 
        Even if they are a terrible match, include them in the output with a low score.
        Output a JSON array of objects with the exact keys: 
        'Candidate_File' (string), 'Match_Score' (integer 0-100), 'Missing_Skills' (array of strings), and 'Reasoning' (short string).
        Sort from highest score to lowest."""),
        ("user", "JOB DESCRIPTION:\n{jd}\n\nRESUME EXCERPTS:\n{context}\n\nReturn ONLY valid JSON.")
    ])
    
    # Chain the prompt and the LLM together
    chain = prompt | llm
    response = chain.invoke({"jd": job_description, "context": context})
    
    # 4. Clean and parse the JSON output
    try:
        content = response.content.strip()
        # Strip markdown formatting if the LLM adds it
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
            
        return json.loads(content)
    except Exception as e:
        print("Failed to parse JSON:", e)
        return {"raw_response": response.content}

if __name__ == "__main__":
    # A quick test to run directly in the terminal
    sample_jd = """
    We are looking for a Backend Developer to join our team. 
    Must have strong Python programming skills.
    Experience building web APIs with FastAPI is highly required.
    Familiarity with AI, LLMs, or LangChain is a massive plus.
    """
    
    results = rank_candidates(sample_jd)
    
    print("\n🏆 Top Candidates (JSON Output):")
    print(json.dumps(results, indent=2))