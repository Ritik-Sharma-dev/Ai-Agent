import streamlit as st
import requests
import os

st.set_page_config(page_title="TalentScout AI", page_icon="🕵️‍♂️", layout="wide")

API_RANK_URL = "http://localhost:8000/api/rank-candidates"
API_UPLOAD_URL = "http://localhost:8000/api/upload-resumes"

st.title("TalentScout AI 🕵️‍♂️")
st.markdown("### Enterprise RAG Resume Screener")

# --- NEW: Sidebar Uploader ---
with st.sidebar:
    st.header("📂 Database Management")
    st.markdown("Add new resumes to the Vector Database.")
    
    # Drag and drop file uploader
    uploaded_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)
    
    if st.button("⚙️ Process & Add to Database", use_container_width=True):
        if not uploaded_files:
            st.warning("Please select files to upload first.")
        else:
            with st.spinner("Chunking & Embedding documents..."):
                try:
                    # Prepare files for the multipart/form-data request
                    files_to_send = [("files", (file.name, file.getvalue(), "application/pdf")) for file in uploaded_files]
                    response = requests.post(API_UPLOAD_URL, files=files_to_send)
                    
                    if response.status_code == 200:
                        st.success(response.json()["message"])
                    else:
                        st.error(f"Error: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("🚨 API connection failed. Is FastAPI running?")
                    
    st.markdown("---")
    st.caption("Architecture: Streamlit → FastAPI → LangChain → ChromaDB")

# --- Existing Main Interface ---
jd_text = st.text_area("Paste the Job Description here:", height=200)

if st.button("🔍 Find Best Candidates", type="primary"):
    if not jd_text.strip():
        st.warning("Please enter a Job Description first.")
    else:
        with st.spinner("🧠 Querying Vector Database & Analyzing Candidates..."):
            try:
                response = requests.post(API_RANK_URL, json={"job_description": jd_text})
                
                if response.status_code == 200:
                    candidates = response.json().get("candidates", [])
                    if not candidates:
                        st.info("No candidates found or the AI rejected everyone.")
                    else:
                        st.success(f"Successfully ranked {len(candidates)} candidate(s)!")
                        
                        for idx, cand in enumerate(candidates):
                            file_name = os.path.basename(cand.get("Candidate_File", "Unknown"))
                            score = cand.get("Match_Score", 0)
                            
                            with st.expander(f"#{idx + 1}: {file_name} - Match Score: {score}/100", expanded=(idx == 0)):
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.markdown(f"**🤖 AI Reasoning:**\n{cand.get('Reasoning', '')}")
                                    missing = cand.get("Missing_Skills", [])
                                    if missing:
                                        st.markdown("**❌ Missing Skills:**")
                                        st.write(", ".join([f"`{m}`" for m in missing]))
                                    else:
                                        st.success("**✅ No critical skills missing!**")
                                with col2:
                                    st.metric("Match Score", f"{score}%")
                else:
                    st.error(f"API Error: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("🚨 Could not connect to the API.")