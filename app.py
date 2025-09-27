# app.py
import streamlit as st
import os, io, json
from dotenv import load_dotenv
import pdfplumber
from docx import Document
import openai

# Load .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Set API key for OpenAI
openai.api_key = ("OPENAI_API_KEY")
st.set_page_config(page_title="JD -> Cover Letter + Video Script", layout="wide")

# ------------------ Helpers ------------------
def read_file_text(uploaded_file):
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    try:
        if name.endswith(".pdf"):
            text = []
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for p in pdf.pages:
                    txt = p.extract_text()
                    if txt:
                        text.append(txt)
            return "\n".join(text)
        elif name.endswith(".docx"):
            doc = Document(io.BytesIO(data))
            return "\n".join([p.text for p in doc.paragraphs])
        else:
            return data.decode("utf-8", errors="ignore")
    except Exception:
        return data.decode("utf-8", errors="ignore")

def call_chat_model(messages, max_tokens=800, temperature=0, model="gpt-3.5-turbo"):
    resp = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return resp.choices[0].message["content"]

# ------------------ Core Functions ------------------
def extract_requirements(jd_text):
    system = {"role":"system","content":"You are an expert technical recruiter and extractor. Return only valid JSON (no markdown)."}
    user = {
        "role":"user",
        "content":(
            "Job description below. Extract a JSON object with keys:\n"
            "title (string), responsibilities (array of short strings), must_have (array), nice_to_have (array), "
            "experience_years (string or null), skills (array), keywords (array).\n\n"
            f"Job description:\n{jd_text}\n\n"
            "IMPORTANT: Output only valid JSON."
        )
    }
    out = call_chat_model([system, user], max_tokens=800, temperature=0)
    try:
        return json.loads(out)
    except Exception:
        start = out.find("{")
        end = out.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(out[start:end+1])
            except Exception:
                return {"raw": out}
        return {"raw": out}

def generate_cover_letter(profile_text, requirements_json, job_title, tone="professional"):
    system = {"role":"system","content":"You are a helpful career coach that writes tailored cover letters."}
    user = {"role":"user","content":f"""
Write a concise, tailored cover letter for the job below. Use the candidate's profile to highlight exact matches to the MUST_HAVE requirements.
Keep it to ~300-400 words. Use a confident, {tone} tone.

Job title: {job_title}
Extracted requirements JSON: {json.dumps(requirements_json, ensure_ascii=False)}

Candidate profile / resume:
{profile_text}

Output the cover letter as plain text (no headers).
"""}
    out = call_chat_model([system, user], max_tokens=700, temperature=0.2)
    return out

def generate_video_script(profile_text, requirements_json, job_title, name="Candidate"):
    system = {"role":"system","content":"You are a scriptwriter for short professional intro videos."}
    user = {"role":"user","content":f"""
Produce a one-minute (60 seconds) video script introducing the candidate for this role.
Provide timestamps for sections (e.g., 0:00-0:10) and short camera/shot suggestions (e.g., medium close-up).
Keep it natural, concise and persuasive.

Job title: {job_title}
Extracted requirements: {json.dumps(requirements_json, ensure_ascii=False)}
Candidate profile:
{profile_text}

Make the script ~60 seconds and include a 1-line spoken call-to-action at the end (e.g., 'I'd love to discuss how I can help at ...').
"""}
    out = call_chat_model([system, user], max_tokens=500, temperature=0.3)
    return out

# ------------------ Streamlit UI ------------------
st.title("AI Agent — JD → Tailored Cover Letter + 1-minute Video Script")

with st.sidebar:
    st.markdown("## Upload inputs")
    jd_file = st.file_uploader("Upload Job Description (PDF / DOCX / TXT)", type=["pdf","docx","txt"])
    resume_file = st.file_uploader("Upload Your Resume (optional)", type=["pdf","docx","txt"])
    profile_text_input = st.text_area("Or paste your resume/profile text (optional)", height=200)
    name = st.text_input("Your full name (for video script)", "Your Name")
    job_title_override = st.text_input("Override job title (optional)", "")
    tone = st.selectbox("Cover letter tone", ["professional","friendly","direct","enthusiastic"])

st.markdown("### Steps")
if st.button("Generate tailored cover letter + video script"):
    with st.spinner("Parsing files and calling the model..."):
        jd_text = read_file_text(jd_file) if jd_file else ""
        resume_text = read_file_text(resume_file) if resume_file else ""
        profile_text = profile_text_input.strip() or resume_text
        if not jd_text:
            st.error("Please upload a job description to continue.")
        else:
            requirements = extract_requirements(jd_text)
            st.subheader("Extracted requirements")
            st.json(requirements)

            job_title = job_title_override.strip() or (requirements.get("title") if isinstance(requirements, dict) else "")

            cover = generate_cover_letter(profile_text, requirements, job_title, tone=tone)
            st.subheader("Tailored cover letter")
            st.text_area("Cover letter", cover, height=300)
            st.download_button("Download cover letter (.txt)", cover, file_name="cover_letter.txt")

            script = generate_video_script(profile_text, requirements, job_title, name=name)
            st.subheader("1-minute video script")
            st.text_area("Video script", script, height=260)
            st.download_button("Download video script (.txt)", script, file_name="video_script.txt")

st.markdown("---")
st.caption("Built with an LLM. Check outputs for factual accuracy and customize the cover letter before sending.")
