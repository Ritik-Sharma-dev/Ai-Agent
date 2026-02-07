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
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    st.error("OPENAI_API_KEY not found in .env file")
    st.stop()
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
    
def normalize_requirements(requirements_json):
    system = {
        "role": "system",
        "content": (
            "You are an expert tech recruiter and software architect. "
            "Your job is to translate vague, metaphorical, or HR-fluff requirements "
            "into concrete, industry-standard skills, responsibilities, tools, and role titles. "
            "Avoid metaphors. Be specific and realistic."
        )
    }

    user = {
        "role": "user",
        "content": f"""
Given the following extracted job requirements JSON, rewrite it into a NORMALIZED, CONCRETE version.

Rules:
- Replace metaphors with real technical skills and responsibilities
- Infer the most likely real-world job title
- Fill in skills and keywords with real technologies and concepts
- Keep the same JSON structure
- Output ONLY valid JSON

Input JSON:
{json.dumps(requirements_json, ensure_ascii=False, indent=2)}
"""
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
    system = {
        "role": "system",
        "content": (
            "You are a senior technical recruiter and hiring manager. "
            "You write sharp, concrete, evidence-based cover letters for software and ML engineers. "
            "You avoid generic phrases, avoid fluff, and prefer specific technologies, systems, and outcomes. "
            "You write like someone who has shipped production systems."
        )
    }

    user = {
        "role": "user",
        "content": f"""
Write a concise, high-quality cover letter for the job below.

Rules:
- Use concrete examples, tools, and systems wherever possible
- Prefer actions and results over generic claims
- Avoid phrases like "I am confident" and "I am excited" unless backed by evidence
- If the resume lacks metrics, infer reasonable, conservative impact statements
- Keep it ~300-400 words
- Match the tone: {tone}
- Write like a mid-to-senior engineer applying for a real production role

Job title: {job_title}

Normalized requirements JSON:
{json.dumps(requirements_json, ensure_ascii=False, indent=2)}

Candidate profile / resume:
{profile_text}

Output only the cover letter text (no headers).
"""
    }

    out = call_chat_model([system, user], max_tokens=700, temperature=0.2)
    return out


def generate_video_script(profile_text, requirements_json, job_title, name="Candidate"):
    system = {
        "role": "system",
        "content": "You are a scriptwriter for short professional self-introduction videos for tech candidates."
    }

    user = {
        "role": "user",
        "content": f"""
Create a 60-second professional self-introduction video script for a job application.

Rules:
- Use concrete, professional language (no metaphors)
- Reference relevant skills and experience from the profile and job requirements
- Include timestamps (e.g., 0:00–0:10, 0:10–0:30, etc.)
- Include short camera/shot suggestions in brackets
- End with a short, confident call-to-action line
- Keep it realistic and suitable for LinkedIn or company career pages

Job title: {job_title}

Normalized job requirements:
{json.dumps(requirements_json, ensure_ascii=False, indent=2)}

Candidate name: {name}

Candidate profile / resume:
{profile_text}

Output only the script text.
"""
    }

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
            st.subheader("Extracted requirements (raw)")
            st.json(requirements)

            normalized = normalize_requirements(requirements)
            st.subheader("Normalized requirements (concrete & real-world)")
            st.json(normalized)

            job_title = job_title_override.strip() or (normalized.get("title") if isinstance(normalized, dict) else "")

            cover = generate_cover_letter(profile_text, normalized, job_title, tone=tone)
            st.subheader("Tailored cover letter")
            st.text_area("Cover letter", cover, height=300)
            st.download_button("Download cover letter (.txt)", cover, file_name="cover_letter.txt")

            script = generate_video_script(profile_text, normalized, job_title, name=name)
            st.subheader("1-minute video script")
            st.text_area("Video script", script, height=260)
            st.download_button("Download video script (.txt)", script, file_name="video_script.txt")

st.markdown("---")
st.caption("Built with an LLM. Check outputs for factual accuracy and customize the cover letter before sending.")
