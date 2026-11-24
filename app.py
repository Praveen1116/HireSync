from pathlib import Path
import streamlit as st

from src.extractor import clean_text, extract_text_from_pdf
from src.matcher import HireSyncMatcher

# Streamlit Page Configuration
st.set_page_config(
    page_title="HireSync - Resume to JD Matcher",
    page_icon="🎯",
    layout="wide"
)

# Cache model so it loads only once into memory
@st.cache_resource
def load_matcher():
    return HireSyncMatcher()

# Load matcher
matcher = load_matcher()

# Header & Project Context
st.title("🎯 HireSync: Semantic Resume-to-Job Matching")
st.markdown(
    "**SIH26044 / Topic 12** — Deep Learning Based Skill Mapping & Scoring System. "
    "Compares **Baseline SBERT Cosine Similarity** against a **Learned PyTorch Neural Scorer**."
)
st.divider()

# Input Columns
col_resume, col_jd = st.columns(2)

with col_resume:
    st.subheader("📄 1. Candidate Resume")
    resume_input_mode = st.radio(
        "Choose input method:",
        options=["Upload PDF", "Paste Plain Text"],
        horizontal=True
    )

    resume_text = ""
    if resume_input_mode == "Upload PDF":
        uploaded_pdf = st.file_uploader("Upload Resume PDF", type=["pdf"])
        if uploaded_pdf is not None:
            resume_text = extract_text_from_pdf(uploaded_pdf)
            st.success("✅ PDF parsed successfully!")
            with st.expander("Preview Extracted Resume Text"):
                st.write(resume_text[:600] + ("..." if len(resume_text) > 600 else ""))
    else:
        sample_resume = (
            "Software Engineer with 3 years of experience in Python, PyTorch, Docker, "
            "and PostgreSQL on AWS. Built REST APIs using FastAPI and deployed deep learning models."
        )
        resume_raw = st.text_area("Paste Resume Text:", value=sample_resume, height=220)
        resume_text = clean_text(resume_raw)

with col_jd:
    st.subheader("📋 2. Job Description (JD)")
    sample_jd = (
        "Looking for a Machine Learning Engineer. Required skills: Python, PyTorch, "
        "Docker, and Kubernetes on AWS or GCP. Experience in deep learning and REST APIs is required."
    )
    jd_raw = st.text_area("Paste Job Description:", value=sample_jd, height=270)
    jd_text = clean_text(jd_raw)

st.write("")
analyze_btn = st.button("🚀 Analyze Match & Skill Gap", type="primary", use_container_width=True)

# Results Section
if analyze_btn:
    if not resume_text.strip():
        st.error("Please provide a resume (upload PDF or paste text) before analyzing.")
    elif not jd_text.strip():
        st.error("Please paste a Job Description before analyzing.")
    else:
        with st.spinner("Generating SBERT embeddings and calculating match scores..."):
            results = matcher.match(resume_text, jd_text)

        learned_score = results["learned_score"]
        baseline_score = results["baseline_cosine_score"]
        category = results["category"]
        skill_gap = results["skill_gap"]
        overlap_pct = round(skill_gap["overlap_ratio"] * 100, 1)

        st.divider()
        st.subheader("📊 Match Results")

        # 3 Score Metric Cards
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric(label="🧠 Learned Scorer (PyTorch MLP)", value=f"{learned_score}%", delta=category)
        with m_col2:
            diff = round(learned_score - baseline_score, 1)
            st.metric(label="📐 Baseline Cosine Similarity", value=f"{baseline_score}%", delta=f"{diff:+}% vs Baseline")
        with m_col3:
            st.metric(label="🎯 Explicit Skill Overlap", value=f"{overlap_pct}%")

        # Tabs for Skills and Model Evaluation Details
        tab_skills, tab_eval = st.tabs(["🧩 Skills Gap Analysis", "📈 Model Architecture & Test Evaluation"])

        with tab_skills:
            s_col1, s_col2 = st.columns(2)
            
            with s_col1:
                st.markdown("#### ✅ Matched Skills (Present in Resume)")
                matched = skill_gap["matched_skills"]
                if matched:
                    st.write(" ".join([f"`{s}`" for s in matched]))
                else:
                    st.info("No direct skill keyword matches found.")

            with s_col2:
                st.markdown("#### ⚠️ Missing Skills (Skills Gap in Resume)")
                missing = skill_gap["missing_skills"]
                if missing:
                    st.write(" ".join([f"`{s}`" for s in missing]))
                else:
                    st.success("All JD required skills are present in the resume!")

        with tab_eval:
            st.markdown("#### Academic Evaluation (Trained on Hugging Face Dataset)")
            st.markdown(
                """
                | Metric | Baseline Cosine | Learned Scorer (PyTorch MLP) | Improvement |
                | :--- | :--- | :--- | :--- |
                | **Test Set Accuracy** | 64.44% | **80.00%** | **+15.56%** |
                | **Separation Gap** | 11.27% | **26.74%** | **+15.47%** |
                | **NDCG@5 Ranking Quality** | 88.69% | **93.13%** | **+4.44%** |
                """
            )
            
            # Display charts if available in reports/
            report_dir = Path(__file__).resolve().parent / "reports"
            loss_img = report_dir / "loss_curve_regularized.png"
            eval_img = report_dir / "evaluation_comparison.png"

            img_col1, img_col2 = st.columns(2)
            if loss_img.exists():
                with img_col1:
                    st.image(str(loss_img), caption="Regularized Training & Validation Loss Curve")
            if eval_img.exists():
                with img_col2:
                    st.image(str(eval_img), caption="Test Set Ranking & Accuracy Comparison")

