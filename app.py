import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import ast
import re

from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Job Market Intelligence",
    page_icon="💼",
    layout="wide"
)


# =========================================================
# CONFIGURATION
# =========================================================

HF_REPO = "Srichetan/job-market-distilbert"
MAX_LENGTH = 128


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 40px;
        font-weight: bold;
    }

    .sub-title {
        font-size: 18px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    text = str(text)

    text = re.sub(r"<.*?>", " ", text)

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def clean_resume_text(text):

    text = str(text)

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = re.sub(
        r"[^\w\s.,+#/&-]",
        " ",
        text
    )

    return text.strip()


# =========================================================
# LOAD DISTILBERT MODEL
# =========================================================

@st.cache_resource
def load_distilbert():

    tokenizer = AutoTokenizer.from_pretrained(
        HF_REPO
    )

    model = TFAutoModelForSequenceClassification.from_pretrained(
        HF_REPO
    )

    label_encoder_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="label_encoder.pkl",
        repo_type="model"
    )

    with open(label_encoder_path, "rb") as f:

        label_encoder = pickle.load(f)

    return tokenizer, model, label_encoder


# =========================================================
# LOAD JOB EMBEDDINGS
# =========================================================

@st.cache_resource
def load_job_embeddings():

    embedding_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="job_embeddings.npy",
        repo_type="model"
    )

    metadata_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="job_embedding_metadata.csv",
        repo_type="model"
    )

    embeddings = np.load(
        embedding_path
    )

    metadata = pd.read_csv(
        metadata_path
    )

    return embeddings, metadata


# =========================================================
# LOAD SENTENCE-BERT
# =========================================================

@st.cache_resource
def load_sbert():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        device="cpu"
    )

    return model


# =========================================================
# RESUME PDF EXTRACTION
# =========================================================

def extract_resume_text(uploaded_file):

    pdf_reader = PdfReader(
        uploaded_file
    )

    resume_text = ""

    for page in pdf_reader.pages:

        text = page.extract_text()

        if text:

            resume_text += text + "\n"

    return resume_text


# =========================================================
# JOB CATEGORY PREDICTION
# =========================================================

def predict_job_category(job_description):

    tokenizer, model, label_encoder = load_distilbert()

    cleaned_job = clean_text(
        job_description
    )

    tokens = tokenizer(
        cleaned_job,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="tf"
    )

    prediction = model.predict(
        {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"]
        },
        verbose=0
    )

    logits = prediction.logits

    probabilities = tf.nn.softmax(
        logits,
        axis=1
    ).numpy()[0]

    predicted_class = np.argmax(
        probabilities
    )

    confidence = (
        probabilities[predicted_class] * 100
    )

    predicted_category = (
        label_encoder.inverse_transform(
            [predicted_class]
        )[0]
    )

    return predicted_category, confidence


# =========================================================
# JOB TITLE CLASSIFIER
# =========================================================
# This is used to filter the precomputed jobs.
# It matches the same categories used during training.

def classify_job(title):

    title = str(title).lower()

    if any(keyword in title for keyword in [
        "machine learning",
        "ml engineer",
        "artificial intelligence",
        "ai engineer",
        "deep learning",
        "nlp",
        "natural language",
        "computer vision",
        "data scientist"
    ]):

        return "AI & Machine Learning"


    elif any(keyword in title for keyword in [
        "data analyst",
        "data analytics",
        "data engineer",
        "business analyst",
        "business intelligence",
        "bi analyst",
        "analytics",
        "reporting analyst",
        "data quality",
        "data architect"
    ]):

        return "Data & Analytics"


    elif any(keyword in title for keyword in [
        "software engineer",
        "software developer",
        "developer",
        "programmer",
        "web developer",
        "frontend",
        "front end",
        "backend",
        "back end",
        "full stack",
        "full-stack",
        "application developer",
        "mobile developer",
        "ios developer",
        "android developer",
        "java developer",
        "python developer"
    ]):

        return "Software Development"


    elif any(keyword in title for keyword in [
        "cloud",
        "devops",
        "site reliability",
        "sre",
        "system administrator",
        "systems administrator",
        "systems engineer",
        "network engineer",
        "network administrator",
        "it support",
        "information technology",
        "cyber security",
        "cybersecurity",
        "security engineer",
        "technical support",
        "infrastructure"
    ]):

        return "Cloud & IT"


    elif any(keyword in title for keyword in [
        "project manager",
        "product manager",
        "program manager",
        "operations manager",
        "business manager",
        "general manager",
        "management",
        "manager",
        "director",
        "vice president",
        "vp ",
        "chief",
        "supervisor",
        "team lead",
        "lead"
    ]):

        return "Business & Management"


    elif any(keyword in title for keyword in [
        "sales",
        "marketing",
        "business development",
        "account executive",
        "account manager",
        "sales representative",
        "sales associate",
        "marketing specialist",
        "marketing coordinator",
        "brand manager",
        "growth"
    ]):

        return "Sales & Marketing"


    elif any(keyword in title for keyword in [
        "accountant",
        "accounting",
        "financial analyst",
        "finance",
        "banking",
        "auditor",
        "controller",
        "bookkeeper",
        "investment",
        "tax analyst",
        "credit analyst"
    ]):

        return "Finance & Accounting"


    elif any(keyword in title for keyword in [
        "nurse",
        "nursing",
        "doctor",
        "physician",
        "healthcare",
        "health care",
        "medical",
        "clinical",
        "pharmacy",
        "therapist",
        "dental",
        "hospital"
    ]):

        return "Healthcare"


    elif any(keyword in title for keyword in [
        "mechanical engineer",
        "electrical engineer",
        "civil engineer",
        "chemical engineer",
        "industrial engineer",
        "manufacturing engineer",
        "quality engineer",
        "process engineer",
        "engineering"
    ]):

        return "Engineering"


    else:

        return "Other"


# =========================================================
# SKILL FUNCTIONS
# =========================================================

def get_skill_column(metadata):

    possible_columns = [
        "structured_skills",
        "skills",
        "job_skills",
        "skill_names",
        "required_skills"
    ]

    for column in possible_columns:

        if column in metadata.columns:

            return column

    return None


def convert_to_skill_list(value):

    if pd.isna(value):

        return []

    if isinstance(value, list):

        return value

    value = str(value).strip()

    try:

        result = ast.literal_eval(value)

        if isinstance(result, list):

            return [
                str(skill).strip()
                for skill in result
            ]

    except:

        pass

    if "," in value:

        return [
            skill.strip()
            for skill in value.split(",")
            if skill.strip()
        ]

    return [value]


def extract_resume_skills(
    resume_text,
    metadata
):

    skill_column = get_skill_column(
        metadata
    )

    if skill_column is None:

        return []

    all_skills = []

    for value in metadata[skill_column]:

        skills = convert_to_skill_list(
            value
        )

        all_skills.extend(
            skills
        )

    unique_skills = list(
        set(
            skill
            for skill in all_skills
            if skill
        )
    )

    resume_lower = resume_text.lower()

    resume_skills = []

    for skill in unique_skills:

        if skill.lower() in resume_lower:

            resume_skills.append(
                skill
            )

    return sorted(
        list(set(resume_skills))
    )


# =========================================================
# NEW JOB MATCHING FUNCTION
# =========================================================

def find_matching_jobs(
    resume_text,
    job_description,
    top_n=10
):

    # ---------------------------------------------
    # STEP 1: Predict category from Job Description
    # ---------------------------------------------

    predicted_category, confidence = (
        predict_job_category(
            job_description
        )
    )


    # ---------------------------------------------
    # STEP 2: Load precomputed jobs
    # ---------------------------------------------

    embeddings, metadata = (
        load_job_embeddings()
    )


    # ---------------------------------------------
    # STEP 3: Create job categories from titles
    # ---------------------------------------------
    # We do this here so the metadata does not
    # need a job_category column.

    if "title" not in metadata.columns:

        raise ValueError(
            "The job_embedding_metadata.csv file "
            "must contain a 'title' column."
        )

    metadata = metadata.copy()

    metadata["filter_category"] = (
        metadata["title"].apply(
            classify_job
        )
    )


    # ---------------------------------------------
    # STEP 4: Filter jobs according to JD category
    # ---------------------------------------------

    filtered_mask = (
        metadata["filter_category"]
        == predicted_category
    )

    filtered_metadata = (
        metadata.loc[
            filtered_mask
        ].copy()
    )

    filtered_embeddings = (
        embeddings[filtered_mask.values]
    )


    # ---------------------------------------------
    # STEP 5: Fallback
    # ---------------------------------------------
    # If no jobs exist for the predicted category,
    # use all jobs instead of showing an error.

    if len(filtered_metadata) == 0:

        st.warning(
            "No jobs were found in the predicted "
            "category. Showing the closest jobs "
            "from the available dataset."
        )

        filtered_metadata = metadata.copy()

        filtered_embeddings = embeddings


    # ---------------------------------------------
    # STEP 6: Create Resume Embedding
    # ---------------------------------------------

    sbert_model = load_sbert()

    resume_embedding = sbert_model.encode(
        resume_text,
        convert_to_numpy=True
    )

    resume_embedding = (
        resume_embedding.reshape(1, -1)
    )


    # ---------------------------------------------
    # STEP 7: Calculate similarity
    # ---------------------------------------------

    similarity_scores = cosine_similarity(
        resume_embedding,
        filtered_embeddings
    )[0]


    # ---------------------------------------------
    # STEP 8: Create results
    # ---------------------------------------------

    results = filtered_metadata.copy()

    results["similarity_score"] = (
        similarity_scores
    )

    results = results.sort_values(
        by="similarity_score",
        ascending=False
    )

    results = results.head(
        top_n
    ).reset_index(
        drop=True
    )

    results["match_percentage"] = (
        results["similarity_score"] * 100
    ).round(2)

    return (
        results,
        predicted_category,
        confidence
    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "💼 Navigation"
)

page = st.sidebar.radio(
    "Select a Module",
    [
        "🏠 Home",
        "📄 Resume Upload",
        "🎯 Job Prediction",
        "🔎 Job Matching",
        "📊 Skill Gap Analysis",
        "💡 Skill Recommendations"
    ]
)


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.markdown(
        '<div class="main-title">'
        'AI-Powered Job Market Intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "### Skill Gap Analysis System"
    )

    st.write(
        """
        An AI-powered application that helps job seekers
        understand job requirements, identify suitable job
        opportunities, analyze skill gaps, and receive
        skill recommendations.
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader(
            "🤖 Job Classification"
        )

        st.write(
            "Uses a fine-tuned DistilBERT model "
            "to classify job descriptions."
        )

    with col2:

        st.subheader(
            "🔎 Job Matching"
        )

        st.write(
            "Uses Sentence-BERT embeddings "
            "and cosine similarity."
        )

    with col3:

        st.subheader(
            "📊 Skill Gap"
        )

        st.write(
            "Compares candidate skills with "
            "job requirements."
        )

    st.divider()

    st.subheader(
        "🔄 Project Workflow"
    )

    st.write(
        """
        📄 Resume Upload
        ↓
        📝 Enter Job Description
        ↓
        🎯 Job Category Prediction
        ↓
        🔎 Category-Based Job Filtering
        ↓
        🤝 Resume–Job Matching
        ↓
        📈 Similarity Score
        ↓
        🛠️ Required Skills
        ↓
        👤 Candidate Skills
        ↓
        ❌ Missing Skills
        ↓
        💡 Skill Recommendations
        """
    )


# =========================================================
# RESUME UPLOAD
# =========================================================

elif page == "📄 Resume Upload":

    st.header(
        "📄 Upload Your Resume"
    )

    uploaded_file = st.file_uploader(
        "Upload your resume in PDF format",
        type=["pdf"]
    )

    if uploaded_file is not None:

        st.success(
            "Resume uploaded successfully! ✅"
        )

        st.write(
            "**File Name:**",
            uploaded_file.name
        )

        try:

            resume_text = extract_resume_text(
                uploaded_file
            )

            clean_resume = clean_resume_text(
                resume_text
            )

            st.session_state[
                "resume_text"
            ] = clean_resume

            st.success(
                "Resume text extracted successfully! ✅"
            )

            st.subheader(
                "📄 Extracted Resume Text"
            )

            st.text_area(
                "Resume Content",
                clean_resume,
                height=400
            )

            st.info(
                f"Resume contains approximately "
                f"{len(clean_resume.split())} words."
            )

        except Exception as e:

            st.error(
                f"Error reading resume: {e}"
            )


# =========================================================
# JOB PREDICTION
# =========================================================

elif page == "🎯 Job Prediction":

    st.header(
        "🎯 Job Category Prediction"
    )

    st.write(
        "Enter a complete job description below."
    )

    job_description = st.text_area(
        "Enter Job Description",
        height=300,
        placeholder=(
            "Paste the complete job description here..."
        )
    )

    if st.button(
        "🚀 Predict Job Category"
    ):

        if not job_description.strip():

            st.warning(
                "Please enter a job description."
            )

        else:

            try:

                with st.spinner(
                    "Loading DistilBERT and predicting..."
                ):

                    category, confidence = (
                        predict_job_category(
                            job_description
                        )
                    )

                st.success(
                    f"🎯 Predicted Category: {category}"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Predicted Category",
                        category
                    )

                with col2:

                    st.metric(
                        "Confidence",
                        f"{confidence:.2f}%"
                    )

                st.progress(
                    min(
                        int(confidence),
                        100
                    )
                )

            except Exception as e:

                st.error(
                    f"Prediction error: {e}"
                )


# =========================================================
# JOB MATCHING
# =========================================================

elif page == "🔎 Job Matching":

    st.header(
        "🔎 Resume–Job Matching"
    )

    # ---------------------------------------------
    # Resume check
    # ---------------------------------------------

    if "resume_text" not in st.session_state:

        st.warning(
            "Please upload your resume first "
            "from the Resume Upload section."
        )

    else:

        st.success(
            "Resume is ready for matching. ✅"
        )

        # -----------------------------------------
        # Job Description Input
        # -----------------------------------------

        st.subheader(
            "📝 Enter Job Description"
        )

        job_description = st.text_area(
            "Paste the job description you want to match",
            height=300,
            placeholder=(
                "Example: We are looking for a "
                "Digital Marketing Specialist..."
            )
        )

        top_n = st.slider(
            "Number of jobs to display",
            min_value=5,
            max_value=20,
            value=10
        )

        if st.button(
            "🔎 Find Suitable Jobs"
        ):

            if not job_description.strip():

                st.warning(
                    "Please enter a job description."
                )

            else:

                try:

                    with st.spinner(
                        "Analyzing job description and "
                        "finding related jobs..."
                    ):

                        (
                            matching_jobs,
                            predicted_category,
                            category_confidence
                        ) = find_matching_jobs(
                            st.session_state[
                                "resume_text"
                            ],
                            job_description,
                            top_n
                        )

                    # Save results
                    st.session_state[
                        "matching_jobs"
                    ] = matching_jobs

                    st.session_state[
                        "job_description"
                    ] = job_description

                    st.session_state[
                        "predicted_job_category"
                    ] = predicted_category

                    # ---------------------------------
                    # Display predicted category
                    # ---------------------------------

                    st.success(
                        "Job matching completed! ✅"
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "Job Category",
                            predicted_category
                        )

                    with col2:

                        st.metric(
                            "Category Confidence",
                            f"{category_confidence:.2f}%"
                        )

                    st.info(
                        f"Showing jobs related to "
                        f"**{predicted_category}**."
                    )

                    # ---------------------------------
                    # Display matching jobs
                    # ---------------------------------

                    st.subheader(
                        "💼 Suitable Jobs"
                    )

                    display_columns = []

                    possible_columns = [
                        "title",
                        "company_name",
                        "location",
                        "job_category",
                        "filter_category",
                        "match_percentage"
                    ]

                    for column in possible_columns:

                        if column in matching_jobs.columns:

                            display_columns.append(
                                column
                            )

                    if display_columns:

                        results = matching_jobs[
                            display_columns
                        ].copy()

                        results.insert(
                            0,
                            "Rank",
                            range(
                                1,
                                len(results) + 1
                            )
                        )

                        st.dataframe(
                            results,
                            use_container_width=True
                        )

                    else:

                        st.dataframe(
                            matching_jobs,
                            use_container_width=True
                        )

                except Exception as e:

                    st.error(
                        f"Matching error: {e}"
                    )


# =========================================================
# SKILL GAP ANALYSIS
# =========================================================

elif page == "📊 Skill Gap Analysis":

    st.header(
        "📊 Skill Gap Analysis"
    )

    if "resume_text" not in st.session_state:

        st.warning(
            "Please upload your resume first."
        )

    elif "matching_jobs" not in st.session_state:

        st.warning(
            "Please run Job Matching first."
        )

    else:

        matching_jobs = st.session_state[
            "matching_jobs"
        ]

        if st.button(
            "📊 Analyze Skill Gap"
        ):

            try:

                skill_column = get_skill_column(
                    matching_jobs
                )

                if skill_column is None:

                    st.warning(
                        "Skill information is not available "
                        "in job_embedding_metadata.csv."
                    )

                else:

                    # -----------------------------
                    # Resume skills
                    # -----------------------------

                    resume_skills = (
                        extract_resume_skills(
                            st.session_state[
                                "resume_text"
                            ],
                            matching_jobs
                        )
                    )

                    # -----------------------------
                    # Required skills
                    # -----------------------------

                    required_skills = []

                    for value in matching_jobs[
                        skill_column
                    ]:

                        skills = (
                            convert_to_skill_list(
                                value
                            )
                        )

                        required_skills.extend(
                            skills
                        )

                    required_skills = list(
                        set(
                            skill
                            for skill in required_skills
                            if skill
                        )
                    )

                    # -----------------------------
                    # Missing skills
                    # -----------------------------

                    resume_skill_lower = {
                        skill.lower()
                        for skill in resume_skills
                    }

                    missing_skills = []

                    for skill in required_skills:

                        if (
                            skill.lower()
                            not in resume_skill_lower
                        ):

                            missing_skills.append(
                                skill
                            )

                    missing_skills = sorted(
                        list(
                            set(
                                missing_skills
                            )
                        )
                    )

                    # Save
                    st.session_state[
                        "missing_skills"
                    ] = missing_skills

                    # -----------------------------
                    # Display
                    # -----------------------------

                    st.subheader(
                        "👤 Your Skills"
                    )

                    if resume_skills:

                        st.write(
                            ", ".join(
                                resume_skills
                            )
                        )

                    else:

                        st.info(
                            "No matching skills were detected."
                        )

                    st.subheader(
                        "❌ Missing Skills"
                    )

                    if missing_skills:

                        for skill in missing_skills:

                            st.write(
                                f"• {skill}"
                            )

                    else:

                        st.success(
                            "No skill gaps detected! 🎉"
                        )

            except Exception as e:

                st.error(
                    f"Skill gap analysis error: {e}"
                )


# =========================================================
# SKILL RECOMMENDATIONS
# =========================================================

elif page == "💡 Skill Recommendations":

    st.header(
        "💡 Skill Recommendations"
    )

    if "missing_skills" not in st.session_state:

        st.warning(
            "Please complete Skill Gap Analysis first."
        )

    else:

        missing_skills = st.session_state[
            "missing_skills"
        ]

        if missing_skills:

            st.write(
                "Based on your skill gap analysis, "
                "consider developing these skills:"
            )

            for index, skill in enumerate(
                missing_skills,
                start=1
            ):

                st.write(
                    f"### {index}. {skill}"
                )

                st.write(
                    "Consider learning this skill "
                    "through practical projects, "
                    "courses, documentation, and "
                    "hands-on practice."
                )

        else:

            st.success(
                "Your resume already covers the "
                "identified job requirements. 🎉"
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AI-Powered Job Market Intelligence & "
    "Skill Gap Analysis System | "
    "Python | TensorFlow | Hugging Face | "
    "Sentence-BERT | Streamlit"
)